"""PDF を1ページずつ画像化して DeepSeek-OCR-2 でOCRし、1つの Markdown に結合する。

    uv run python ocr_pdf.py --pdf "C:/Users/livec/Downloads/pdfbook-out/省察.pdf"
    uv run python ocr_pdf.py --pdf "...省察.pdf" --max-pages 1   # 見開き1枚だけ試す

特徴:
- モデルは一度だけロードし、ページをループ処理する。infer(eval_mode=True) でクリーンな
  テキストを取得し、結合 Markdown に逐次追記（途中終了しても部分結果が残る）。
- 無限反復を検知して生成を強制カットする（repetition.py）。
- 見開き(横長)ページは左右に分割し、縦書き右→左の順でOCRできる（--split auto)。
"""

import argparse
import os
import re
import time

import fitz  # pymupdf
import torch

from main import MODES, load_model
from preprocess import extract_page_images, pix_to_image, preprocess_for_ocr
from repetition import install_repetition_stopper, trim_repeated_tail

# grounding タグ（<|ref|>...<|/ref|><|det|>...<|/det|>）除去用
_REF = re.compile(r"<\|ref\|>.*?<\|/ref\|>", re.DOTALL)
_DET = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)

# CJK（ひらがな・カタカナ・漢字・全角記号）の間のスペースだけ除去する。
# Free OCR は縦書きを1文字ずつ空白区切りで出すことがあるため。英数の間の空白は残す。
_CJK = r"　-〿぀-ヿ㐀-鿿＀-￯"
_CJK_SPACE = re.compile(rf"(?<=[{_CJK}])[ \t]+(?=[{_CJK}])")


def clean_text(text: str) -> str:
    text = _REF.sub("", text)
    text = _DET.sub("", text)
    text = _CJK_SPACE.sub("", text)  # CJK 文字間の空白を詰める
    text = re.sub(r"\n{3,}", "\n\n", text)  # 連続空行を圧縮
    return text.strip()


def page_regions(page, split_mode: str):
    """OCR 対象の (ラベル, クリップ矩形) を返す。見開きは右→左の順。"""
    r = page.rect
    do_split = split_mode == "on" or (split_mode == "auto" and r.width > r.height)
    if not do_split:
        return [("", None)]
    cx = (r.x0 + r.x1) / 2
    right = fitz.Rect(cx, r.y0, r.x1, r.y1)
    left = fitz.Rect(r.x0, r.y0, cx, r.y1)
    return [("R", right), ("L", left)]  # 縦書きは右ページが先


def page_units(doc, i: int, args):
    """ページ i の OCR 対象を (ラベル, PIL画像) のリストで返す。

    source=native: 埋め込み画像をネイティブ解像度で取り出す（高解像→gundam で崩壊しにくい）。
    source=render: ページを dpi で描画し、見開きは左右に分割する（従来方式）。
    """
    if args.source == "native":
        imgs = extract_page_images(doc, i - 1)
        labels = "RL" if len(imgs) == 2 else [str(j) for j in range(len(imgs))]
        return [
            (f"{i}{('-' + labels[j]) if len(imgs) > 1 else ''}", im) for j, im in enumerate(imgs)
        ]
    page = doc[i - 1]
    units = []
    for suffix, clip in page_regions(page, args.split):
        im = pix_to_image(page.get_pixmap(dpi=args.dpi, clip=clip))
        units.append((f"{i}{('-' + suffix) if suffix else ''}", im))
    return units


def main():
    ap = argparse.ArgumentParser(description="OCR a whole PDF with DeepSeek-OCR-2")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default="./out_pdf")
    ap.add_argument("--quant", choices=["none", "4bit"], default="none", help="none=bf16(高精度)")
    ap.add_argument("--mode", choices=list(MODES), default="gundam")
    ap.add_argument(
        "--source", choices=["native", "render"], default="native", help="native=埋め込み画像を使用"
    )
    ap.add_argument("--dpi", type=int, default=300, help="render時のレンダリング解像度")
    ap.add_argument(
        "--split", choices=["auto", "on", "off"], default="auto", help="render時の見開き分割"
    )
    ap.add_argument(
        "--crop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="本文領域の自動切り出し",
    )
    ap.add_argument(
        "--binarize", action=argparse.BooleanOptionalAction, default=False, help="二値化"
    )
    ap.add_argument("--prompt", default="<image>\nFree OCR.")
    ap.add_argument("--start", type=int, default=1, help="開始ページ(1始まり)")
    ap.add_argument("--max-pages", type=int, default=0, help="OCRするページ数上限(0=全部)")
    args = ap.parse_args()

    book = os.path.splitext(os.path.basename(args.pdf))[0]
    workdir = os.path.join(args.out, book)
    pagedir = os.path.join(workdir, "pages")
    rawdir = os.path.join(workdir, "raw")
    os.makedirs(pagedir, exist_ok=True)
    os.makedirs(rawdir, exist_ok=True)

    doc = fitz.open(args.pdf)
    n = doc.page_count
    start = max(1, args.start)
    end = n if args.max_pages == 0 else min(n, start + args.max_pages - 1)
    base_size, image_size, crop_mode = MODES[args.mode]
    print(f"[pdf] {book}: {n}p / OCR {start}..{end}")
    print(f"[cfg] source={args.source} mode={args.mode} quant={args.quant}")
    print(f"[cfg] crop={args.crop} binarize={args.binarize} dpi={args.dpi} split={args.split}")

    print("[load] モデルロード中...")
    tokenizer, model = load_model("gpu", args.quant)
    install_repetition_stopper(model)  # 無限反復を生成中に強制カット
    print("[load] 完了")

    combined = os.path.join(workdir, f"{book}.md")
    # --start > 1 はレジューム扱い: 既存 md を残して追記する（先頭からは新規作成）
    if args.start > 1 and os.path.exists(combined):
        print(f"[resume] {combined} に追記（page {start} から）")
    else:
        with open(combined, "w", encoding="utf-8") as cf:
            cf.write(f"# {book}\n")

    t0 = time.time()
    for i in range(start, end + 1):
        for label, im in page_units(doc, i, args):
            stem = f"page_{label.replace('-', '_')}"
            if args.crop or args.binarize:
                im = preprocess_for_ocr(im, do_crop=args.crop, do_binarize=args.binarize)
            img_path = os.path.join(pagedir, f"{stem}.png")
            im.save(img_path)

            ts = time.time()
            raw = (
                model.infer(
                    tokenizer,
                    prompt=args.prompt,
                    image_file=img_path,
                    output_path=workdir,  # infer が makedirs するため有効パスを渡す
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=False,
                    eval_mode=True,
                )
                or ""
            )
            clean = trim_repeated_tail(clean_text(raw))

            with open(os.path.join(rawdir, f"{stem}.mmd"), "w", encoding="utf-8") as rf:
                rf.write(raw)
            with open(combined, "a", encoding="utf-8") as cf:
                cf.write(f"\n\n---\n\n## p.{label}\n\n{clean}\n")

            torch.cuda.empty_cache()
            cut = " [CUT]" if len(clean) < len(raw.rstrip()) else ""
            print(f"[page {label}] {len(clean)} chars in {time.time() - ts:.1f}s{cut}", flush=True)

    print(f"[done] {combined}  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

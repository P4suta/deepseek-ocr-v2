"""OCR 結合md を、構造を保った Markdown に汎用的に整形する後処理。

特定の本に依存しない汎用ロジック:
- 純数字行＝ノンブル、繰り返す短い見出し行＝柱(ランニングヘッダ) を除去
- 空行ブロック単位で分類: 見出し / リスト / 表 / 本文
- 見出しは正規化(前後のノンブル除去)して初出だけ ## に、重複(柱)は捨てる
- 本文はリフローして段落化。参照番号 [14]「2」は脚注参照 [^14] として保持し段落分割
- 反復残渣の圧縮、崩壊ページの検出・印付け

入力: out_pdf/<book>/<book>.md   出力: out_pdf/<book>/<book>.clean.md

    PYTHONUTF8=1 uv run python cleanup.py --book "<book-name>"
"""

import argparse
import os
import re
from collections import Counter

SECTION_SPLIT = re.compile(r"\n\n---\n\n## p\.([0-9A-Za-z-]+)\n\n")
NUM_LINE = re.compile(r"^\s*\d{1,4}\s*$")  # ノンブル/余白の行番号
MD_HEADING = re.compile(r"^#{1,6}\s+(.*\S)\s*$")  # モデルが付けた見出し
LIST_ITEM = re.compile(r"^\s*(?:[・\-•*‣◦]|\d{1,2}[.)、])\s+")  # 箇条書き
SECTION_REF = re.compile(r"\[\s*(\d{1,3})\s*\]|[「『](\d{1,3})[」』]")  # 節/参照番号 → [^N]
REPEAT = re.compile(r"(.{6,}?)\1+")  # 直後に繰り返される句(反復崩壊)
EDGE_NUM = re.compile(r"^\s*\d{1,4}\s*|\s*\d{1,4}\s*$")  # 見出し前後のノンブル
SENT_END = "。．！？!?）」』,，."  # 文末記号(これで終わる行は見出しでなく本文とみなす)

# CJK(かな・漢字・和文約物)に隣接する行内アラビア数字＝ノンブル残渣
_CJK = r"一-鿿぀-ヿ、。・（）"
INLINE_NUM = re.compile(rf"(?<=[{_CJK}])\d{{1,4}}|\d{{1,4}}(?=[{_CJK}])")
HAS_WORD = re.compile(r"[一-鿿぀-ヿA-Za-z0-9]")  # 文字を含むか(ゴミ判定用)

HEADING_MAX = 25  # これ以下の長さの非文末・単一行は見出し候補
_PARA = "\x00"  # 段落分割用センチネル


def split_blocks(body: str) -> list[str]:
    return [b for b in re.split(r"\n\s*\n", body.strip()) if b.strip()]


def heading_key(s: str) -> str:
    """見出しの正規化: 前後のノンブルを除いた本体。柱の重複排除キーに使う。"""
    return EDGE_NUM.sub("", s).strip()


def is_cjk(ch: str) -> bool:
    return bool(ch) and ord(ch) > 0x2E7F  # かな・漢字・全角記号など(ラテン/数字より上)


def join_soft(lines: list[str]) -> str:
    """ソフト改行を連結。CJK同士は空白なし、ラテン境界は空白を入れて結合。"""
    parts = [ln.strip() for ln in lines if ln.strip()]
    if not parts:
        return ""
    out = parts[0]
    for p in parts[1:]:
        out += ("" if is_cjk(out[-1]) and is_cjk(p[0]) else " ") + p
    return out


def is_heading_line(s: str) -> bool:
    """短く・文末記号で終わらず・数字/箇条書きでない・文字を含む単一行＝見出しらしい。"""
    return (
        0 < len(s) <= HEADING_MAX
        and s[-1] not in SENT_END
        and not NUM_LINE.match(s)
        and not LIST_ITEM.match(s)
        and bool(HAS_WORD.search(s))  # ``` や ─── などの記号だけの行を除外
    )


def collapse_repeats(text: str) -> str:
    return REPEAT.sub(r"\1", text)


def collapse_dupes(text: str) -> str:
    out: list[str] = []
    for sent in re.split(r"(?<=。)", text):
        if out and sent.strip() and sent.strip() == out[-1].strip():
            continue
        out.append(sent)
    return "".join(out)


def is_repetitive(text: str) -> bool:
    # 文字多様性は日本語だと正常でも低いので使わない。同一n-gramの多数回出現のみで判定。
    n = len(text)
    if n < 40:
        return False
    grams = [text[i : i + 12] for i in range(0, n - 12, 3)]
    return bool(grams) and Counter(grams).most_common(1)[0][1] >= 6


def is_suspicious(body: str, min_chars: int) -> bool:
    blocks = split_blocks(body)
    if any(len(b.split("\n")) == 1 and is_heading_line(heading_key(b.strip())) for b in blocks):
        return False  # 見出しを含む=タイトル/章頭ページは正規に短いことがある
    text = collapse_repeats("".join(ln.strip() for b in blocks for ln in b.split("\n")))
    return len(text) < min_chars or is_repetitive(text)


def html_table_to_md(block: str) -> str:
    rows = re.findall(r"<tr>(.*?)</tr>", block, re.DOTALL)
    md = []
    for i, row in enumerate(rows):
        cells = [re.sub(r"<[^>]*>", "", c).strip() for c in re.split(r"</t[dh]>", row) if c.strip()]
        if not cells:
            continue
        md.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(md)


def text_to_paragraphs(text: str, refs: list[str]) -> list[str]:
    """連結済み本文をノンブル除去→反復圧縮し、節番号位置で段落分割。"""

    def to_ref(m: re.Match) -> str:  # [14]「2」→ 段落区切り + 脚注参照 [^N]
        n = m.group(1) or m.group(2)
        refs.append(n)
        return f"{_PARA}[^{n}]"

    text = SECTION_REF.sub(to_ref, text)  # INLINE_NUM より前(全角「2」の数字を守る)
    text = INLINE_NUM.sub("", text)  # 行内ノンブル除去([^N] は安全)
    paras = []
    for piece in text.split(_PARA):
        piece = collapse_dupes(collapse_repeats(piece.strip()))
        if piece:
            paras.append(piece)
    return paras


def heading_candidate(lines: list[str]) -> str | None:
    """単一行ブロックが見出し候補なら、正規化したキーを返す。

    モデルが付けた `#` 行でも、本文が同一行に続いて長い場合は見出しにしない
    （is_heading_line の長さ判定を必ず通す）。
    """
    if len(lines) != 1:
        return None
    m = MD_HEADING.match(lines[0])
    key = heading_key(m.group(1) if m else lines[0])
    return key if is_heading_line(key) else None


def build(sections: list[tuple[str, str]], min_chars: int):
    seen_headers: set[str] = set()
    refs: list[str] = []
    flagged: list[str] = []
    blocks: list[tuple] = []
    buf: list[str] = []  # ページを跨いで蓄積する本文(段落は flush 時に切る)

    def flush_body() -> None:
        if buf:
            for para in text_to_paragraphs(join_soft(buf), refs):
                blocks.append(("p", para))
            buf.clear()

    for label, body in sections:
        susp = is_suspicious(body, min_chars)
        if susp:
            flush_body()
            flagged.append(label)
            blocks.append(("flag", label))
        for blk in split_blocks(body):
            # 先にブロック内の純数字行(ノンブル/余白行番号)を落とす
            lines = [ln.strip() for ln in blk.split("\n") if ln.strip() and not NUM_LINE.match(ln)]
            if not lines:
                continue
            cand = heading_candidate(lines)
            if cand is not None:
                # 初出の見出し候補＝見出し(本文を切る)。既出＝柱(furniture)なので本文を割らず破棄。
                if cand not in seen_headers:
                    flush_body()
                    seen_headers.add(cand)
                    blocks.append(("h", cand))
                continue
            if all(LIST_ITEM.match(ln) for ln in lines):
                flush_body()
                blocks.append(("list", [LIST_ITEM.sub("", ln).strip() for ln in lines]))
                continue
            if "<table" in blk:
                flush_body()
                blocks.append(("table", html_table_to_md(blk)))
                continue
            # 本文は蓄積(ページを跨いで連結)。長い # 行が本文に落ちた場合は先頭の # を除去
            buf.append(join_soft([re.sub(r"^#{1,6}\s+", "", ln) for ln in lines]))
        if susp:
            flush_body()
            blocks.append(("flagend", label))
    flush_body()

    return blocks, flagged, refs


def render(blocks: list[tuple]) -> str:
    out = []
    for b in blocks:
        if b[0] == "h":
            out.append("## " + b[1])
        elif b[0] == "p":
            out.append(b[1])
        elif b[0] == "list":
            out.append("\n".join("- " + it for it in b[1]))
        elif b[0] == "table":
            out.append(b[1])
        elif b[0] == "flag":
            out.append(f"> ⚠️ 認識崩壊の可能性 — 要再OCR (p.{b[1]})")
        elif b[0] == "flagend":
            out.append(f"> ⚠️ (p.{b[1]} ここまで)")
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Clean OCR markdown into structured Markdown")
    ap.add_argument("--book", required=True, help="out_pdf/<book>/<book>.md を入力に")
    ap.add_argument("--out-root", default="./out_pdf")
    ap.add_argument("--md", help="入力mdを直接指定（省略時は out_pdf/<book>/<book>.md）")
    ap.add_argument("--min-chars", type=int, default=90, help="句反復を畳んでこれ未満なら崩壊候補")
    args = ap.parse_args()

    md_path = args.md or os.path.join(args.out_root, args.book, f"{args.book}.md")
    with open(md_path, encoding="utf-8") as f:
        text = f.read()

    parts = SECTION_SPLIT.split(text)
    sections = list(zip(parts[1::2], parts[2::2], strict=True))
    print(f"[in] {md_path}: {len(sections)} page-sections")

    blocks, flagged, refs = build(sections, args.min_chars)
    doc = [f"# {args.book}", render(blocks)]
    if flagged:
        doc.append("## 要再OCRページ（崩壊/短文の疑い）\n" + "\n".join(f"- p.{x}" for x in flagged))
    if refs:  # 脚注定義(節番号マーカーなので内容は番号のみ。重複番号は1つに集約)
        uniq = sorted(set(refs), key=int)
        doc.append("\n".join(f"[^{n}]: 原文の節番号 {n}" for n in uniq))

    out_path = os.path.join(args.out_root, args.book, f"{args.book}.clean.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(doc) + "\n")
    print(f"[out] {out_path}: flagged {len(flagged)} pages, {len(set(refs))} footnote refs")


if __name__ == "__main__":
    main()

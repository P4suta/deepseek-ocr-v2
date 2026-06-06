"""OCR 前処理: 本文領域の自動切り出し・コントラスト調整・任意の二値化。

ページから余白・ノンブル(ページ番号)・柱(ヘッダ)を落として本文ブロックだけにすると、
モデルへ渡る際に文字が相対的に大きく写り、認識精度と反復崩壊への耐性が上がる。
インク(暗画素)の行・列投影から本文ブロックの外接矩形を推定して切り出す。
"""

import io

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def pix_to_image(pix) -> Image.Image:
    """pymupdf Pixmap を PIL.Image(RGB) に変換する。"""
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def extract_page_images(doc, page_index: int) -> list[Image.Image]:
    """PDF ページに埋め込まれた画像を、ネイティブ解像度で読み順(右→左)に返す。

    この本は1見開き=2枚の高解像度(2480x3496)二値画像。再レンダリングで潰すより
    埋め込み画像をそのまま使う方が解像度が高く、gundam のタイル分割で文字が大きく保たれ
    OCR が崩壊しにくい。縦書きなので配置 x の大きい(右)ページを先に読む。
    """
    page = doc[page_index]
    items = []
    for info in page.get_images(full=True):
        xref = info[0]
        rects = page.get_image_rects(xref)
        x0 = rects[0].x0 if rects else 0.0
        ext = doc.extract_image(xref)
        img = Image.open(io.BytesIO(ext["image"])).convert("RGB")
        items.append((x0, img))
    items.sort(key=lambda t: -t[0])  # 右(x0 大)が先 = 縦書き RTL
    return [img for _, img in items]


def _ink_profiles(gray: Image.Image):
    a = np.asarray(gray, dtype=np.float32)
    ink = 255.0 - a  # 暗い=インク を高い値に
    thr = ink.mean() + 0.5 * ink.std()  # 背景ノイズ除去の閾値
    mask = ink > thr
    return mask.sum(axis=1), mask.sum(axis=0)  # 行プロファイル, 列プロファイル


def crop_text_region(img: Image.Image, frac: float = 0.12, pad_frac: float = 0.015) -> Image.Image:
    """インク投影から本文ブロックの bbox を求めて切り出す。

    各方向で「最大密度の frac 倍」を超える行/列を本文とみなし、その最初〜最後の範囲だけ残す。
    密度の低い外周(余白)や、本文から離れた疎なノンブル/柱の帯は自然に外れる。
    """
    gray = ImageOps.grayscale(img)
    rows, cols = _ink_profiles(gray)
    if rows.max() == 0 or cols.max() == 0:
        return img
    r = np.where(rows > frac * rows.max())[0]
    c = np.where(cols > frac * cols.max())[0]
    if len(r) == 0 or len(c) == 0:
        return img
    w, h = img.width, img.height
    px, py = int(pad_frac * w), int(pad_frac * h)
    box = (
        max(0, int(c[0]) - px),
        max(0, int(r[0]) - py),
        min(w, int(c[-1]) + px + 1),
        min(h, int(r[-1]) + py + 1),
    )
    return img.crop(box)


def binarize(img: Image.Image, bias: int = 0) -> Image.Image:
    """Otsu 法で白地・黒文字に二値化する。"""
    gray = np.asarray(ImageOps.grayscale(img), dtype=np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    w0 = np.cumsum(hist)
    cum_mean = np.cumsum(hist * np.arange(256))
    total_mean = cum_mean[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = cum_mean / w0
        mu1 = (total_mean - cum_mean) / (w0[-1] - w0)
        var_between = w0 * (w0[-1] - w0) * (mu0 - mu1) ** 2
    t = int(np.nanargmax(var_between)) + bias
    out = (gray > t).astype(np.uint8) * 255
    return Image.fromarray(out, mode="L").convert("RGB")


def split_columns(
    img: Image.Image,
    thr_frac: float = 0.12,
    merge_gap_frac: float = 0.6,
    min_w_frac: float = 0.3,
    pad: int = 6,
) -> list[Image.Image]:
    """縦書きページを列(縦の行)ごとに分割し、右→左の読み順で列画像のリストを返す。

    列ごとのインク量(縦方向の和)のプロファイルから、本文列の x 範囲を検出する。
    小さな隙間(列内のばらつき)で割れた塊は併合し、細すぎる塊(ルビや汚れ)は捨てる。
    1 列ずつ OCR に渡すと 1 回の生成が短くなり、反復崩壊を避けて最後まで読みやすい。
    """
    gray = ImageOps.grayscale(img)
    ink = 255.0 - np.asarray(gray, dtype=np.float32)
    thr = ink.mean() + 0.5 * ink.std()
    col = (ink > thr).sum(axis=0).astype(np.float32)  # 列(x)ごとのインク量
    if col.max() <= 0:
        return [img]

    is_text = col > max(1.0, thr_frac * col.max())
    runs: list[list[int]] = []
    x, w = 0, len(col)
    while x < w:
        if is_text[x]:
            x0 = x
            while x < w and is_text[x]:
                x += 1
            runs.append([x0, x])
        else:
            x += 1
    if not runs:
        return [img]

    med = float(np.median([r[1] - r[0] for r in runs]))
    merged = [runs[0]]
    for r in runs[1:]:
        if r[0] - merged[-1][1] < merge_gap_frac * med:  # 近い塊は同じ列として併合
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    h = img.height
    cols = []
    for x0, x1 in merged:
        if (x1 - x0) < max(3, min_w_frac * med):  # 細すぎる塊は除外
            continue
        box = (max(0, x0 - pad), 0, min(img.width, x1 + pad), h)
        cols.append(img.crop(box))
    cols.reverse()  # 右→左（縦書きの読み順）
    return cols or [img]


def preprocess_for_ocr(
    img: Image.Image,
    do_crop: bool = True,
    do_binarize: bool = False,
    sharpen: bool = False,
    margin: int = 24,
) -> Image.Image:
    """切り出し → コントラスト調整 →(任意)シャープ化/二値化 → 白縁付与 の順で整える。"""
    if do_crop:
        img = crop_text_region(img)
    img = ImageOps.autocontrast(img.convert("RGB"), cutoff=1)
    if sharpen:  # 小さい文字の輪郭を立てる
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2))
    if do_binarize:
        img = binarize(img)
    if margin:
        img = ImageOps.expand(img, border=margin, fill="white")
    return img

"""検証用のサンプル文書画像 (sample.png) を生成する簡易スクリプト。

OCR が読み取りやすいよう、白背景に黒の印刷風テキストを配置する。
    uv run python make_sample.py
"""

from PIL import Image, ImageDraw, ImageFont

W, H = 1000, 1300
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)


def load_font(size: int):
    # Windows 標準フォントを順に試し、無ければデフォルト
    for name in ("arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


title_font = load_font(46)
head_font = load_font(30)
body_font = load_font(24)

lines = [
    (60, title_font, "Quarterly Sales Report"),
    (140, head_font, "Section 1: Overview"),
    (190, body_font, "This document summarizes the sales performance for Q2 2026."),
    (225, body_font, "Revenue increased by 18% compared to the previous quarter."),
    (300, head_font, "Section 2: Key Figures"),
    (350, body_font, "Region        Units Sold      Revenue (USD)"),
    (385, body_font, "North          1,240          124,000"),
    (420, body_font, "South            980           98,500"),
    (455, body_font, "East           1,510          151,200"),
    (490, body_font, "West             760           76,300"),
    (565, head_font, "Section 3: Notes"),
    (615, body_font, "- Online channel outperformed retail by a wide margin."),
    (650, body_font, "- Customer retention rate reached 92.4%."),
    (685, body_font, "- Next review meeting scheduled for July 15, 2026."),
]

for y, font, text in lines:
    draw.text((60, y), text, fill="black", font=font)

# 区切り線
draw.line((60, 120, W - 60, 120), fill="black", width=2)

img.save("sample.png")
print("sample.png を作成しました")

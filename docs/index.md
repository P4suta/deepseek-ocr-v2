# deepseek-ocr-v2

DeepSeek-OCR-2 をローカル（Windows / 6GB クラスのGPU）で動かし、PDFをOCRして構造を保った Markdown に整えるツール群。

## 構成

- **main.py** — 1枚の画像をOCR
- **ocr_pdf.py** — PDFを1ページずつOCRし、1つの Markdown に結合
- **cleanup.py** — OCR結果のノンブル・柱・崩れた改行を除去し、見出し付きの Markdown に整形

## クイックスタート

```powershell
uv sync
uv run python main.py --smoke                          # ロード確認（初回はモデル約6.3GBをDL）
uv run python ocr_pdf.py --pdf "C:/path/to/book.pdf"   # PDFを丸ごとOCR
uv run python cleanup.py --book "book"                 # Markdownに整形
```

詳しくは [使い方](usage.md)、設定と制約は [仕様と制約](internals.md) を参照。

## 必要環境

- Windows / NVIDIA GPU（VRAM 6GB〜） / Python 3.12（uv で取得）
- flash-attn は不要（`eager` で動作）

# deepseek-ocr-v2

**[DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) をローカル（Windows / 6GB クラスのGPU）で動かし、PDFを丸ごとOCRして、構造を保ったきれいな Markdown に整えるツール群です。**

---

## これは何？

3つの小さなツールで「画像 → OCR → 読めるテキスト」を完結させます。

- **`main.py`** — 1枚の画像を DeepSeek-OCR-2 でOCR（動作確認・お試し用）
- **`ocr_pdf.py`** — PDFを1ページずつOCRして1つのMarkdownに結合（本命）
- **`cleanup.py`** — OCR結果のノンブル・柱・崩れた改行を機械的に除去し、見出し付きの整った Markdown に変換（汎用）

## 特徴

- :material-flash: **flash-attn 不要** — `_attn_implementation="eager"` で起動し、Windowsでのビルド地獄を回避。
- :material-memory: **6GB GPU でも動く** — bf16 でも 4bit でも実用速度。Sysmem Fallback も活用。
- :material-book-open-page-variant: **縦書き日本語の全文OCR** — ネイティブ高解像度を `gundam` タイル分割で渡すことで、密な縦書きページも崩壊させずに読み切る。
- :material-broom: **無限反復の自動カット** — 生成中の degeneration を n-gram で検知して停止。
- :material-language-markdown: **汎用 Markdown 整形** — 特定の本に依存しないクリーンアップで、見出し・段落・脚注参照を復元。

## クイックスタート

```powershell
# 1. 環境構築（torch CUDA 版などを取得）
uv sync

# 2. 動作確認（モデル約6.3GBを初回DL）
uv run python main.py --smoke

# 3. PDFを丸ごとOCR（ネイティブ画像 + gundam + bf16）
uv run python ocr_pdf.py --pdf "C:/path/to/book.pdf"

# 4. 読めるMarkdownに整える
uv run python cleanup.py --book "book"
```

## 経緯（なぜ動くのか）

当初、密な縦書き日本語ページは途中から同じ語句を繰り返す **degeneration（崩壊）** を起こしていました。原因は画質でも量子化でもなく、**低解像度で文字が潰れていたこと**。

PDFに埋め込まれた **ネイティブ高解像度画像（約2480×3496・二値）** を抽出し、`gundam`（crop_mode=True）でタイル分割して渡すことで、文字の大きさが保たれ、ページ全文を崩壊なくOCRできるようになりました。詳しくは [仕組みと知見](internals.md) を参照してください。

---

!!! note "対象環境での検証"
    NVIDIA GeForce RTX 3060 Laptop（専用VRAM 6GB）/ Windows 11 / Python 3.12（uv管理）で検証しています。

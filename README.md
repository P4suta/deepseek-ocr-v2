# deepseek-ocr-v2

[![CI](https://github.com/P4suta/deepseek-ocr-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/P4suta/deepseek-ocr-v2/actions/workflows/ci.yml)
[![Docs](https://github.com/P4suta/deepseek-ocr-v2/actions/workflows/docs.yml/badge.svg)](https://P4suta.github.io/deepseek-ocr-v2/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)（3B, DeepEncoder V2）をローカル（Windows / RTX 3060 6GB クラス）で動かすための最小構成。

📖 **ドキュメント: <https://P4suta.github.io/deepseek-ocr-v2/>**

## 特徴 / 前提

- **flash-attn 不要**。`_attn_implementation="eager"` で起動する（Windows でのビルド地獄を回避）。
- モデル本体は bf16 で約 6.3GB。専用 VRAM 6GB は少し超えるが、NVIDIA の **CUDA Sysmem Fallback**（標準 ON）で共有メモリへ自動退避するためロード・実行は可能。ただし退避分は低速。
- **実用速度の本命は 4bit 量子化**（重み約 2GB が専用 VRAM に収まる）。

## セットアップ

```powershell
uv sync
```

`torch` / `torchvision` は PyPI の CPU 版ではなく PyTorch の CUDA (cu124) index から取得する設定（`pyproject.toml` の `[tool.uv.sources]`）。

## 使い方

```powershell
# ロード確認だけ（推論せず終了。初回はモデル約6.3GBをDL）
uv run python main.py --smoke

# 検証用サンプル画像を生成
uv run python make_sample.py

# bf16 GPU で実行（公式どおり・遅め）
uv run python main.py --image sample.png

# 実用速度の本命: 4bit 量子化
uv run python main.py --image sample.png --quant 4bit

# 最終手段: CPU 実行
uv run python main.py --image sample.png --device cpu --mode tiny
```

結果は `--out`（既定 `./out`）に Markdown とグラウンディング可視化画像で保存される。

### 主なオプション

| オプション | 既定 | 説明 |
|---|---|---|
| `--image` | （必須） | OCR 対象の画像パス |
| `--prompt` | `<image>\n<\|grounding\|>Convert the document to markdown.` | プロンプト |
| `--device` | `gpu` | `gpu` / `cpu` |
| `--quant` | `none` | `none`(bf16) / `4bit` |
| `--mode` | `small` | `small`(768) / `base`(1024) / `gundam`(動的) ※v2 は 768/1024 のみ対応 |
| `--out` | `./out` | 結果の保存先 |
| `--smoke` | - | ロード確認だけして終了 |

## PDF を丸ごと OCR（`ocr_pdf.py`）

```powershell
uv run python ocr_pdf.py --pdf "C:/path/to/book.pdf"
uv run python ocr_pdf.py --pdf "...book.pdf" --max-pages 5   # 先頭5ページだけ試す
```

- モデルを一度だけロードし、ページをループ。結果は `out_pdf/<book>/<book>.md` に逐次追記（途中終了でも部分結果が残る）。
- 既定は **`--source native`**（PDF 埋め込み画像をネイティブ解像度で抽出）＋ **`--mode gundam`**＋ **bf16**。
- 見開き(横長)の本は左右2ページに分割し、縦書きは右→左の順で処理。

### 縦書き日本語の重要な知見

密な縦書き日本語ページは、低解像度で渡すと途中から**同じ語句を繰り返す崩壊(degeneration)**を起こす。原因は画質や量子化ではなく、**文字が小さく潰れること**。対策:

- **ネイティブ高解像度画像を `gundam`(crop_mode=True) でタイル分割**して渡す → 文字が大きく保たれ、ページ全文を崩壊なく読める（これが最重要）。
- 無限反復は `repetition.py` の n-gram ストッパーで生成中に強制カット＋後処理で末尾を整える。
- 列ごとに分割して渡すのは逆効果（LLM が全体文脈で補完できず、疎な列で幻覚する）。
- `repetition_penalty` は幻覚を招くため既定 OFF。

## OCR結果のクリーンアップ（`cleanup.py`）

OCRした結合md（`out_pdf/<book>/<book>.md`）を、構造を保った **Markdown** に機械的に整える汎用後処理（特定の本に依存しない）。

```powershell
PYTHONUTF8=1 uv run python cleanup.py --book "<book-name>"
# → out_pdf/<book>/<book>.clean.md
```

やること（汎用ロジック）:
- **ノンブル・余白の行番号・柱（ランニングヘッダ）を除去**。純数字行、行内ノンブル、繰り返す短い見出しの重複（柱）を機械的に判定。
- **空行ブロック単位で分類**し、短い非文末行を `##` 見出しに、本文はページを跨いでリフロー＝段落化。
- **節番号「2」`[14]` は脚注参照 `[^14]` として保持**し、その位置で段落を分割。
- **反復残渣を圧縮**（句の即時反復＋隣接重複文）。
- **崩壊/幻覚の疑いページを検出**して印＋末尾に一覧（自動削除はしない）。

設計メモ:
- 固有名（章タイトル等）はハードコードせず、「ページ跨ぎで繰り返す短い行＝柱」「初出のみ見出し、重複は柱として破棄」で汎用化。
- 見出し推定はヒューリスティック（短い誤OCR断片や柱の誤字を拾うことがある）。真の段落境界（字下げ）はOCR出力に残らないため、見出し＋節番号での粗い段落化に留まる。文字レベルのOCR誤りは対象外。
- 表・リストは入力にあれば Markdown 化するが、`Free OCR` 出力には無いため通常は付かない（実構造が要るなら Markdown プロンプトで再OCR）。`--min-chars` で崩壊検出の閾値を調整可能。

## 開発

```powershell
uv run ruff check .    # lint
uv run ruff format .   # format
uv run ty check .      # 型チェック
```

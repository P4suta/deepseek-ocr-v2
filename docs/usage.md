# 使い方

## セットアップ

```powershell
uv sync
```

`torch` / `torchvision` は PyPI の CPU 版ではなく PyTorch の CUDA (cu124) index から取得する設定（`pyproject.toml` の `[tool.uv.sources]`）。

## 単一画像のOCR（`main.py`）

```powershell
uv run python main.py --smoke                              # ロード確認のみ
uv run python make_sample.py                               # 検証用サンプル画像を生成
uv run python main.py --image sample.png                   # bf16 GPU
uv run python main.py --image sample.png --quant 4bit      # 4bit（VRAM内に収め高速）
uv run python main.py --image sample.png --device cpu --mode tiny  # 最終手段
```

| オプション | 既定 | 説明 |
|---|---|---|
| `--image` | （必須） | OCR対象の画像 |
| `--device` | `gpu` | `gpu` / `cpu` |
| `--quant` | `none` | `none`(bf16) / `4bit` |
| `--mode` | `small` | `small`(768) / `base`(1024) / `gundam`(動的)。v2は768/1024のみ対応 |
| `--smoke` | - | ロード確認だけして終了 |

## PDFを丸ごとOCR（`ocr_pdf.py`）

```powershell
uv run python ocr_pdf.py --pdf "C:/path/to/book.pdf"
uv run python ocr_pdf.py --pdf "...book.pdf" --max-pages 5   # 先頭だけ試す
uv run python ocr_pdf.py --pdf "...book.pdf" --start 34       # 34ページ目から再開
```

- モデルは一度だけロードし、ページをループ。結果は `out_pdf/<book>/<book>.md` に**逐次追記**（途中終了でも部分結果が残る）。
- 既定は **`--source native`**（PDF埋め込み画像をネイティブ解像度で抽出）＋ **`--mode gundam`** ＋ **bf16**。
- 見開き（横長）の本は左右2ページに分割し、縦書きは右→左の順で処理。
- `--start N` でレジューム（既存mdに追記）。長い本やクラッシュ後の続行に。

## クリーンアップして Markdown 化（`cleanup.py`）

```powershell
PYTHONUTF8=1 uv run python cleanup.py --book "book"
# → out_pdf/book/book.clean.md
```

- ノンブル・余白の行番号・柱を除去、行内改行をリフロー、節番号 `[14]`「2」を脚注参照 `[^14]` として保持、反復残渣を圧縮、崩壊ページを検出して印付け。
- 特定の本に依存しない汎用ロジック（詳細は [仕組みと知見](internals.md)）。
- `--min-chars` で崩壊検出の閾値を調整可能。

## 開発

```powershell
uv run ruff check .     # lint
uv run ruff format .    # format
uv run ty check .       # 型チェック
uv run pytest           # cleanup の単体テスト
```

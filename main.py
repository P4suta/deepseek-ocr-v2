"""DeepSeek-OCR-2 をローカル(RTX 3060 6GB / Windows)で動かすための実行スクリプト。

使い方の例:
    # ロード確認だけ（推論せず終了。flash-attn 不要で起動できるかの確認）
    uv run python main.py --smoke

    # bf16 GPU で実行（公式どおり。6GB 超過分は共有メモリへ自動退避＝遅め）
    uv run python main.py --image sample.png

    # 実用速度の本命: 4bit 量子化で専用 VRAM 内に収めて高速化
    uv run python main.py --image sample.png --quant 4bit

    # 最終手段: CPU 実行（VRAM 不使用・遅い）
    uv run python main.py --image sample.png --device cpu --mode tiny
"""

import os

# torch を import する前に設定する必要がある（GPU メモリ断片化由来の OOM を緩和）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"

DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

OOM_HINT = (
    "メモリ不足です。`--quant 4bit` か `--mode tiny`、"
    "それでも厳しければ `--device cpu` を試してください。"
)

# 解像度モード -> (base_size, image_size, crop_mode)
# v2 の DeepEncoder は入力 768px(=144 トークン) か 1024px(=256 トークン) のみ対応。
# crop_mode=False では base_size==image_size が必須（トークン数整合のため）。
# 512/640 など他のサイズは UnboundLocalError(param_img) で落ちるので使わない。
MODES = {
    "small": (768, 768, False),  # 144 視覚トークン（最小・6GB 向けの既定）
    "base": (1024, 1024, False),  # 256 視覚トークン
    "gundam": (1024, 768, True),  # 動的クロップ（大きい文書向け・VRAM 多め）
}


def load_model(device: str, quant: str):
    """指定の方式でモデルとトークナイザをロードする。

    重要: `_attn_implementation="eager"` は 3 パス全てで必ず明示する。
    デコーダは実行時に config._attn_implementation から ATTENTION_CLASSES で
    attn クラスを選ぶが、この辞書には sdpa キーが無い。未指定だと transformers の
    既定 sdpa で KeyError、または flash_attention_2 で未 import の flash_attn を
    呼んで NameError になる。"eager" を渡せばデコーダ・視覚塔の両方を満たす。
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    common = {
        "_attn_implementation": "eager",  # flash-attn 不要
        "trust_remote_code": True,
        "use_safetensors": True,
    }

    if device == "gpu" and quant == "4bit":
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # torch_dtype=bfloat16 が重要: これが無いと非量子化部（embedding）は float16、
        # 視覚塔の conv は float32 のままで、テキスト埋め込みへ視覚特徴を埋め込む
        # masked_scatter で dtype 不一致になる。bfloat16 に揃えて回避する。
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            **common,
        )
        model = model.eval()
        # 注意: 4bit では .cuda().to(bfloat16) を呼ばない（device_map が配置済み・呼ぶとエラー）

    elif device == "gpu":
        model = AutoModel.from_pretrained(MODEL_NAME, **common)
        model = model.eval().cuda().to(torch.bfloat16)  # 公式どおり

    else:  # cpu
        model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, **common)
        model = model.eval()

    return tokenizer, model


def main():
    ap = argparse.ArgumentParser(description="Run DeepSeek-OCR-2 locally")
    ap.add_argument("--image", help="OCR 対象の画像パス")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="プロンプト")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--quant", choices=["none", "4bit"], default="none")
    ap.add_argument("--mode", choices=list(MODES), default="small")
    ap.add_argument("--out", default="./out", help="結果の保存先ディレクトリ")
    ap.add_argument("--smoke", action="store_true", help="ロード確認だけして終了")
    args = ap.parse_args()

    print(f"[load] model={MODEL_NAME} device={args.device} quant={args.quant}")
    print(f"[load] CUDA available: {torch.cuda.is_available()}")
    tokenizer, model = load_model(args.device, args.quant)
    print("[load] モデルのロード成功（flash-attn 不要・eager で起動）")

    if args.smoke:
        print("[smoke] OK")
        return

    if not args.image:
        ap.error("--image は必須です（--smoke 時を除く）")

    base_size, image_size, crop_mode = MODES[args.mode]
    os.makedirs(args.out, exist_ok=True)
    print(f"[infer] mode={args.mode} base={base_size} image={image_size} crop={crop_mode}")

    # CPU の場合、.infer() 内でハードコードされた .cuda() を無効化して入力を CPU に留める。
    # （forward 内に .to('cuda') 直書きが残っていればここでは拾えないため、その時は追加対応）
    patched_cuda = None
    if args.device == "cpu":
        patched_cuda = torch.Tensor.cuda
        torch.Tensor.cuda = lambda self, *a, **k: self  # noqa: E731  # ty: ignore[invalid-assignment]

    try:
        res = model.infer(
            tokenizer,
            prompt=args.prompt,
            image_file=args.image,
            output_path=args.out,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=True,
        )
    except torch.cuda.OutOfMemoryError:
        print(f"[error] CUDA out of memory。{OOM_HINT}")
        raise
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[error] {OOM_HINT}")
        raise
    finally:
        if patched_cuda is not None:
            torch.Tensor.cuda = patched_cuda

    print("[infer] 完了")
    if res is not None:
        print(res)
    print(f"[infer] 結果は {os.path.abspath(args.out)} に保存されました")


if __name__ == "__main__":
    main()

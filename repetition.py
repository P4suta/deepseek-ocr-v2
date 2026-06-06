"""無限反復（degeneration）の機械的検知と強制カット。

DeepSeek-OCR は不得手な入力（縦書き・低解像度など）で、良い冒頭のあと同じ語句や
文を延々と繰り返すことがある。これを 2 段構えで止める:

1. RepetitionStoppingCriteria … 生成中に「同じ n-gram が規定回数以上出現」したら
   generate を停止する。短い周期反復から、文単位の（多少揺れる）長いループまで検知。
   無駄な生成時間も節約できる。
2. trim_repeated_tail … 生成後テキストに残った反復末尾を 1 ブロックだけ残して切る。
"""

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class RepetitionStoppingCriteria(StoppingCriteria):  # ty: ignore[unsupported-base]
    """生成トークン列で同じ n-gram が max_count 回以上出現したら停止する。

    周期反復は当然、文単位の長い反復ループも、その内部の同一 n-gram が繰り返し現れる
    ことで検知できる（末尾に簡体字混入などの揺れがあっても、一致する部分で捕捉する）。
    n-gram を 10 程度にすると、表の繰り返し構造などでの誤検知も避けやすい。
    """

    def __init__(self, prompt_len: int, ngram: int = 10, max_count: int = 3):
        self.prompt_len = int(prompt_len)
        self.ngram = ngram
        self.max_count = max_count
        self.counts: dict[tuple[int, ...], int] = {}

    def __call__(self, input_ids, scores=None, **kwargs):
        seq = input_ids[0].tolist()
        n = len(seq)
        if n - self.prompt_len < self.ngram:  # まだ生成トークンが n-gram に満たない
            return torch.tensor([False], device=input_ids.device)
        ng = tuple(seq[n - self.ngram : n])  # 末尾 n-gram（全て生成領域内）
        c = self.counts.get(ng, 0) + 1
        self.counts[ng] = c
        return torch.tensor([c >= self.max_count], device=input_ids.device)


def install_repetition_stopper(model, repetition_penalty: float | None = None, **kw):
    """model.generate をラップし、毎回の生成に反復対策を差し込む。

    DeepSeek-OCR の .infer() は内部で self.generate(...) を固定引数で呼ぶため、
    ここで generate を包んで以下を注入する:
    - stopping_criteria: 反復ループに陥ったら生成中に「切る」（RepetitionStoppingCriteria）。
    - repetition_penalty: 反復を「起こしにくくする」確率的ペナルティ。ただし本タスク
      （縦書き日本語）では 1.15 でも幻覚的な続きを生むだけで精度が下がったため既定 OFF。
      横書き等で有効なケース用に引数としては残す。
    生成ごとに新しい判定器（カウンタは生成単位でリセット）を作る。
    """
    orig_generate = model.generate

    def wrapped(*args, **kwargs):
        ids = args[0] if args else kwargs.get("input_ids", kwargs.get("inputs"))
        prompt_len = ids.shape[1] if hasattr(ids, "shape") else 0
        existing = list(kwargs.get("stopping_criteria") or [])
        kwargs["stopping_criteria"] = StoppingCriteriaList(
            existing + [RepetitionStoppingCriteria(prompt_len, **kw)]
        )
        if repetition_penalty:
            kwargs.setdefault("repetition_penalty", repetition_penalty)
        return orig_generate(*args, **kwargs)

    model.generate = wrapped
    return model


def trim_repeated_tail(
    text: str, max_period: int = 120, min_repeats: int = 3, min_total: int = 24
) -> str:
    """末尾の周期反復を 1 ブロックだけ残して切り落とす（後処理の保険）。"""
    n = len(text)
    cut = n
    for period in range(1, min(max_period, n) + 1):
        block = text[n - period : n]
        if not block.strip():
            continue
        reps, idx = 1, n - period
        while idx - period >= 0 and text[idx - period : idx] == block:
            reps += 1
            idx -= period
        if reps >= min_repeats and reps * period >= min_total:
            cut = min(cut, idx + period)  # 1 ブロックだけ残す
    return text[:cut].rstrip()

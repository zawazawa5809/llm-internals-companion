#!/usr/bin/env python3
"""token_loop.py — LLM の自己回帰ループを「1トークンずつ」手元で覗く (llm-internals Part 1).

やること:
  1. tokenize-demo : トークン化が文字でも単語でもないことを観察 (strawberry / 日本語 / 空白 / 数字)
  2. loop          : tokenize -> generate_step -> 各ステップの top-k 確率と1トークンを print (cached decode)
  3. bench         : 生成トークン数 N と 累積 wall-time が線形であることを実測 (warmup + median)
  4. selftest      : MLX 非依存の純ロジック検証 (top-k 抽出・累積時間) — 実機なしで通る

設計メモ (記事 methodology に直結):
  - **cached decode で測る**: generate_step は内部で KV キャッシュを使う。よって per-token 時間は
    decode フェーズの単価であり、毎ステップ full-context を計算し直しているのではない。prefill
    (プロンプト一括処理) の時間は別物。bench は「最初のトークンを除いた」decode 区間で線形を見る。
  - **数値は環境依存**: tok/s も top-k 確率も「そのモデル・その量子化 (例 4bit)・その機種」の値。
    一般化しない。free != fast。所要時間 (モデル DL を含む) を明記すること。
  - **供給網の固定 (再現性)**: --model に revision 付き repo を渡し、run-meta に id/version/dtype を
    記録する。trust_remote_code は使わない。

Tested with: mlx-lm 0.x (generate_step が (token, logprobs) を yield する版). 正確な version は
README と run-meta に pin する。API が変わったら _generate_step_compat() を調整する。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, asdict


# --- MLX 非依存の純ロジック (selftest はここだけを検証し、実機なしで通る) ---------------------


def top_k_from_logprobs(logprobs, k: int = 5):
    """全語彙の対数確率ベクトルから top-k の (index, probability) を返す.

    logprobs: 1次元の array-like (numpy / mlx を numpy 化したもの)。log p(token) が入っている前提。
    返り値: [(token_id, prob), ...] を prob 降順で k 個。prob = exp(logprob)。

    純粋関数。mlx に依存しないので selftest で直接叩ける。
    """
    import math

    pairs = list(enumerate(float(x) for x in logprobs))
    pairs.sort(key=lambda t: t[1], reverse=True)
    return [(idx, math.exp(lp)) for idx, lp in pairs[:k]]


def linear_fit_r2(xs, ys) -> float:
    """単回帰の決定係数 R^2 を返す (累積 wall-time が N に線形かの指標). 純粋関数."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return 0.0
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


# --- run-meta (再現性: 環境とモデル素性を記録) -------------------------------------------------


@dataclass
class RunMeta:
    model: str
    mlx_lm_version: str
    python: str
    platform: str
    note: str = ""


def collect_run_meta(model: str) -> RunMeta:
    try:
        import mlx_lm  # noqa: WPS433

        ver = getattr(mlx_lm, "__version__", "unknown")
    except Exception:  # pragma: no cover - 実機以外
        ver = "not-installed"
    return RunMeta(
        model=model,
        mlx_lm_version=ver,
        python=sys.version.split()[0],
        platform=platform.platform(),
    )


# --- MLX 実走部 (実機でのみ import される) -----------------------------------------------------


def _load(model: str):
    from mlx_lm import load  # 遅延 import: selftest を mlx なしで通すため

    # trust_remote_code は使わない (供給網の固定)
    return load(model)


def _generate_step_compat(prompt_ids, model, sampler):
    """generate_step を呼び、(token, logprobs) を yield する generator を返す.

    cached decode: generate_step は内部で KV キャッシュを確保・更新する。
    """
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    prompt = mx.array(prompt_ids)
    # sampler を渡すと各ステップで logits -> token を選ぶ。logprobs は全語彙ぶん返る。
    # max_tokens=-1 で generate_step を無制限にし、停止はこのモジュール側のループ break に
    # 一本化する。既定 256 のままだと --max-tokens >= 256 で無音のうちに 255 点で頭打ちになる。
    return generate_step(prompt, model, sampler=sampler, max_tokens=-1)


def tokenize_demo(tokenizer) -> None:
    samples = [
        "strawberry",
        "苺がすきです",
        "    spaces",  # 連続空白
        "1234567890",  # 数字
        "tokenization",
    ]
    print("# tokenize-demo (トークンは文字でも単語でもない)")
    for s in samples:
        # 特殊トークン (BOS 等) を除いた「中身」のトークンだけ見せる
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(s)
        pieces = [tokenizer.decode([i]) for i in ids]
        print(f"  {s!r:18} -> {len(ids)} tokens: {pieces}")
    print()


def run_loop(model, tokenizer, prompt: str, max_tokens: int, top_k: int) -> None:
    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)  # greedy: 分布の最大を選ぶ
    prompt_ids = tokenizer.encode(prompt)

    print(f"# loop  prompt={prompt!r}  (各ステップの top-{top_k} と選ばれた1トークン)")
    steps = _generate_step_compat(prompt_ids, model, sampler)
    for i, (token, logprobs) in enumerate(steps):
        if i >= max_tokens:
            break
        tok_id = int(token)
        mx.eval(logprobs)  # この step の計算完了を待つ (device sync)
        topk = top_k_from_logprobs([float(x) for x in logprobs], k=top_k)
        chosen = tokenizer.decode([tok_id])
        shown = " | ".join(f"{tokenizer.decode([t])!r}:{p:.2f}" for t, p in topk)
        print(f"  step {i:3d}  pick={chosen!r:12} top{top_k}=[{shown}]")
    print()


def bench(model, tokenizer, prompt: str, max_tokens: int, reps: int) -> dict:
    """生成トークン数 vs 累積 wall-time を測る。warmup 1 回 + reps 回の median。

    最初のトークン (prefill 込み) は除外し、decode 区間の累積時間で線形を見る。
    """
    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    prompt_ids = tokenizer.encode(prompt)

    def one_run():
        cum = []  # (n_tokens, cumulative_seconds)
        steps = _generate_step_compat(prompt_ids, model, sampler)
        t0 = None  # decode 起点 (最初のトークン直後にセット)
        for i, (token, logprobs) in enumerate(steps):
            mx.eval(token)  # この step の計算完了を待つ (device sync)
            now = time.perf_counter()
            if i == 0:
                t0 = now  # 最初のトークン (prefill 込み) の直後を decode 起点に
                continue
            cum.append((i, now - t0))
            if i >= max_tokens:
                break
        return cum

    one_run()  # warmup (キャッシュ/コンパイルを温める)
    runs = [one_run() for _ in range(reps)]

    # token 数ごとに median を取る
    by_n: dict[int, list[float]] = {}
    for run in runs:
        for n, sec in run:
            by_n.setdefault(n, []).append(sec)
    points = sorted((n, median(v)) for n, v in by_n.items())
    ns = [n for n, _ in points]
    secs = [s for _, s in points]
    r2 = linear_fit_r2(ns, secs)
    tps = (ns[-1] / secs[-1]) if secs and secs[-1] > 0 else float("nan")
    return {
        "points": [{"n_tokens": n, "cum_seconds": s} for n, s in points],
        "linear_r2": r2,
        "decode_tps": tps,
        "reps": reps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM の自己回帰ループを1トークンずつ覗く")
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--demo-tokenize", action="store_true", help="tokenize-demo のみ")
    ap.add_argument("--bench", action="store_true", help="累積 wall-time の線形を実測し JSONL 出力")
    ap.add_argument("--selftest", action="store_true", help="MLX 非依存のロジック検証")
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    if args.selftest:
        from selftest import run_selftest  # 同ディレクトリ

        return run_selftest()

    meta = collect_run_meta(args.model)
    print(f"# run-meta: {json.dumps(asdict(meta), ensure_ascii=False)}\n")

    model, tokenizer = _load(args.model)

    if args.demo_tokenize:
        tokenize_demo(tokenizer)
        return 0

    tokenize_demo(tokenizer)
    run_loop(model, tokenizer, args.prompt, args.max_tokens, args.top_k)

    if args.bench:
        result = bench(model, tokenizer, args.prompt, args.max_tokens, args.reps)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"meta": asdict(meta), **result}, ensure_ascii=False) + "\n")
        print(f"# bench: decode_tps={result['decode_tps']:.1f}  linear_r2={result['linear_r2']:.4f}")
        print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

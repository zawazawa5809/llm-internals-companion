#!/usr/bin/env python3
"""GSM8K で self-consistency (maj@N) を実測する (llm-internals Part 4: llm-sampling-self-consistency).

やること:
  1. sample   : 各問題について greedy (T=0, 1本) + M本の temperature>0 サンプルを生成し、
                正答判定つきで JSONL に保存する (生成コストが高いため一度だけ実行してキャッシュする)。
  2. analyze  : 保存済みサンプルから N in {1,3,5,10,20} それぞれの maj@N 正答率を
                bootstrap CI 付きで算出する (再生成なし、保存済みプールの再利用のみ)。
  3. selftest : mlx 非依存の純ロジック検証 (answer 抽出・多数決・bootstrap CI)。実機なしで通る。

設計メモ (記事 methodology に直結):
  - **greedy (N=1) はサンプルプールから作らない**: temperature=0 は決定論的で常に同じ答えを返すため、
    多数決を取る母集団が生まれない。self-consistency の母集団は temperature>0 のサンプルのみで、
    greedy は「サンプリングしない場合の基準点」として独立に1回だけ生成する。
  - **M=20 本を一度だけサンプリングしプールする**: N<20 の maj@N は、このプールから非復元抽出した
    部分集合で見積もる。N 毎に生成し直すと N=1+3+5+10+20=39 倍のコストになるところを、
    実際には greedy 1 回 + サンプル 20 回 = 21 回の生成で済ませる標準的な評価トリック。
  - **bootstrap は2段階**:
      (a) 問題ごと: プール(サイズM)からサイズNを非復元抽出する試行を複数回繰り返し多数決精度を平均する
          (どの N 本が偶然選ばれたかのブレを均す)。
      (b) データセット全体: 問題そのものを複数回リサンプル(復元抽出)し、(a)の結果の平均のばらつきから
          データセット全体の正答率の信頼区間を percentile 法で出す。
  - **コストは実測**: 生成トークン数・wall-time を1完了ごとに記録し、N倍のクエリがどれだけの
    コストに対応するかを実データで示す (free != fast)。
  - **数値は環境依存**: モデル・machine・mlx-lm バージョンに依存する。一般化しない。run-meta に記録する。
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from collections import Counter


# --- mlx 非依存の純ロジック (selftest はここだけを検証する) ---------------------------------


def extract_gold(answer: str) -> str | None:
    """GSM8K の正解フォーマット `#### <数値>` から数値文字列を取り出す。"""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer)
    return m.group(1).replace(",", "") if m else None


def extract_pred(completion: str) -> str | None:
    """モデル出力から最終回答を取り出す。`#### <数値>` を優先し、無ければ最後の数値にフォールバック。"""
    m = re.search(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", completion)
    if m:
        return m.group(1).replace(",", "")
    # completion 全体からカンマを剥がすと "12, 34" のような無関係な2数字が "1234" に融合しうるため、
    # マッチ後(桁区切りカンマを含む1トークン)にだけ replace する。
    nums = re.findall(r"-?\d[\d,]*\.?\d*", completion)
    return nums[-1].replace(",", "") if nums else None


def normalize(x) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def is_correct(gold: float | None, pred: float | None, tol: float = 1e-4) -> bool:
    return gold is not None and pred is not None and abs(gold - pred) < tol


def _finite(x, nd: int):
    """非有限値 (nan/inf) は None に正規化する。JSONL に NaN を書くと下流の JSON.parse が壊れるため
    (kv_cache_demo.py / mlx-vs-ollama companion の bench.py と同じ不変条件)。"""
    import math

    return round(x, nd) if isinstance(x, (int, float)) and math.isfinite(x) else None


def majority_vote(preds: list):
    """最頻値を返す。同数タイは preds 内で先に出現した方 (決定論的タイブレーク)。空リストは None。"""
    if not preds:
        return None
    counts = Counter(preds)
    best_count = max(counts.values())
    for p in preds:
        if counts[p] == best_count:
            return p
    return preds[0]  # unreachable in practice


def per_question_accuracy_at_n(pool_preds: list, gold: float | None, n: int, resamples: int, rng) -> float:
    """プール(サイズM)からサイズnを非復元抽出しR回多数決精度を平均する (問題1問分)。"""
    m = len(pool_preds)
    if n >= m:
        chosen = majority_vote(pool_preds)
        return 1.0 if is_correct(gold, chosen) else 0.0
    hits = 0
    for _ in range(resamples):
        idx = rng.choice(m, size=n, replace=False)
        chosen = majority_vote([pool_preds[i] for i in idx])
        hits += is_correct(gold, chosen)
    return hits / resamples


def dataset_bootstrap_ci(per_q_acc: list, resamples: int, rng) -> tuple:
    """per_q_acc (問題ごとの accuracy_at_n, 0-1) を問題単位で復元抽出し percentile CI を出す。"""
    import numpy as np

    arr = np.array(per_q_acc)
    n = len(arr)
    means = []
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        means.append(arr[idx].mean())
    means = np.array(means)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --- mlx 依存部 (import は遅延させ、selftest を mlx なしで通す) -----------------------------


def run_meta(model_repo: str, m: int, temperature: float, top_p: float, max_tokens: int, seed: int) -> dict:
    import mlx.core as mx
    import mlx_lm

    return {
        "kind": "run-meta",
        "model": model_repo,
        "mlx_lm": getattr(mlx_lm, "__version__", "unknown"),
        "mlx": mx.__version__ if hasattr(mx, "__version__") else "unknown",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "m_samples": m,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
    }


def _build_prompt(question: str) -> str:
    return (
        'Solve this grade school math problem step by step. '
        'End your answer with "#### <number>".\n\n'
        f"Question: {question}\n\nAnswer:"
    )


def sample_all(
    model_repo: str,
    data_path: str,
    m: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    out_path: str,
    limit: int | None = None,
) -> list[dict]:
    """各問題について greedy 1本 + サンプル M本を生成し JSONL に保存する。"""
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_repo)
    mx.random.seed(seed)

    data = [json.loads(line) for line in open(data_path)]
    if limit:
        data = data[:limit]

    greedy_sampler = make_sampler(temp=0.0)
    sample_sampler = make_sampler(temp=temperature, top_p=top_p)

    records: list[dict] = []
    t_start = time.time()
    for i, ex in enumerate(data):
        prompt = _build_prompt(ex["question"])
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )
        gold = normalize(extract_gold(ex["answer"]))

        t0 = time.time()
        greedy_out = generate(
            model, tokenizer, prompt=formatted, max_tokens=max_tokens, sampler=greedy_sampler, verbose=False
        )
        greedy_time = time.time() - t0
        greedy_pred = normalize(extract_pred(greedy_out))

        samples = []
        for _ in range(m):
            t0 = time.time()
            out = generate(
                model, tokenizer, prompt=formatted, max_tokens=max_tokens, sampler=sample_sampler, verbose=False
            )
            dt = time.time() - t0
            pred = normalize(extract_pred(out))
            samples.append({"pred": pred, "ok": is_correct(gold, pred), "time_s": round(dt, 3), "n_chars": len(out)})

        rec = {
            "i": i,
            "gold": gold,
            "greedy_pred": greedy_pred,
            "greedy_ok": is_correct(gold, greedy_pred),
            "greedy_time_s": round(greedy_time, 3),
            "samples": samples,
        }
        records.append(rec)
        acc = sum(r["greedy_ok"] for r in records) / len(records)
        print(
            f"[{i + 1}/{len(data)}] greedy_acc_so_far={acc * 100:.1f}%  elapsed={time.time() - t_start:.0f}s",
            flush=True,
        )

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(run_meta(model_repo, m, temperature, top_p, max_tokens, seed)) + "\n")
    print(f"\nwrote {out_path}  (total elapsed {time.time() - t_start:.0f}s)")
    return records


def analyze(
    in_path: str,
    ns: list[int],
    resamples_per_question: int,
    dataset_resamples: int,
    seed: int,
    out_path: str | None,
) -> list[dict]:
    """保存済み results_samples.jsonl から N ごとの maj@N 正答率を bootstrap CI 付きで算出する。"""
    import numpy as np

    rng = np.random.default_rng(seed)
    lines = [json.loads(line) for line in open(in_path)]
    records = [r for r in lines if r.get("kind") != "run-meta"]
    meta = next((r for r in lines if r.get("kind") == "run-meta"), {})

    n_questions = len(records)
    total_sample_tokens_time = sum(s["time_s"] for r in records for s in r["samples"])
    total_greedy_time = sum(r["greedy_time_s"] for r in records)

    results = []
    print(f"# maj@N analysis: n_questions={n_questions}  model={meta.get('model', '?')}")
    print(f"{'N':>4} {'accuracy':>10} {'ci_lo':>8} {'ci_hi':>8} {'n_queries':>10} {'est_time_s':>11}")
    for n in ns:
        if n == 1:
            # N=1 は greedy 基準点 (self-consistency 適用前)。
            per_q = [1.0 if r["greedy_ok"] else 0.0 for r in records]
            est_time = total_greedy_time
            n_queries = n_questions
        else:
            per_q = [
                per_question_accuracy_at_n(
                    [s["pred"] for s in r["samples"]], r["gold"], n, resamples_per_question, rng
                )
                for r in records
            ]
            avg_sample_time = total_sample_tokens_time / sum(len(r["samples"]) for r in records)
            est_time = n_questions * n * avg_sample_time
            n_queries = n_questions * n

        mean_acc, lo, hi = dataset_bootstrap_ci(per_q, dataset_resamples, rng)
        rec = {
            "kind": "maj-at-n",
            "n": n,
            "accuracy": _finite(mean_acc, 4),
            "ci_lo": _finite(lo, 4),
            "ci_hi": _finite(hi, 4),
            "n_queries_total": n_queries,
            "est_wall_time_s": _finite(est_time, 1),
        }
        results.append(rec)
        print(f"{n:>4} {mean_acc * 100:>9.1f}% {lo * 100:>7.1f}% {hi * 100:>7.1f}% {n_queries:>10} {est_time:>11.1f}")

    if out_path:
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({**meta, "kind": "run-meta", "analysis_of": in_path}) + "\n")
        print(f"\nwrote {out_path}")
    return results


def selftest() -> int:
    """mlx 非依存のロジック検証 (実機なしで通る)。"""
    import numpy as np

    ok = True

    if extract_gold("blah blah #### 72") != "72":
        print("FAIL: extract_gold basic")
        ok = False
    if extract_gold("x = <<9*2=18>>18\n#### 1,234") != "1234":
        print("FAIL: extract_gold comma")
        ok = False
    if extract_pred("So the answer is #### $162") != "162":
        print("FAIL: extract_pred hash format")
        ok = False
    if extract_pred("I think the total is 42 apples.") != "42":
        print("FAIL: extract_pred fallback last number")
        ok = False
    if extract_pred("no numbers here") is not None:
        print("FAIL: extract_pred empty")
        ok = False
    if extract_pred("she has 12, then buys 34 more") != "34":
        print("FAIL: extract_pred must not fuse unrelated comma-separated numbers into one")
        ok = False
    if extract_pred("the total comes to 1,200 dollars") != "1200":
        print("FAIL: extract_pred thousands-separator number")
        ok = False

    if normalize("72") != 72.0 or normalize(None) is not None or normalize("abc") is not None:
        print("FAIL: normalize")
        ok = False

    if not is_correct(72.0, 72.0) or is_correct(72.0, 71.0) or is_correct(None, 72.0):
        print("FAIL: is_correct")
        ok = False

    if majority_vote([1.0, 1.0, 2.0]) != 1.0:
        print("FAIL: majority_vote simple majority")
        ok = False
    if majority_vote([1.0, 2.0]) != 1.0:  # tie -> 出現順で先に出た方
        print("FAIL: majority_vote tie-break")
        ok = False
    if majority_vote([]) is not None:
        print("FAIL: majority_vote empty")
        ok = False

    # per_question_accuracy_at_n: プール全体が正解で埋まっていれば N によらず1.0
    rng = np.random.default_rng(0)
    all_correct_pool = [5.0] * 20
    for n in (1, 3, 5, 10, 20):
        acc = per_question_accuracy_at_n(all_correct_pool, 5.0, n, resamples=10, rng=rng)
        if acc != 1.0:
            print(f"FAIL: per_question_accuracy_at_n all-correct N={n} got {acc}")
            ok = False

    # プールが少数派正解(20本中6本のみ正解)なら、Nが大きいほど多数決は不正解(誤答多数派)に收束するはず
    minority_correct_pool = [5.0] * 6 + [9.0] * 14  # 正解=5.0 は少数派
    acc_n1 = per_question_accuracy_at_n(minority_correct_pool, 5.0, 1, resamples=200, rng=rng)
    acc_n20 = per_question_accuracy_at_n(minority_correct_pool, 5.0, 20, resamples=1, rng=rng)
    if not (0.2 < acc_n1 < 0.4):  # N=1でランダムに1本引く場合 6/20=0.3 に近いはず
        print(f"FAIL: per_question_accuracy_at_n minority N=1 got {acc_n1} (expected ~0.3)")
        ok = False
    if acc_n20 != 0.0:  # N=20はプール全体=多数決は9.0(誤答)で確定
        print(f"FAIL: per_question_accuracy_at_n minority N=20 got {acc_n20} (expected 0.0)")
        ok = False

    # dataset_bootstrap_ci: 全問正解なら CI も [1.0, 1.0]
    mean, lo, hi = dataset_bootstrap_ci([1.0] * 50, resamples=200, rng=rng)
    if not (mean == 1.0 and lo == 1.0 and hi == 1.0):
        print(f"FAIL: dataset_bootstrap_ci all-correct got mean={mean} lo={lo} hi={hi}")
        ok = False
    # 半々なら CI は 0.5 付近で lo < mean < hi の順序が保たれる
    mixed = [1.0, 0.0] * 25
    mean, lo, hi = dataset_bootstrap_ci(mixed, resamples=500, rng=rng)
    if not (lo <= mean <= hi and 0.3 < mean < 0.7):
        print(f"FAIL: dataset_bootstrap_ci mixed got mean={mean} lo={lo} hi={hi}")
        ok = False

    print("selftest:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def _parse_ns(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",")]
    try:
        return [int(p) for p in parts if p]
    except ValueError as e:
        raise SystemExit(f"--ns の値が不正です: {raw!r} ({e})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="greedy + M本サンプリングを実行しJSONLに保存")
    ap.add_argument("--analyze", action="store_true", help="保存済みサンプルからmaj@N正答率(CI付き)を算出")
    ap.add_argument("--selftest", action="store_true", help="mlx不要のロジック検証")
    ap.add_argument("--model", type=str, default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--data", type=str, default="gsm8k_subset.jsonl")
    ap.add_argument("--m", type=int, default=20, help="self-consistency用サンプルプールのサイズ")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=350)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="デバッグ用: 先頭N問だけ処理")
    ap.add_argument("--ns", type=str, default="1,3,5,10,20")
    ap.add_argument("--resamples-per-question", type=int, default=200)
    ap.add_argument("--dataset-resamples", type=int, default=2000)
    ap.add_argument("--in", dest="in_path", type=str, default="results_samples.jsonl")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.sample:
        sample_all(
            args.model,
            args.data,
            args.m,
            args.temperature,
            args.top_p,
            args.max_tokens,
            args.seed,
            args.out or "results_samples.jsonl",
            limit=args.limit,
        )
    if args.analyze:
        analyze(
            args.in_path,
            _parse_ns(args.ns),
            args.resamples_per_question,
            args.dataset_resamples,
            args.seed,
            args.out or "results_analysis.jsonl",
        )
    if not (args.sample or args.analyze or args.selftest):
        ap.print_help()


if __name__ == "__main__":
    main()

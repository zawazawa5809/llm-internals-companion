#!/usr/bin/env python3
"""selftest.py — token_loop の純ロジックを MLX なしで検証する (実機なしで通る).

CI / サンドボックスで `python selftest.py` が緑になることを保証する。MLX や実モデルは不要。
"""

from __future__ import annotations

import math

from token_loop import linear_fit_r2, median, top_k_from_logprobs


def run_selftest() -> int:
    failures = []

    # top_k_from_logprobs: log p から top-k の (id, prob) を prob 降順で返す
    logprobs = [math.log(p) for p in [0.1, 0.5, 0.05, 0.3, 0.05]]
    topk = top_k_from_logprobs(logprobs, k=2)
    if [i for i, _ in topk] != [1, 3]:
        failures.append(f"top_k index 不一致: {topk}")
    if not (abs(topk[0][1] - 0.5) < 1e-6 and abs(topk[1][1] - 0.3) < 1e-6):
        failures.append(f"top_k prob 不一致: {topk}")

    # 確率は exp(logprob) で復元される
    single = top_k_from_logprobs([math.log(0.25)], k=1)
    if abs(single[0][1] - 0.25) > 1e-9:
        failures.append(f"exp 復元 不一致: {single}")

    # linear_fit_r2: 完全な直線は R^2 = 1
    xs = [1, 2, 3, 4, 5]
    ys = [0.1 * x + 0.02 for x in xs]
    r2 = linear_fit_r2(xs, ys)
    if abs(r2 - 1.0) > 1e-9:
        failures.append(f"linear_fit_r2 (直線) 不一致: {r2}")

    # 非線形は R^2 < 1
    if linear_fit_r2(xs, [x * x for x in xs]) >= 0.999:
        failures.append("linear_fit_r2 (二次) が 1 に近すぎる")

    # median: 偶数/奇数
    if median([3, 1, 2]) != 2 or median([1, 2, 3, 4]) != 2.5:
        failures.append("median 不一致")

    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SELFTEST PASSED (mlx 非依存ロジック: top_k / linear_fit_r2 / median)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_selftest())

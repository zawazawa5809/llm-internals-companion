#!/usr/bin/env python3
"""self-attention を numpy だけで最小実装し、系列長 T に対する計算量を実測する。

連載「ローカルLLMで覗く 言語モデルの中身」Part 2（llm-attention）の companion。

- scaled dot-product attention: softmax(Q Kᵀ / √d) · V （causal mask 付き）
- 系列長 T を変えて wall-time を測る。射影パート（Q/K/V = 線形 O(T·d²)）と
  attention コアパート（QKᵀ と attn·V = 二乗 O(T²·d)）を分けて計測する。
- スコア行列の理論メモリは T×T×4byte（float32）。これは計算で明示できる（実測 RSS ではない）。

数値は機種・BLAS・dtype 依存で一般化しない（free ≠ fast）。run-meta に環境を記録する。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from statistics import median

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数値安定な softmax（行ごとの max を引いてから exp する）。"""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_self_attention(
    x: np.ndarray, w_q: np.ndarray, w_k: np.ndarray, w_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """1 ヘッドの causal self-attention。out (T,d) と attention 重み (T,T) を返す。"""
    seq_len, d = x.shape
    q = x @ w_q  # (T, d)  各トークンの「問い合わせ」
    k = x @ w_k  # (T, d)  各トークンの「見出し」
    v = x @ w_v  # (T, d)  各トークンの「中身」
    scores = (q @ k.T) / np.sqrt(d)  # (T, T) ← ここが二乗の発生源
    future = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)  # 上三角=未来
    scores = np.where(future, -np.inf, scores)  # 未来を見えなくする（causal mask）
    attn = softmax(scores, axis=-1)  # (T, T) 行ごとに和=1
    out = attn @ v  # (T, d)
    return out, attn


def _split_timed(
    x: np.ndarray, w_q: np.ndarray, w_k: np.ndarray, w_v: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """射影パートと attention コアパートを分けて経過時間を測る。"""
    seq_len, d = x.shape
    t0 = time.perf_counter()
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v
    t1 = time.perf_counter()  # ← ここまで: 射影 O(T·d²)（T に線形）
    scores = (q @ k.T) / np.sqrt(d)
    future = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores = np.where(future, -np.inf, scores)
    attn = softmax(scores, axis=-1)
    out = attn @ v
    t2 = time.perf_counter()  # ← ここまで: attention コア O(T²·d)（T に二乗）
    return (t1 - t0), (t2 - t1), out


def run_meta() -> dict:
    return {
        "kind": "run-meta",
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def demo(seq_len: int = 8, d: int = 16, seed: int = 0, scale: float = 0.3) -> None:
    """小さい T で causal attention 行列を表示し、過去しか見ないことを可視化する。

    scale は Q/K 射影の初期化スケール。小さくすると softmax が緩み、重みが
    過去のトークンに分散する様子が見える（大きいと 1 トークンに尖る）。
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((seq_len, d)).astype(np.float32)
    w_q = (rng.standard_normal((d, d)) * scale).astype(np.float32)
    w_k = (rng.standard_normal((d, d)) * scale).astype(np.float32)
    w_v = rng.standard_normal((d, d)).astype(np.float32)
    _, attn = causal_self_attention(x, w_q, w_k, w_v)
    np.set_printoptions(precision=2, suppress=True)
    print(f"# causal attention 重み (T={seq_len}, d={d})  ※各行の和=1・上三角(未来)=0")
    print(attn)
    print()
    print("行 i = トークン i が、列 j = トークン j をどれだけ見るか。")
    print("下三角（と対角）だけに値が入る＝過去と自分しか見ていない。")
    print(f"この行列は T×T = {seq_len}×{seq_len} = {seq_len * seq_len} マス。T が増えると二乗で膨らむ。")


def bench(
    d: int = 128,
    ts: list[int] | None = None,
    reps: int = 5,
    warmup: int = 2,
    seed: int = 0,
    out_path: str | None = None,
) -> list[dict]:
    """T スイープで射影/attention コア/合計の wall-time とスコア行列メモリを測る。"""
    if ts is None:
        ts = [64, 128, 256, 512, 1024, 2048]
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    print(f"# T-sweep bench (d={d}, reps={reps}, warmup={warmup})")
    print(f"{'T':>6} {'proj_ms':>10} {'core_ms':>10} {'total_ms':>10} {'scores_MiB':>11}")
    for seq_len in ts:
        x = rng.standard_normal((seq_len, d)).astype(np.float32)
        w_q = rng.standard_normal((d, d)).astype(np.float32)
        w_k = rng.standard_normal((d, d)).astype(np.float32)
        w_v = rng.standard_normal((d, d)).astype(np.float32)
        for _ in range(warmup):
            _split_timed(x, w_q, w_k, w_v)
        projs, cores, totals = [], [], []
        for _ in range(reps):
            p, c, _ = _split_timed(x, w_q, w_k, w_v)
            projs.append(p)
            cores.append(c)
            totals.append(p + c)
        rec = {
            "kind": "bench",
            "T": seq_len,
            "d": d,
            "reps": reps,
            "proj_ms": round(median(projs) * 1000, 4),
            "attn_core_ms": round(median(cores) * 1000, 4),
            "total_ms": round(median(totals) * 1000, 4),
            "scores_bytes": seq_len * seq_len * 4,
            "scores_mib": round(seq_len * seq_len * 4 / 1024 / 1024, 3),
        }
        records.append(rec)
        print(
            f"{seq_len:>6} {rec['proj_ms']:>10.3f} {rec['attn_core_ms']:>10.3f} "
            f"{rec['total_ms']:>10.3f} {rec['scores_mib']:>11.3f}"
        )

    print()
    print("# T を 2 倍にしたときの時間倍率（attention コアが O(T²) なら約 4 倍、射影は約 2 倍）")
    print(f"{'T':>6} {'core_x':>8} {'proj_x':>8}")
    for i in range(1, len(records)):
        prev, cur = records[i - 1], records[i]
        core_x = cur["attn_core_ms"] / prev["attn_core_ms"] if prev["attn_core_ms"] else float("nan")
        proj_x = cur["proj_ms"] / prev["proj_ms"] if prev["proj_ms"] else float("nan")
        print(f"{cur['T']:>6} {core_x:>8.2f} {proj_x:>8.2f}")

    if out_path:
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps(run_meta()) + "\n")
        print(f"\nwrote {out_path}")
    return records


def selftest() -> int:
    """MLX 不要のロジック検証（実機なしで通る）。"""
    rng = np.random.default_rng(0)
    seq_len, d = 32, 16
    x = rng.standard_normal((seq_len, d)).astype(np.float32)
    w_q = rng.standard_normal((d, d)).astype(np.float32)
    w_k = rng.standard_normal((d, d)).astype(np.float32)
    w_v = rng.standard_normal((d, d)).astype(np.float32)
    out, attn = causal_self_attention(x, w_q, w_k, w_v)

    ok = True
    if out.shape != (seq_len, d):
        print("FAIL: out shape", out.shape)
        ok = False
    row_sums = attn.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        print("FAIL: softmax row-sum", float(row_sums.min()), float(row_sums.max()))
        ok = False
    future = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    if not np.allclose(attn[future], 0.0, atol=1e-7):
        print("FAIL: causal leak (未来に重みが漏れている)", float(attn[future].max()))
        ok = False
    s = softmax(np.array([[1.0, 2.0, 3.0]]))
    if not np.allclose(s.sum(), 1.0):
        print("FAIL: softmax sum")
        ok = False

    print("selftest:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="小さい T で causal attention 行列を表示")
    ap.add_argument("--bench", action="store_true", help="T スイープで wall-time とメモリを実測")
    ap.add_argument("--selftest", action="store_true", help="MLX 不要のロジック検証")
    ap.add_argument("--d", type=int, default=128, help="モデル次元 d (default 128)")
    ap.add_argument("--reps", type=int, default=5, help="計測の反復回数 (median を取る)")
    ap.add_argument("--ts", type=str, default="64,128,256,512,1024,2048", help="T のスイープ列 (カンマ区切り)")
    ap.add_argument("--scale", type=float, default=0.3, help="demo: Q/K 射影スケール (小さいと重みが分散)")
    ap.add_argument("--out", type=str, default=None, help="bench 結果の保存先 JSONL")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.demo:
        demo(scale=args.scale)
    if args.bench:
        ts = [int(t) for t in args.ts.split(",")]
        bench(d=args.d, ts=ts, reps=args.reps, out_path=args.out)
    if not (args.demo or args.bench or args.selftest):
        ap.print_help()


if __name__ == "__main__":
    main()

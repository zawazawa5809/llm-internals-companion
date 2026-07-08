#!/usr/bin/env python3
"""KV キャッシュのサイズと decode 速度を実測する (llm-internals Part 3: llm-kv-cache).

やること:
  1. probe  : mlx_lm の内部キャッシュ (`KVCache`) を1本 introspect し、
              「確保容量ベースの nbytes（罠）」と「論理長ベースの state nbytes」の違いを、
              256 トークン境界をまたぐ細かい context スイープで見せる。
  2. kv     : context 長 1K〜32K で KV バイト数を実測し、理論式・state(論理長)・raw(確保容量) を突き合わせる。
  3. decode : 同じ context 長スイープで decode tok/s を実測する（memory-bound の実測）。
  4. selftest: mlx 非依存の純ロジック検証（理論式・padding 計算）。実機なしで通る。

設計メモ (記事 methodology に直結):
  - **KV introspection の罠**: mlx_lm の `KVCache` は `step=256` 単位でキャッシュ配列を事前確保する
    (`mlx_lm/models/cache.py` を直接読解して確認)。`cache[i].nbytes` は確保済みの生配列サイズを返すため、
    論理長 (`cache[i].offset`) とは 256 トークン単位でしか一致しない。理論式と正しく突き合わせるには
    `cache[i].state`（`offset` でスライス済み）の nbytes を使う。
  - **prefill と decode を分離して測る**: `generate_step(..., prompt_cache=cache)` の最初の yield は
    prefill 込みなので decode tok/s の統計から除外する（Part 1 と同じ規律）。
  - **KV キャッシュの dtype は重み量子化と独立**: 4bit 量子化モデルでも既定の `KVCache` は fp16 で
    Key/Value を保持する（`QuantizedKVCache` を明示的に使わない限り）。理論式の dtype_bytes は
    実際に `cache[i].keys.dtype` を読んで決める（決め打ちしない）。
  - **数値は環境依存**: tok/s もバイト数も「そのモデル・その機種・その mlx-lm バージョン」の値。
    一般化しない (free != fast)。run-meta に環境を記録する。
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from statistics import median


# --- mlx 非依存の純ロジック (selftest はここだけを検証する) ---------------------------------


def theoretical_kv_bytes(
    n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int, dtype_bytes: int
) -> int:
    """KV キャッシュの理論バイト数: 2 (K+V) x 層数 x KVヘッド数 x head_dim x 系列長 x dtypeバイト数。"""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * dtype_bytes


def padded_length(seq_len: int, step: int = 256) -> int:
    """mlx_lm の KVCache が実際に確保する長さ (step の倍数に切り上げ)。"""
    if seq_len <= 0:
        return 0
    n_steps = (step + seq_len - 1) // step
    return n_steps * step


def _finite(x, nd: int):
    """非有限値 (nan/inf) は None に正規化する。JSONL に NaN を書くと下流の JSON.parse が壊れるため
    (mlx-vs-ollama companion の bench.py と同じ不変条件)。"""
    return round(x, nd) if isinstance(x, (int, float)) and math.isfinite(x) else None


# --- mlx 依存部 (import は遅延させ、selftest を mlx なしで通す) -----------------------------


def _load(model_repo: str):
    from mlx_lm import load

    return load(model_repo)


def _model_config(model) -> dict:
    """1 層目の attention から n_heads/n_kv_heads/head_dim を、model.layers から n_layers を読む。"""
    layer0 = model.layers[0]
    attn = layer0.self_attn
    return {
        "n_layers": len(model.layers),
        "n_heads": attn.n_heads,
        "n_kv_heads": attn.n_kv_heads,
        "head_dim": attn.head_dim,
    }


def run_meta(model_repo: str, cfg: dict, dtype_bytes: int) -> dict:
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
        "model_config": cfg,
        "kv_dtype_bytes": dtype_bytes,
    }


def _prefill(model, cache, context_len: int, seed: int = 0):
    """seed 由来の合成トークン列 (語彙からランダムサンプル) を1回の forward で cache に流し込む。

    生成品質は測らない (意味のある文である必要はない)。context 長そのものと、KV キャッシュの
    大きさ・decode 速度への効果だけを見る。
    """
    import mlx.core as mx
    import numpy as np

    vocab_size = model.args.vocab_size
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab_size, size=(1, context_len))
    x = mx.array(ids)
    logits = model(x, cache=cache)
    mx.eval(logits)
    return logits


def probe(model_repo: str, around: int = 256, span: int = 12) -> None:
    """256 境界をまたぐ細かい context スイープで「罠」(raw nbytes の階段関数) を見せる。"""
    from mlx_lm.models.cache import make_prompt_cache

    model, _ = _load(model_repo)
    cfg = _model_config(model)
    dtype_bytes = None

    print(f"# probe: model={model_repo}  n_layers={cfg['n_layers']} n_heads={cfg['n_heads']} "
          f"n_kv_heads={cfg['n_kv_heads']} head_dim={cfg['head_dim']}")
    print(f"{'ctx':>6} {'offset':>8} {'padded_len(理論)':>16} {'raw_MiB(確保容量)':>18} "
          f"{'state_MiB(論理長)':>18} {'theory_MiB':>12}")

    lo = max(1, around - span)
    hi = around + span
    for context_len in range(lo, hi + 1):
        cache = make_prompt_cache(model)
        _prefill(model, cache, context_len)
        c0 = cache[0]
        if dtype_bytes is None:
            dtype_bytes = c0.keys.dtype.size
        raw_bytes = sum(c.nbytes for c in cache)
        state_bytes = sum(s.nbytes for c in cache for s in c.state)
        theory_bytes = theoretical_kv_bytes(
            cfg["n_layers"], cfg["n_kv_heads"], cfg["head_dim"], context_len, dtype_bytes
        )
        pred_padded = padded_length(context_len)
        print(
            f"{context_len:>6} {c0.offset:>8} {pred_padded:>16} "
            f"{raw_bytes / 1024 / 1024:>18.4f} {state_bytes / 1024 / 1024:>18.4f} "
            f"{theory_bytes / 1024 / 1024:>12.4f}"
        )

    print()
    print("# 観察: raw (確保容量ベース) は 256 の倍数でしか値が変わらない階段関数。")
    print("#       state (論理長ベース) と theory (理論式) は 1 トークンごとに滑らかに増え、常に一致する。")


def bench_kv(
    model_repo: str,
    contexts: list[int],
    seed: int = 0,
    out_path: str | None = None,
) -> list[dict]:
    """context 長スイープで KV バイト数を実測し、理論式・state・raw を突き合わせる。"""
    from mlx_lm.models.cache import make_prompt_cache

    model, _ = _load(model_repo)
    cfg = _model_config(model)
    dtype_bytes = None
    records: list[dict] = []

    print(f"# kv-bytes bench: model={model_repo}")
    print(f"{'context':>8} {'raw_MiB':>10} {'state_MiB':>10} {'theory_MiB':>11} {'match':>7}")
    for context_len in contexts:
        cache = make_prompt_cache(model)
        _prefill(model, cache, context_len, seed=seed)
        c0 = cache[0]
        if dtype_bytes is None:
            dtype_bytes = c0.keys.dtype.size
        raw_bytes = sum(c.nbytes for c in cache)
        state_bytes = sum(s.nbytes for c in cache for s in c.state)
        theory_bytes = theoretical_kv_bytes(
            cfg["n_layers"], cfg["n_kv_heads"], cfg["head_dim"], context_len, dtype_bytes
        )
        match = "OK" if state_bytes == theory_bytes else "MISMATCH"
        rec = {
            "kind": "kv-bench",
            "context": context_len,
            "raw_bytes": raw_bytes,
            "state_bytes": state_bytes,
            "theory_bytes": theory_bytes,
            "raw_mib": round(raw_bytes / 1024 / 1024, 4),
            "state_mib": round(state_bytes / 1024 / 1024, 4),
            "theory_mib": round(theory_bytes / 1024 / 1024, 4),
            "match": match,
        }
        records.append(rec)
        print(
            f"{context_len:>8} {rec['raw_mib']:>10.3f} {rec['state_mib']:>10.3f} "
            f"{rec['theory_mib']:>11.3f} {match:>7}"
        )

    if out_path:
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps(run_meta(model_repo, cfg, dtype_bytes)) + "\n")
        print(f"\nwrote {out_path}")
    return records


def bench_decode(
    model_repo: str,
    contexts: list[int],
    max_new_tokens: int = 20,
    warmup_tokens: int = 4,
    seed: int = 0,
    out_path: str | None = None,
) -> list[dict]:
    """context 長スイープで decode tok/s を実測する (prefill 分は統計から除外)。"""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _load(model_repo)
    cfg = _model_config(model)
    dtype_bytes = None
    sampler = make_sampler(temp=0.0)
    records: list[dict] = []

    print(f"# decode bench: model={model_repo}  (max_new_tokens={max_new_tokens}, "
          f"warmup_tokens={warmup_tokens} を統計から除外)")
    print(f"{'context':>8} {'decode_tok_s':>13} {'first_step_ms(prefill込み)':>26}")
    for context_len in contexts:
        import numpy as np

        vocab_size = model.args.vocab_size
        rng = np.random.default_rng(seed)
        prompt_ids = mx.array(rng.integers(0, vocab_size, size=(context_len,)))

        cache = make_prompt_cache(model)
        if dtype_bytes is None:
            # cache はまだ空。1トークン流して dtype だけ確認してから作り直す。
            probe_cache = make_prompt_cache(model)
            _prefill(model, probe_cache, 1)
            dtype_bytes = probe_cache[0].keys.dtype.size

        step_times = []
        t_prev = time.perf_counter()
        count = 0
        for token, _logprobs in generate_step(
            prompt_ids, model, max_tokens=max_new_tokens, sampler=sampler, prompt_cache=cache
        ):
            mx.eval(token)
            t_now = time.perf_counter()
            step_times.append(t_now - t_prev)
            t_prev = t_now
            count += 1
            if count >= max_new_tokens:
                break

        first_step_ms = step_times[0] * 1000 if step_times else float("nan")
        decode_steps = step_times[warmup_tokens:]
        med = median(decode_steps) if decode_steps else float("nan")
        tok_s = (1.0 / med) if med and med > 0 else float("nan")
        rec = {
            "kind": "decode-bench",
            "context": context_len,
            "decode_tok_s": _finite(tok_s, 3),
            "first_step_ms": _finite(first_step_ms, 3),
            "n_decode_samples": len(decode_steps),
        }
        records.append(rec)
        print(f"{context_len:>8} {tok_s:>13.3f} {first_step_ms:>26.2f}")

    if out_path:
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps(run_meta(model_repo, cfg, dtype_bytes)) + "\n")
        print(f"\nwrote {out_path}")
    return records


def selftest() -> int:
    """mlx 非依存のロジック検証 (実機なしで通る)。"""
    ok = True

    # 理論式: n_layers=16, n_kv_heads=8, head_dim=64, dtype_bytes=2 (fp16) の Llama 3.2 1B 相当
    b = theoretical_kv_bytes(16, 8, 64, 1000, 2)
    expected = 2 * 16 * 8 * 64 * 1000 * 2
    if b != expected:
        print("FAIL: theoretical_kv_bytes", b, "!=", expected)
        ok = False

    # padding: 1000 は 256 の倍数に切り上げると 1024 (4*256)
    if padded_length(1000) != 1024:
        print("FAIL: padded_length(1000)", padded_length(1000))
        ok = False
    # ちょうど 256 の倍数はそのまま (切り上げで増えない)
    if padded_length(1024) != 1024:
        print("FAIL: padded_length(1024)", padded_length(1024))
        ok = False
    if padded_length(1) != 256:
        print("FAIL: padded_length(1)", padded_length(1))
        ok = False
    if padded_length(0) != 0:
        print("FAIL: padded_length(0)", padded_length(0))
        ok = False

    # 線形性: context を2倍にすれば理論バイト数もちょうど2倍
    b1 = theoretical_kv_bytes(16, 8, 64, 2048, 2)
    b2 = theoretical_kv_bytes(16, 8, 64, 4096, 2)
    if b2 != b1 * 2:
        print("FAIL: linearity", b1, b2)
        ok = False

    print("selftest:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def _parse_contexts(raw: str) -> list[int]:
    """カンマ区切りの context 長リストをパースする。空要素(例: 末尾カンマ)は無視する。"""
    parts = [p.strip() for p in raw.split(",")]
    try:
        return [int(p) for p in parts if p]
    except ValueError as e:
        raise SystemExit(f"--contexts の値が不正です: {raw!r} ({e})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true", help="256境界をまたぐ細かいスイープで罠を見せる")
    ap.add_argument("--kv", action="store_true", help="context 1K-32K で KV バイト数を実測")
    ap.add_argument("--decode", action="store_true", help="context 1K-32K で decode tok/s を実測")
    ap.add_argument("--selftest", action="store_true", help="mlx 不要のロジック検証")
    ap.add_argument(
        "--model", type=str, default="mlx-community/Llama-3.2-1B-Instruct-4bit", help="mlx-community モデル repo"
    )
    ap.add_argument("--contexts", type=str, default="1024,2048,4096,8192,16384,32768", help="context 長スイープ(カンマ区切り)")
    ap.add_argument("--max-new-tokens", type=int, default=20, help="decode: 生成するトークン数")
    ap.add_argument("--warmup-tokens", type=int, default=4, help="decode: 統計から除外する先頭ステップ数")
    ap.add_argument("--around", type=int, default=256, help="probe: 中心にする context 長 (256境界)")
    ap.add_argument("--span", type=int, default=12, help="probe: around の前後何トークン分見るか")
    ap.add_argument("--out", type=str, default=None, help="bench 結果の保存先 JSONL")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.probe:
        probe(args.model, around=args.around, span=args.span)
    if args.kv:
        contexts = _parse_contexts(args.contexts)
        bench_kv(args.model, contexts, out_path=args.out)
    if args.decode:
        contexts = _parse_contexts(args.contexts)
        bench_decode(args.model, contexts, max_new_tokens=args.max_new_tokens, warmup_tokens=args.warmup_tokens, out_path=args.out)
    if not (args.probe or args.kv or args.decode or args.selftest):
        ap.print_help()


if __name__ == "__main__":
    main()

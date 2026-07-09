#!/usr/bin/env python3
"""tiny_gpt.py — nanoGPT 級の decoder-only Transformer を char-level でゼロから学習する (Part 5: llm-train-tiny).

Part 1-4 は既存モデルの重みを使った推論（forward のみ）を扱った。本 Part は forward + backward +
optimizer step という学習ループそのものを最小実装し、TinyShakespeare で train/val loss・perplexity・
bits-per-char・所要時間・ピークメモリを実測する。

  - **モデル**: Karpathy の nanoGPT 標準 config（n_layer=6, n_head=6, n_embd=384, block_size=256,
    dropout=0.2, ≈10.6M params）。causal self-attention + MLP + LayerNorm の decoder-only block。
  - **データ**: TinyShakespeare（`input.txt`, karpathy/char-rnn 由来, ライセンス表記なし=出典明記のうえ
    教育目的として使用）。char-level tokenizer（1 文字 = 1 トークン）。
  - **MPS の注意点**: bf16 autocast は環境によって不安定という報告があるため fp32 を既定にする。
    `--dtype` は float32 既定、float16/bfloat16 を指定した場合のみ実際に `torch.autocast` で forward
    に反映する（値を受け取るだけで forward に反映されない、という不整合は起こさない）。
    `PYTORCH_ENABLE_MPS_FALLBACK=1` を既定で有効にし、未対応演算があれば CPU にフォールバックさせる
    （フォールバックが発生した場合は run-meta に `mps_fallback_triggered=true` として記録し、計測値が
    汚染されている可能性を明示する）。
  - **メモリ計測**: torch.mps には CUDA の `max_memory_allocated()` に相当する peak tracker が無いため、
    `torch.mps.current_allocated_memory()` を毎 iteration（backward+optimizer.step 直後、勾配とoptimizer
    状態が確保された直後）でサンプリングし、その最大値を peak の近似値として報告する（この方法であることを
    run-meta に明記する）。
  - **RNG の分離**: 学習バッチ用と評価バッチ用で別の Generator を使う。同じ Generator を共有すると
    `--eval-iters` を変えるだけで消費される乱数列がずれ、学習バッチの抽選順（＝学習軌跡そのもの）が
    変わってしまうため。
  - **数値は環境依存**: 所要時間・ピークメモリは machine・torch バージョンに依存する。run-meta に記録し、
    一般化しない（free ≠ fast の系譜）。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DATA = "input.txt"
DEFAULT_OUT = "results.jsonl"


# ---------------------------------------------------------------------------
# Tokenizer (char-level — 1 文字 = 1 トークン。Part 1 の「token != 文字」の裏返し)
# ---------------------------------------------------------------------------


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


# ---------------------------------------------------------------------------
# Model — decoder-only Transformer（Karpathy nanoGPT 相当の最小実装）
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2


class CausalSelfAttention(nn.Module):
    """Part 2 (attention_from_scratch.py) の学習可能版。KV キャッシュ（Part 3）は使わない —
    学習では毎回全系列を forward するため、推論時専用の最適化であるキャッシュは不要。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        assert t <= self.config.block_size, f"系列長 {t} が block_size {self.config.block_size} を超過"
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path: Path) -> tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    text = path.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(ids))
    return tok, ids[:n], ids[n:]


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: torch.device, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Measurement helpers（perplexity / bits-per-char — 01_research.md の定義に対応）
# ---------------------------------------------------------------------------


def perplexity(loss_nats: float) -> float:
    """PPL = exp(cross-entropy loss in nats)。loss が発散した場合 math.exp は OverflowError を
    投げうるため inf に丸める（呼び出し側は _finite() で JSON-safe な None に正規化する）。"""
    try:
        return math.exp(loss_nats)
    except OverflowError:
        return math.inf


def bits_per_char(loss_nats: float) -> float:
    """BPC = loss(nats) / ln(2)。PPL と同じ量を bits 単位で表したもの。"""
    return loss_nats / math.log(2)


def _finite(x, nd: int = 6):
    """非有限値 (nan/inf) は None に正規化する。JSONL に NaN/Infinity を書くと下流の JSON.parse が壊れるため
    (kv_cache_demo.py / self_consistency.py と同じ不変条件)。"""
    return round(x, nd) if isinstance(x, (int, float)) and math.isfinite(x) else None


def amp_context(device: torch.device, dtype_name: str):
    """--dtype float32 (既定) では no-op。float16/bfloat16 を指定した場合のみ実際に autocast する
    （以前は --dtype を受け取っても forward に反映されず、run-meta にだけ値が記録される不整合があった）。"""
    if dtype_name == "float32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=getattr(torch, dtype_name))


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
    eval_iters: int,
    dtype_name: str = "float32",
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device, generator)
        with amp_context(device, dtype_name):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def run_meta(args: argparse.Namespace, model: TinyGPT, device: torch.device, mps_fallback_triggered: bool) -> dict:
    return {
        "kind": "run-meta",
        "config": {
            "vocab_size": model.config.vocab_size,
            "block_size": model.config.block_size,
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_embd": model.config.n_embd,
            "dropout": model.config.dropout,
        },
        "num_params": model.num_params(),
        "train_args": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_iters": args.max_iters,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "seed": args.seed,
        },
        "device": str(device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        "mps_fallback_triggered": mps_fallback_triggered,
        "peak_mem_method": "current_allocated_memory() sampled every iteration right after backward()+optimizer.step(), max taken as peak approximation (torch.mps has no CUDA-equivalent peak tracker)",
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "mps" and args.dtype != "float32":
        print(
            "警告: MPS では bf16 autocast が不安定という報告があります(01_research.md参照)。"
            " fp32(--dtype float32)を推奨します。"
        )

    torch.manual_seed(args.seed)
    if device.type == "mps":
        torch.mps.manual_seed(args.seed)
    # 学習バッチ用と評価バッチ用で Generator を分離する（同一 Generator だと --eval-iters を
    # 変えただけで学習バッチの抽選順がずれ、学習軌跡そのものが変わってしまうため）。
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    eval_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    tok, train_data, val_data = load_data(Path(args.data))
    config = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"vocab_size={tok.vocab_size} params={model.num_params():,} device={device}")

    mps_fallback_triggered = False
    records: list[dict] = []
    peak_mem_bytes = 0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        t0 = time.time()
        for it in range(args.max_iters):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device, generator)
            with amp_context(device, args.dtype):
                _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if device.type == "mps":
                # backward+optimizer.step 直後（勾配・optimizer状態が確保された直後）にサンプリングする。
                # 評価チェックポイントでしか読まないと、その時点では勾配バッファが既に解放されており
                # 実際のピーク（backward中）を過小評価しうる。同期なしで読める軽量な呼び出しなので
                # 毎 iteration サンプリングしてもコストは無視できる。
                peak_mem_bytes = max(peak_mem_bytes, torch.mps.current_allocated_memory())

            if it % args.eval_interval == 0 or it == args.max_iters - 1:
                if device.type == "mps":
                    torch.mps.synchronize()
                train_loss = estimate_loss(
                    model,
                    train_data,
                    args.block_size,
                    args.batch_size,
                    device,
                    eval_generator,
                    args.eval_iters,
                    args.dtype,
                )
                val_loss = estimate_loss(
                    model,
                    val_data,
                    args.block_size,
                    args.batch_size,
                    device,
                    eval_generator,
                    args.eval_iters,
                    args.dtype,
                )
                elapsed = time.time() - t0
                current_mem = torch.mps.current_allocated_memory() if device.type == "mps" else None

                rec = {
                    "kind": "train-step",
                    "iter": it,
                    "train_loss": _finite(train_loss),
                    "val_loss": _finite(val_loss),
                    "train_ppl": _finite(perplexity(train_loss)),
                    "val_ppl": _finite(perplexity(val_loss)),
                    "train_bpc": _finite(bits_per_char(train_loss)),
                    "val_bpc": _finite(bits_per_char(val_loss)),
                    "elapsed_s": _finite(elapsed, 1),
                    "current_mem_bytes": current_mem,
                }
                records.append(rec)
                print(
                    f"iter {it:5d} | train {train_loss:.4f} val {val_loss:.4f} "
                    f"| ppl(val) {perplexity(val_loss):7.2f} bpc(val) {bits_per_char(val_loss):.3f} "
                    f"| {elapsed:7.1f}s"
                )

        for w in caught:
            msg = str(w.message)
            if "MPS backend" in msg or "fall back" in msg.lower():
                mps_fallback_triggered = True

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        meta = run_meta(args, model, device, mps_fallback_triggered)
        meta["peak_mem_bytes"] = peak_mem_bytes if device.type == "mps" else None
        f.write(json.dumps(meta) + "\n")

    print(f"done. wrote {len(records)} records + run-meta to {args.out}")
    if mps_fallback_triggered:
        print("警告: MPS fallback が発火しました。計測値の一部がCPU実行分を含む可能性があります。")


# ---------------------------------------------------------------------------
# Selftest — 実機(MPS実走)なしで CPU 上で通るロジック検証
# ---------------------------------------------------------------------------


def selftest() -> int:
    failures: list[str] = []
    torch.manual_seed(0)

    # tokenizer roundtrip
    tok = CharTokenizer("hello world")
    ids = tok.encode("hello")
    if tok.decode(ids) != "hello":
        failures.append(f"tokenizer roundtrip 不一致: {tok.decode(ids)!r}")

    # forward shape + finite loss
    config = GPTConfig(vocab_size=tok.vocab_size, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = TinyGPT(config)
    idx = torch.randint(0, tok.vocab_size, (2, 8))
    targets = torch.randint(0, tok.vocab_size, (2, 8))
    logits, loss = model(idx, targets)
    if tuple(logits.shape) != (2, 8, tok.vocab_size):
        failures.append(f"forward shape 不一致: {tuple(logits.shape)}")
    if loss is None or not math.isfinite(loss.item()):
        failures.append("loss が有限でない")

    # 過学習チェック: 同じ小バッチを繰り返し学習すれば loss は大幅に下がるはず
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first_loss = None
    last_loss = None
    for step in range(80):
        _, step_loss = model(idx, targets)
        optimizer.zero_grad(set_to_none=True)
        step_loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = step_loss.item()
        last_loss = step_loss.item()
    if first_loss is None or last_loss is None or not (last_loss < first_loss * 0.5):
        failures.append(f"過学習でlossが十分下がらない: first={first_loss} last={last_loss}")

    # perplexity / bits-per-char の定義確認
    if abs(perplexity(0.0) - 1.0) > 1e-9:
        failures.append(f"perplexity(0)が1でない: {perplexity(0.0)}")
    if abs(bits_per_char(math.log(2)) - 1.0) > 1e-9:
        failures.append(f"bits_per_char(ln2)が1でない: {bits_per_char(math.log(2))}")

    # causal mask バッファ自体が下三角になっているか(構造チェック)
    attn = model.blocks[0].attn
    if attn.causal_mask[0, 0, 0, 1].item() != 0:
        failures.append("causal_mask が下三角になっていない")

    # causal mask が forward で実際に効いているか(end-to-endチェック)。
    # 上のバッファチェックだけだと、forward内でスライス方向を間違えていても検出できない。
    # 位置0のlogitsは、位置0より後ろのトークンを変えても不変であるべき(未来は見えない)。
    model.eval()
    with torch.no_grad():
        base = torch.randint(0, tok.vocab_size, (1, 8))
        perturbed = base.clone()
        perturbed[0, 4:] = (perturbed[0, 4:] + 1) % tok.vocab_size
        logits_base, _ = model(base)
        logits_perturbed, _ = model(perturbed)
        if not torch.allclose(logits_base[0, 0], logits_perturbed[0, 0], atol=1e-5):
            failures.append("causality違反: 未来トークンの変更で過去位置のlogitsが変化した")
    model.train()

    # get_batch: y は x を1つ右にずらしたものであるべき(next-token予測の根幹、off-by-one混入の典型箇所)
    data = torch.arange(100, dtype=torch.long)
    gen = torch.Generator(device="cpu").manual_seed(0)
    bx, by = get_batch(data, block_size=8, batch_size=4, device=torch.device("cpu"), generator=gen)
    if not torch.equal(by, bx + 1):
        failures.append(f"get_batch: y が x+1(次の文字)になっていない: x[0]={bx[0].tolist()} y[0]={by[0].tolist()}")

    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        "SELFTEST PASSED (tokenizer roundtrip / forward shape / 過学習でloss減少 "
        "/ perplexity・bpc定義 / causal mask整合(バッファ+forward経路のend-to-end) / get_batchのnext-token整合)"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="学習ループを実行し results.jsonl に記録する")
    mode.add_argument(
        "--selftest", action="store_true", help="torch非依存ではないが実機(MPS実走)なしで通るロジック検証"
    )
    ap.add_argument("--data", type=str, default=DEFAULT_DATA)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=50)
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())
    train(args)


if __name__ == "__main__":
    main()

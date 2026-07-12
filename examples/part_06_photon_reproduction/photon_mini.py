#!/usr/bin/env python3
"""photon_mini.py — PHOTON (arXiv:2512.20687) のチャンク階層アーキテクチャを char-level でミニチュア再現する
(Part 6: llm-photon-reproduction, 連載 llm-internals 最終回).

Part 5 の tiny_gpt.py (vanilla decoder-only Transformer) を直接 import し、bottom-up encoder + top-down
decoder というチャンク階層構造を追加した photon_mini.py を同じデータ・同じ学習ループの上に構築する。

  - **論文からの意図的な単純化 (honest scope)**: 論文の主結果は 2 レベル階層 (L=2, チャンク長 C1=4, C2=4)。
    本実装は **1 レベル階層 (L=1)** に単純化する — トークン列 (level 0) を chunk_size 個ずつまとめて
    1 段だけ粗い latent (level 1) を作り、そこから top-down で token 表現を再構成する。論文のフル階層
    ではなく MEGABYTE (Yu et al., 2023) と同型の 2 層構造 (global/local) に近い、最小限だが本質を保った
    ミニチュアである。recursive consistency 補助損失 (L_rec) も実装しない — 論文の主結果自体が
    alpha=0.0 (階層構造そのものの効果を単離する設定) を採用しているため、この点は論文の主設定と一致する。
  - **RecGen は out of scope**: 論文の「10^3倍」headline は RecGen (decode 時に bottom-up 再エンコードを
    省略する追加最適化)込みの数字。本実装は HierGen (毎ステップ bottom-up 再エンコードする素朴な階層生成)
    のみを実装し、「475倍」(HierGen 相当, Table 1) の再現を狙う。01_research.md §4 参照。

  - **Bottom-up encoder**:
    (1) ContextChunker — token-level embedding を chunk_size 個ずつ concat + 線形射影で 1 chunk = 1 vector
        に圧縮する (論文 p.3: 「concatenating the representations within a chunk, followed by a linear
        projection」をそのまま採用)。
    (2) ContextEncoder — chunk 列を causal self-attention で文脈化する、Part 5 の Block を再利用した
        小さな自己回帰 Transformer。
  - **Top-down decoder**:
    (3) ContextConverter — 1 個の chunk-level latent を converter_len (R) 本の conditioning vector に
        展開する (論文は 1D conv、本実装は Linear + reshape で同等の機能を実装する簡略化)。
    (4) ContextDecoder — 各 chunk を [R 本の prefix; chunk_size 個のtoken位置] という長さ R+C の窓に
        causal mask をかける局所自己回帰 Transformer。chunk g の decode は chunk (g-1) の encoder 状態
        (= 1 つ前の coarse state) を条件とする (論文 p.3 の定義通り、chunk 自身の未来情報が prefix に
        漏れない設計)。**並列性の所在に注意**: 学習 (teacher forcing) では全 chunk の全 token 位置が
        既知なので、chunk をバッチ次元に畳み込んで chunk 間を並列処理できる (forward() 参照)。しかし
        *生成* 時は chunk 内の token は前の token に依存する自己回帰列であり、1 つずつ逐次生成する
        しかない (vanilla の token-by-token 生成と同じ)。論文も「local decoder generates ... latents
        autoregressively in each chunk」(p.3) と明記しており、並列化できるのは chunk 間 (独立した複数
        chunk/系列をバッチ処理すること) であって chunk 内ではない。generate_hiergen() の Python for
        ループは chunk 内の自己回帰性そのものであり、これを「並列化できるはずの処理のオーバーヘッド」
        と誤って書かないこと (code-review指摘への対応)。

  - **KV 相当のメモリ削減の測る場所**: 学習 (teacher forcing) では毎回全系列を並列 forward するため
    KV キャッシュという概念自体が存在しない (Part 5 の tiny_gpt.py と同じ)。PHOTON のメモリ削減は
    *生成 (decode)* 時の話であるため、本ファイルは vanilla 用・photon_mini 用それぞれに
    `generate()` を実装する。**vanilla・photon_mini とも、実際にKVキャッシュを保持する実装は
    していない** (tiny_gpt.py も本ファイルも学習/生成のたびに素朴に forward するのみ)。そのため
    メモリは両者とも同じ土台の理論式で計算する: vanilla は 2×n_layer×n_head×head_dim×T×dtype_bytes
    (Part 3: kv_cache_demo.py と同じ式)、photon_mini の context_encoder は同じ式の変数を
    n_layer_encoder/n_head_encoder/head_dim_encoder/M (chunk数) に置き換えたもの。
    vanilla は O(T)、photon_mini (HierGen) は O(T/chunk_size) で増えることを確認する
    (generate_hiergen() の docstring に詳細)。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Part 5 (tiny_gpt.py) を直接 import して継承する ---------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "part_05_train_tiny"))
from tiny_gpt import (  # noqa: E402
    Block,
    CharTokenizer,
    GPTConfig,
    TinyGPT,
    amp_context,
    bits_per_char,
    estimate_loss,
    get_batch,
    load_data,
    perplexity,
)

DEFAULT_DATA = "input.txt"
DEFAULT_OUT = "results.jsonl"


def _finite(x, nd: int = 6):
    """非有限値 (nan/inf) は None に正規化する (tiny_gpt.py と同じ不変条件)。"""
    return round(x, nd) if isinstance(x, (int, float)) and math.isfinite(x) else None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PhotonConfig:
    vocab_size: int
    block_size: int = 256  # T (level-0 系列長。Part5 と同じ256)
    chunk_size: int = 8  # C (1 chunk あたりの token 数)
    converter_len: int = 4  # R (converterが展開するconditioning vector数)
    n_embd: int = 355  # Part5 tiny_gpt.py (10,795,776 params) とほぼ同数(10,770,700)に較正済み
    n_layer_encoder: int = 3  # ContextEncoder の層数 (chunk列を文脈化する側)
    n_head_encoder: int = 5
    n_layer_decoder: int = 3  # ContextDecoder の層数 (chunk内をローカル生成する側)
    n_head_decoder: int = 5
    dropout: float = 0.2

    @property
    def n_chunks(self) -> int:  # M = T / C
        assert self.block_size % self.chunk_size == 0, "block_size は chunk_size で割り切れる必要がある"
        return self.block_size // self.chunk_size

    @property
    def local_window(self) -> int:  # R + C (ContextDecoder のcausal_mask長)
        return self.converter_len + self.chunk_size


# ---------------------------------------------------------------------------
# Bottom-up encoder
# ---------------------------------------------------------------------------


class ContextChunker(nn.Module):
    """複数 token の埋め込みを concat + 線形射影で 1 chunk vector に圧縮する (論文 p.3)。"""

    def __init__(self, n_embd: int, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.proj = nn.Linear(chunk_size * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        assert t % self.chunk_size == 0
        chunks = x.reshape(b, t // self.chunk_size, self.chunk_size * d)
        return self.proj(chunks)


class ContextEncoder(nn.Module):
    """chunk 列を causal self-attention で文脈化する自己回帰 Transformer (Part5 の Block を再利用)。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

    def forward(self, chunk_embeds: torch.Tensor) -> torch.Tensor:
        b, m, d = chunk_embeds.shape
        pos = torch.arange(m, device=chunk_embeds.device)
        x = self.drop(chunk_embeds + self.pos_emb(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)


# ---------------------------------------------------------------------------
# Top-down decoder
# ---------------------------------------------------------------------------


class ContextConverter(nn.Module):
    """1 個の chunk-level latent を converter_len (R) 本の conditioning vector に展開する。
    論文は 1D conv、本実装は Linear+reshape (同じ「1本→R本の学習可能な展開」という機能の簡略化)。"""

    def __init__(self, n_embd: int, converter_len: int):
        super().__init__()
        self.converter_len = converter_len
        self.n_embd = n_embd
        self.expand = nn.Linear(n_embd, converter_len * n_embd)

    def forward(self, chunk_state: torch.Tensor) -> torch.Tensor:
        b, m, d = chunk_state.shape
        return self.expand(chunk_state).view(b, m, self.converter_len, d)


class ContextDecoder(nn.Module):
    """[R本のprefix; chunk_size個のtoken位置] という長さ R+C の窓に causal mask をかけて
    chunk 間で重みを共有しつつ並列デコードする局所自己回帰 Transformer。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)  # block_size = R+C
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        # window: (batch*n_chunks, R+C, D) — chunk を batch 次元に畳み込んで並列処理する
        n, w, d = window.shape
        pos = torch.arange(w, device=window.device)
        x = self.drop(window + self.pos_emb(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)


# ---------------------------------------------------------------------------
# PhotonMini — bottom-up encoder + top-down decoder を組み合わせた全体モデル
# ---------------------------------------------------------------------------


class PhotonMini(nn.Module):
    def __init__(self, config: PhotonConfig):
        super().__init__()
        self.config = config
        C, R, D = config.chunk_size, config.converter_len, config.n_embd

        self.tok_emb = nn.Embedding(config.vocab_size, D)
        self.pos_emb_local = nn.Embedding(config.block_size, D)  # level-0 の位置埋め込み (chunk内相対位置ではなく全体位置)
        self.drop = nn.Dropout(config.dropout)

        self.chunker = ContextChunker(D, C)
        enc_cfg = GPTConfig(
            vocab_size=config.vocab_size,
            block_size=config.n_chunks,
            n_layer=config.n_layer_encoder,
            n_head=config.n_head_encoder,
            n_embd=D,
            dropout=config.dropout,
        )
        self.context_encoder = ContextEncoder(enc_cfg)

        self.converter = ContextConverter(D, R)
        self.start_latent = nn.Parameter(torch.zeros(1, 1, D))  # 学習可能な X_0 (論文 p.3)

        dec_cfg = GPTConfig(
            vocab_size=config.vocab_size,
            block_size=config.local_window,
            n_layer=config.n_layer_decoder,
            n_head=config.n_head_decoder,
            n_embd=D,
            dropout=config.dropout,
        )
        self.context_decoder = ContextDecoder(dec_cfg)

        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, config.vocab_size, bias=False)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        cfg = self.config
        b, t = idx.shape
        assert t == cfg.block_size, f"PhotonMini は固定長 {cfg.block_size} のみ対応 (受け取った長さ: {t})"

        pos = torch.arange(t, device=idx.device)
        x0 = self.drop(self.tok_emb(idx) + self.pos_emb_local(pos)[None, :, :])  # (B,T,D) level-0

        # --- bottom-up: chunk化 → 文脈化 ---
        a = self.chunker(x0)  # (B,M,D)
        x_enc = self.context_encoder(a)  # (B,M,D) 各chunkの粗い状態 (causal: chunk gはchunk<=gのみ依存)

        # --- top-down: chunk g の decode は chunk (g-1) の粗い状態を条件にする (1つ右にシフト) ---
        u = self.converter(x_enc)  # (B,M,R,D)
        start = self.start_latent.expand(b, 1, cfg.converter_len, -1)  # placeholder shape fix below
        start = self.start_latent.unsqueeze(2).expand(b, 1, cfg.converter_len, cfg.n_embd)
        u_shifted = torch.cat([start, u[:, :-1]], dim=1)  # (B,M,R,D)

        # --- 各chunkを [prefix(R); token(C)] の窓にまとめ、chunkをbatch次元に畳み込んで並列decode ---
        m = cfg.n_chunks
        x0_chunks = x0.view(b, m, cfg.chunk_size, cfg.n_embd)  # (B,M,C,D)
        window = torch.cat([u_shifted, x0_chunks], dim=2)  # (B,M,R+C,D)
        window = window.view(b * m, cfg.local_window, cfg.n_embd)

        decoded = self.context_decoder(window)  # (B*M, R+C, D)
        decoded = decoded.view(b, m, cfg.local_window, cfg.n_embd)
        x_hat = decoded[:, :, cfg.converter_len :, :]  # (B,M,C,D) — token位置だけ取り出す
        x_hat = x_hat.reshape(b, t, cfg.n_embd)

        logits = self.head(self.ln_f(x_hat))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # -----------------------------------------------------------------
    # Generation (HierGen) — decode-time の KV 相当メモリを introspect するために必要
    # -----------------------------------------------------------------

    @torch.no_grad()
    def generate_hiergen(self, idx: torch.Tensor, max_new_tokens: int):
        """HierGen: 1chunk進めるたびに (1) 直前chunkのcoarse stateからそのchunkをローカルdecode
        (2) 生成したchunkをbottom-upで再エンコードしてcoarse streamを1つ伸ばす、を繰り返す。
        RecGen (再エンコード省略) は実装しない (out of scope, 01_research.md §4)。

        戻り値: (生成後の token 列, KVメモリ実測ログ)。ログには各chunk生成直後の
        「保持しているcoarse cache (KV_le_g相当) のバイト数」を記録する — 理論上 O(g) で増える。

        **KV bytesの定義について (code-review指摘への対応、2回目)**: 論文 p.4-5 は PHOTON の global KV
        cache を「レベルlのM_l個のlatent単位に対するkeys/valuesの保存」と定義している。これは vanilla の
        KV cache (2×n_layer×n_head×head_dim×T×dtype_bytes) と同じ「multi-head attentionのK/Vペア」という
        土台の量である。本実装の context_encoder はチャンク列に対するcausal self-attention
        (n_layer_encoder層) なので、その理論的なKV cacheサイズは vanilla と全く同じ式の変数を置き換える
        だけで求まる: 2×n_layer_encoder×n_head_encoder×head_dim_encoder×M×dtype_bytes (Mはchunk数)。

        これに加えて、**現在decode中のchunkのlocal decoder KVキャッシュ**も同時に保持される必要がある
        (論文 p.4: 各local decoderの attention span は R_l+C_l に固定=O(1)。1回目のcode-review修正では
        これを含めておらず過小評価だった、との指摘を受けて追加)。local decoder cache は
        2×n_layer_decoder×n_head_decoder×head_dim_decoder×(R+C)×dtype_bytes で、chunk境界ごとに
        破棄される定数サイズ (Mに依存しない)。PhotonMiniの総メモリ = global cache(M) + local cache(定数)。

        **これはあくまで理論上のKVキャッシュサイズ**であり、vanilla・photon_miniのどちらも実際に
        KVキャッシュを保持する実装はしていない (毎回素朴に forward するのみ。tiny_gpt.pyと同じ)。
        したがってスループット実測(実際にはキャッシュなしの再計算コスト)と、このメモリ理論値
        (キャッシュ実装があった場合の見積もり)は測定条件が異なる。両者を単純に掛け合わせた
        「TPM」は論文の一貫した実測系での TPM とは異なる粗い目安に過ぎないことを本文で明記する
        (2回目のcode-review指摘への対応)。
        """
        cfg = self.config
        device = idx.device
        b = idx.shape[0]
        assert idx.shape[1] % cfg.chunk_size == 0, "prompt長はchunk_sizeの倍数である前提の簡易実装"

        coarse_states: list[torch.Tensor] = []  # 各chunkのcontext_encoder出力 (擬似的なcoarse cache)
        mem_log: list[dict] = []

        tokens = idx
        n_prompt_chunks = idx.shape[1] // cfg.chunk_size
        # prompt を1回だけ hierarchical prefill する
        pos0 = torch.arange(tokens.shape[1], device=device)
        x0 = self.tok_emb(tokens) + self.pos_emb_local(pos0)[None, :, :]
        a = self.chunker(x0)
        x_enc = self.context_encoder(a)
        for g in range(n_prompt_chunks):
            coarse_states.append(x_enc[:, g : g + 1, :])

        n_new_chunks = max_new_tokens // cfg.chunk_size
        head_dim_encoder = cfg.n_embd // cfg.n_head_encoder
        head_dim_decoder = cfg.n_embd // cfg.n_head_decoder
        bytes_per_elem = 4  # fp32
        # local decoder cache: 現在decode中のchunkのあいだだけ保持される定数サイズ (Mに依存しない)。
        # docstring参照 (2回目のcode-review指摘: 1回目の修正はglobal cacheのみでlocal cacheが抜けていた)。
        local_decoder_cache_bytes = 2 * cfg.n_layer_decoder * cfg.n_head_decoder * head_dim_decoder * cfg.local_window * bytes_per_elem
        for _ in range(n_new_chunks):
            # coarse cache のバイト数: context_encoder (causal self-attention, n_layer_encoder層) の
            # 理論上のKVキャッシュサイズ (global, Mに応じて増加) + local decoder の理論上のKVキャッシュ
            # サイズ (現在decode中のchunkの分、定数)。vanilla側と同じ「総メモリ」の土台で比較する。
            m = len(coarse_states)
            global_cache_bytes = 2 * cfg.n_layer_encoder * cfg.n_head_encoder * head_dim_encoder * m * bytes_per_elem
            cache_bytes = global_cache_bytes + local_decoder_cache_bytes
            mem_log.append(
                {
                    "n_coarse_states": m,
                    "global_cache_bytes": global_cache_bytes,
                    "local_decoder_cache_bytes": local_decoder_cache_bytes,
                    "coarse_cache_bytes": cache_bytes,
                }
            )

            prev_state = coarse_states[-1]  # (B,1,D) 直前chunkのcoarse state
            u_prev = self.converter(prev_state).squeeze(1)  # (B,R,D)

            # 直前chunkのcoarse stateだけを条件に、chunk_size個のtokenを1つずつローカル生成する
            local_tokens: list[torch.Tensor] = []
            window = u_prev  # (B, R, D) から始めて token を1つずつ追加していく
            last_tok = tokens[:, -1:]
            for _j in range(cfg.chunk_size):
                tok_emb = self.tok_emb(last_tok) if local_tokens else torch.zeros(b, 0, cfg.n_embd, device=device)
                cur = torch.cat([window, tok_emb], dim=1) if local_tokens else window
                w = min(cur.shape[1], cfg.local_window)
                decoded = self.context_decoder(cur[:, -w:, :])
                logits = self.head(self.ln_f(decoded[:, -1:, :]))
                next_tok = torch.argmax(logits, dim=-1)  # (B,1) greedy (再現性のため)
                local_tokens.append(next_tok)
                last_tok = next_tok
                window = cur

            new_chunk_tokens = torch.cat(local_tokens, dim=1)  # (B,C)
            tokens = torch.cat([tokens, new_chunk_tokens], dim=1)

            # bottom-up re-encode (HierGen): 新しいchunkをchunk化し、coarse streamを1つ伸ばす
            new_pos = torch.arange(tokens.shape[1] - cfg.chunk_size, tokens.shape[1], device=device)
            new_x0 = self.tok_emb(new_chunk_tokens) + self.pos_emb_local(new_pos)[None, :, :]
            new_a = self.chunker(new_x0)  # (B,1,D)
            # 簡易実装: 新chunkのcoarse stateは「自分自身のchunk埋め込み」をcontext_encoderに
            # 1ステップぶん通したものとして近似する (厳密なincremental encodingではないが、
            # coarse cacheのサイズ増加=メモリ計測という本Partの目的には十分)
            combined = torch.cat(coarse_states + [new_a], dim=1)
            encoded = self.context_encoder(combined)
            coarse_states.append(encoded[:, -1:, :])

        return tokens, mem_log


# ---------------------------------------------------------------------------
# Vanilla 側の generate + KV cache 実測 (比較対象。Part3のkv_cache_demo.pyと同じ理論式)
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_vanilla(model: TinyGPT, idx: torch.Tensor, max_new_tokens: int):
    """vanilla TinyGPTの素朴な生成。KVキャッシュを持たず毎回全系列を再計算する代わりに、
    「毎ステップ保持しなければならない系列長ぶんのtoken表現」をO(T)で実測する
    (Part5のTinyGPTはKV cache未実装のため、理論式 2*n_layer*n_head*head_dim*T*dtype_bytes
    で比較する。Part3のkv_cache_demo.pyと同じ定義)。"""
    device = idx.device
    tokens = idx
    mem_log: list[dict] = []
    cfg = model.config
    bytes_per_elem = 4  # fp32
    for _ in range(max_new_tokens):
        t = tokens.shape[1]
        kv_bytes = 2 * cfg.n_layer * cfg.n_head * (cfg.n_embd // cfg.n_head) * t * bytes_per_elem
        mem_log.append({"seq_len": t, "kv_cache_bytes_theoretical": kv_bytes})
        ctx = tokens[:, -cfg.block_size :]
        logits, _ = model(ctx)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_tok], dim=1)
    return tokens, mem_log


# ---------------------------------------------------------------------------
# Selftest — 実機(MPS実走)なしでCPU上で通るロジック検証
# ---------------------------------------------------------------------------


def selftest() -> int:
    failures: list[str] = []
    torch.manual_seed(0)

    tok = CharTokenizer("hello world, this is a small vocabulary for selftest purposes only")
    cfg = PhotonConfig(
        vocab_size=tok.vocab_size,
        block_size=16,
        chunk_size=4,
        converter_len=2,
        n_embd=24,
        n_layer_encoder=2,
        n_head_encoder=2,
        n_layer_decoder=2,
        n_head_decoder=2,
        dropout=0.0,
    )
    model = PhotonMini(cfg)

    # forward shape + finite loss
    idx = torch.randint(0, tok.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, tok.vocab_size, (2, cfg.block_size))
    logits, loss = model(idx, targets)
    if tuple(logits.shape) != (2, cfg.block_size, tok.vocab_size):
        failures.append(f"forward shape 不一致: {tuple(logits.shape)}")
    if loss is None or not math.isfinite(loss.item()):
        failures.append("loss が有限でない")

    # 過学習チェック: 同じ小バッチを繰り返し学習すればlossは大幅に下がるはず
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first_loss = last_loss = None
    for step in range(120):
        _, step_loss = model(idx, targets)
        optimizer.zero_grad(set_to_none=True)
        step_loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = step_loss.item()
        last_loss = step_loss.item()
    if first_loss is None or last_loss is None or not (last_loss < first_loss * 0.5):
        failures.append(f"過学習でlossが十分下がらない: first={first_loss} last={last_loss}")

    # 階層的causality: chunk g の出力は「chunk >= g+1 の未来token」を変更しても不変であるべき
    # (decoderはchunk(g-1)のcoarse stateとchunk自身の過去token位置しか見ないため)
    model.eval()
    base = torch.randint(0, tok.vocab_size, (1, cfg.block_size))
    perturbed = base.clone()
    last_chunk_start = cfg.block_size - cfg.chunk_size
    perturbed[0, last_chunk_start:] = (perturbed[0, last_chunk_start:] + 1) % tok.vocab_size
    logits_base, _ = model(base)
    logits_pert, _ = model(perturbed)
    first_chunk_end = cfg.chunk_size
    if not torch.allclose(logits_base[0, :first_chunk_end], logits_pert[0, :first_chunk_end], atol=1e-5):
        failures.append("階層的causality違反: 最終chunkの変更が先頭chunkのlogitsに影響した")

    # 逆方向: 先頭chunkを変えると、coarse streamを経由して最終chunkのlogitsは変わるはず
    # (もし変わらなければ、bottom-up encoderがそもそも情報を伝搬していない=実装バグの疑い)
    perturbed2 = base.clone()
    perturbed2[0, :cfg.chunk_size] = (perturbed2[0, :cfg.chunk_size] + 1) % tok.vocab_size
    logits_pert2, _ = model(perturbed2)
    if torch.allclose(logits_base[0, last_chunk_start:], logits_pert2[0, last_chunk_start:], atol=1e-5):
        failures.append("bottom-up伝搬の疑い: 先頭chunkの変更が最終chunkのlogitsに全く伝わっていない")

    # パラメータ数が有限かつ正であることの確認 (num_params()のスモークテスト)
    if model.num_params() <= 0:
        failures.append("num_params()が0以下")

    # KV bytes 理論式のセルフチェック (vanilla側): 2*n_layer*n_head*head_dim*T*4byte
    vt_cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=24, dropout=0.0)
    vt_model = TinyGPT(vt_cfg)
    _, mem_log = generate_vanilla(vt_model, torch.randint(0, tok.vocab_size, (1, 4)), max_new_tokens=4)
    expected = 2 * vt_cfg.n_layer * vt_cfg.n_head * (vt_cfg.n_embd // vt_cfg.n_head) * 4 * 4
    if mem_log[0]["kv_cache_bytes_theoretical"] != expected:
        failures.append(f"vanilla KV bytes理論式が期待値と不一致: {mem_log[0]} vs expected={expected}")

    # HierGen生成のcoarse cacheが実際にchunk数に応じて増加することの確認 (O(T/C)の方向性)
    gen_tokens, hier_mem_log = model.generate_hiergen(base[:, : cfg.chunk_size * 2], max_new_tokens=cfg.chunk_size * 2)
    if gen_tokens.shape[1] != cfg.chunk_size * 4:
        failures.append(f"generate_hiergenの出力長が不正: {gen_tokens.shape[1]}")
    if len(hier_mem_log) < 2 or hier_mem_log[-1]["coarse_cache_bytes"] <= hier_mem_log[0]["coarse_cache_bytes"]:
        failures.append("HierGenのcoarse cacheがchunk進行に応じて増加していない")

    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        "SELFTEST PASSED (forward shape / 過学習でloss減少 / 階層的causality(前方向+後方向) "
        "/ num_params / vanilla KV bytes理論式 / HierGen coarse cache増加)"
    )
    return 0


# ---------------------------------------------------------------------------
# Parameter matching helper (uv run python photon_mini.py --count-params で確認)
# ---------------------------------------------------------------------------


def build_default_config(vocab_size: int) -> PhotonConfig:
    return PhotonConfig(vocab_size=vocab_size)


# ---------------------------------------------------------------------------
# Training loop (photon_mini.py --train) — tiny_gpt.py の train() と対になるよう
# 同じ規律 (fp32既定 / MPS fallback保険 / RNG分離 / peak mem サンプリング) で実装する
# ---------------------------------------------------------------------------


def run_meta_photon(args: argparse.Namespace, model: "PhotonMini", device: torch.device, mps_fallback_triggered: bool) -> dict:
    cfg = model.config
    return {
        "kind": "run-meta",
        "model": "photon_mini",
        "config": {
            "vocab_size": cfg.vocab_size,
            "block_size": cfg.block_size,
            "chunk_size": cfg.chunk_size,
            "converter_len": cfg.converter_len,
            "n_embd": cfg.n_embd,
            "n_layer_encoder": cfg.n_layer_encoder,
            "n_head_encoder": cfg.n_head_encoder,
            "n_layer_decoder": cfg.n_layer_decoder,
            "n_head_decoder": cfg.n_head_decoder,
            "dropout": cfg.dropout,
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
        "peak_mem_method": "current_allocated_memory() sampled every iteration right after backward()+optimizer.step(), max taken as peak approximation (tiny_gpt.py run_meta()と同じ定義)",
    }


def train_photon(args: argparse.Namespace) -> None:
    """PhotonMini を TinyShakespeare で学習し、tiny_gpt.py --train と直接比較できる
    results.jsonl (train/val loss, PPL, BPC, elapsed, peak mem) を出力する。"""
    device = torch.device(args.device)
    if device.type == "mps" and args.dtype != "float32":
        print("警告: MPS では bf16 autocast が不安定という報告があります。fp32(--dtype float32)を推奨します。")

    torch.manual_seed(args.seed)
    if device.type == "mps":
        torch.mps.manual_seed(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    eval_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    tok, train_data, val_data = load_data(Path(args.data))
    cfg = PhotonConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        chunk_size=args.chunk_size,
        converter_len=args.converter_len,
        n_embd=args.n_embd,
        n_layer_encoder=args.n_layer_encoder,
        n_head_encoder=args.n_head_encoder,
        n_layer_decoder=args.n_layer_decoder,
        n_head_decoder=args.n_head_decoder,
        dropout=args.dropout,
    )
    model = PhotonMini(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"vocab_size={tok.vocab_size} params={model.num_params():,} device={device} chunk_size={cfg.chunk_size}")

    mps_fallback_triggered = False
    records: list[dict] = []
    peak_mem_bytes = 0

    import warnings

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
                peak_mem_bytes = max(peak_mem_bytes, torch.mps.current_allocated_memory())

            if it % args.eval_interval == 0 or it == args.max_iters - 1:
                if device.type == "mps":
                    torch.mps.synchronize()
                train_loss = estimate_loss(
                    model, train_data, args.block_size, args.batch_size, device, eval_generator, args.eval_iters, args.dtype
                )
                val_loss = estimate_loss(
                    model, val_data, args.block_size, args.batch_size, device, eval_generator, args.eval_iters, args.dtype
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
                    f"| ppl(val) {perplexity(val_loss):7.2f} bpc(val) {bits_per_char(val_loss):.3f} | {elapsed:7.1f}s"
                )

        for w in caught:
            msg = str(w.message)
            if "MPS backend" in msg or "fall back" in msg.lower():
                mps_fallback_triggered = True

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        meta = run_meta_photon(args, model, device, mps_fallback_triggered)
        meta["peak_mem_bytes"] = peak_mem_bytes if device.type == "mps" else None
        f.write(json.dumps(meta) + "\n")

    print(f"done. wrote {len(records)} records + run-meta to {args.out}")
    if mps_fallback_triggered:
        print("警告: MPS fallback が発火しました。計測値の一部がCPU実行分を含む可能性があります。")


# ---------------------------------------------------------------------------
# Efficiency sweep (photon_mini.py --measure-efficiency) — KV bytes / throughput を
# context 長ごとに vanilla(TinyGPT) vs photon_mini(HierGen) で比較する。
# 学習済み重みは不要 (アーキテクチャのメモリ・計算特性の測定であり、生成テキストの質は問わない)。
# ---------------------------------------------------------------------------


def measure_efficiency(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    tok, _, _ = load_data(Path(args.data))

    photon_cfg = PhotonConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        chunk_size=args.chunk_size,
        converter_len=args.converter_len,
        n_embd=args.n_embd,
        n_layer_encoder=args.n_layer_encoder,
        n_head_encoder=args.n_head_encoder,
        n_layer_decoder=args.n_layer_decoder,
        n_head_decoder=args.n_head_decoder,
        dropout=0.0,
    )
    photon_model = PhotonMini(photon_cfg).to(device).eval()

    vanilla_cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=6, n_head=6, n_embd=384, dropout=0.0
    )
    vanilla_model = TinyGPT(vanilla_cfg).to(device).eval()

    print(f"photon_mini params={photon_model.num_params():,} / vanilla params={vanilla_model.num_params():,}")

    gen_len = photon_cfg.chunk_size * 8  # 計測窓を長めにとりtimingノイズを減らす
    # context_encoder の pos_emb は n_chunks=block_size/chunk_size 個の位置しか持たないため、
    # 生成後の総chunk数 (prompt由来+新規4chunk) が n_chunks を超えないよう ctx_len を block_size-gen_len に制限する
    # (vanilla側は generate_vanilla 内で block_size window にクリップして同じ問題を回避している)
    context_lengths = [
        photon_cfg.chunk_size * k
        for k in (2, 4, 8, 16, 24, 32)
        if photon_cfg.chunk_size * k + gen_len <= photon_cfg.block_size
    ]
    records: list[dict] = []
    n_reps = args.efficiency_reps

    for ctx_len in context_lengths:
        photon_tok_s_samples: list[float] = []
        van_tok_s_samples: list[float] = []
        photon_peak_kv_bytes = van_peak_kv_bytes = None

        for _rep in range(n_reps):
            prompt = torch.randint(0, tok.vocab_size, (1, ctx_len), device=device)

            t0 = time.time()
            _, hier_mem_log = photon_model.generate_hiergen(prompt, max_new_tokens=gen_len)
            if device.type == "mps":
                torch.mps.synchronize()
            photon_elapsed = time.time() - t0
            if photon_elapsed > 0:
                photon_tok_s_samples.append(gen_len / photon_elapsed)
            photon_peak_kv_bytes = max(m["coarse_cache_bytes"] for m in hier_mem_log)

            t0 = time.time()
            _, van_mem_log = generate_vanilla(vanilla_model, prompt, max_new_tokens=gen_len)
            if device.type == "mps":
                torch.mps.synchronize()
            van_elapsed = time.time() - t0
            if van_elapsed > 0:
                van_tok_s_samples.append(gen_len / van_elapsed)
            van_peak_kv_bytes = max(m["kv_cache_bytes_theoretical"] for m in van_mem_log)

        photon_tok_s_samples.sort()
        van_tok_s_samples.sort()
        photon_tok_s = photon_tok_s_samples[len(photon_tok_s_samples) // 2] if photon_tok_s_samples else None
        van_tok_s = van_tok_s_samples[len(van_tok_s_samples) // 2] if van_tok_s_samples else None

        rec = {
            "kind": "efficiency-point",
            "context_len": ctx_len,
            "gen_len": gen_len,
            "n_reps": n_reps,
            "photon_tok_per_s_median": _finite(photon_tok_s),
            "vanilla_tok_per_s_median": _finite(van_tok_s),
            "photon_tok_per_s_samples": [_finite(x) for x in photon_tok_s_samples],
            "vanilla_tok_per_s_samples": [_finite(x) for x in van_tok_s_samples],
            "photon_peak_kv_bytes": photon_peak_kv_bytes,
            "vanilla_peak_kv_bytes": van_peak_kv_bytes,
            "kv_bytes_reduction_x": _finite(van_peak_kv_bytes / photon_peak_kv_bytes) if photon_peak_kv_bytes else None,
            "throughput_reduction_x": _finite(photon_tok_s / van_tok_s) if van_tok_s else None,
            "tpm_improvement_x": _finite((photon_tok_s / photon_peak_kv_bytes) / (van_tok_s / van_peak_kv_bytes))
            if van_tok_s and photon_tok_s and photon_peak_kv_bytes
            else None,
        }
        records.append(rec)
        print(
            f"context={ctx_len:4d} (n={n_reps} median) | photon {photon_tok_s:7.2f} tok/s, KV={photon_peak_kv_bytes:8d}B "
            f"| vanilla {van_tok_s:7.2f} tok/s, KV={van_peak_kv_bytes:8d}B "
            f"| KV削減率={rec['kv_bytes_reduction_x']} | TPM改善率={rec['tpm_improvement_x']}"
        )

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"done. wrote {len(records)} records to {args.out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true", help="CPU/実機なしで通るロジック検証")
    mode.add_argument("--count-params", action="store_true", help="vocab_size=65(TinyShakespeare相当)でパラメータ数を表示")
    mode.add_argument("--train", action="store_true", help="PhotonMiniをTinyShakespeareで学習しresults.jsonlを出力")
    mode.add_argument(
        "--measure-efficiency",
        action="store_true",
        help="未学習のphoton_mini(HierGen) vs vanilla(TinyGPT)でKV bytes/throughputをcontext長スイープ計測",
    )
    ap.add_argument("--data", type=str, default=DEFAULT_DATA)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--converter-len", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=355)
    ap.add_argument("--n-layer-encoder", type=int, default=3)
    ap.add_argument("--n-head-encoder", type=int, default=5)
    ap.add_argument("--n-layer-decoder", type=int, default=3)
    ap.add_argument("--n-head-decoder", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--efficiency-reps", type=int, default=5, help="--measure-efficiency の各context長での反復回数(median採用)")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())

    if args.count_params:
        cfg = PhotonConfig(
            vocab_size=65,
            chunk_size=args.chunk_size,
            converter_len=args.converter_len,
            n_embd=args.n_embd,
            n_layer_encoder=args.n_layer_encoder,
            n_head_encoder=args.n_head_encoder,
            n_layer_decoder=args.n_layer_decoder,
            n_head_decoder=args.n_head_decoder,
        )
        model = PhotonMini(cfg)
        print(f"config={cfg}")
        print(f"num_params={model.num_params():,}")
        return

    if args.train:
        train_photon(args)
        return

    if args.measure_efficiency:
        measure_efficiency(args)
        return


if __name__ == "__main__":
    main()

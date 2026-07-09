# llm-internals-companion

連載「ローカルLLMで覗く 言語モデルの中身 — トークンから PHOTON まで」のコード。各 Part の数値は
このリポジトリのコードを Apple Silicon 上で自分で走らせて計測したものです（他者ベンチの引用ではありません）。

> 各 Part のコードは `part-NN` タグで固定しています（例: Part 1 → `part-01`）。記事の各回は、そのタグ時点のコードに対応します。

## レイアウト

```
llm_internals/                       # 共有ユーティリティ（Part1-4は各exampleで完結。Part5/6共有はexampleの直接importで行う）
examples/part_01_token_loop/
  token_loop.py                      # 自己回帰ループを1トークンずつ覗く + 累積 wall-time の線形を実測
  selftest.py                        # MLX 非依存のロジック検証（実機なしで通る）
examples/part_02_attention/
  attention_from_scratch.py          # self-attention 最小実装 + 系列長 T スイープの実測
examples/part_03_kv_cache/
  kv_cache_demo.py                   # mlx_lm の内部 KV キャッシュを introspect + context スイープの実測
examples/part_04_sampling_self_consistency/
  self_consistency.py                # greedy vs self-consistency(maj@N) を GSM8K で実測（bootstrap CI付き）
  gsm8k_subset.jsonl                 # 固定 150 問（official test.jsonl の先頭150件、committed）
examples/part_05_train_tiny/
  tiny_gpt.py                        # decoder-only Transformer をゼロから学習 + train/val loss・PPL・BPC・メモリ実測
  input.txt                          # TinyShakespeare（karpathy/char-rnn由来、40,000行・1.1MB）
pyproject.toml                       # extras: [mlx]（Part1-4）/ [torch]（Part5-6）
```

## Part 1: token_loop

### セットアップ（Apple Silicon）

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[mlx]'      # mlx-lm を導入（version は lockfile で pin）
```

### 実行

```bash
cd examples/part_01_token_loop

# トークン化の観察（モデル DL あり・初回のみ数分）
python token_loop.py --demo-tokenize --model mlx-community/Llama-3.2-1B-Instruct-4bit

# 自己回帰ループを1トークンずつ + 各ステップの top-k を表示
python token_loop.py --prompt "The capital of France is" --max-tokens 32

# 累積 wall-time の線形を実測 → results.jsonl（記事の chart-01 / tok/s の元データ）
python token_loop.py --bench --max-tokens 128 --reps 3 --out results.jsonl

# MLX 不要のロジック検証
python selftest.py
```

### 再現性（このリポジトリの不変条件）

- **モデルを revision で pin**: `--model` に固定 repo を渡し、`run-meta`（出力 JSONL の `meta`）に
  model id / mlx-lm version / python / platform を記録します。`trust_remote_code` は使いません。
- **cached decode で測る**: `--bench` は最初のトークン（prefill 込み）を除いた decode 区間の累積時間で
  線形を見ます。1トークンあたりの時間は decode フェーズの単価です。
- **warmup + median**: 1 回 warmup してから `--reps` 回計測し、トークン数ごとに median を取ります。
- **数値は環境依存**: tok/s も top-k 確率も「そのモデル・量子化（例 4bit）・機種（例 M5 Pro）」の値で、
  一般化しません。所要時間（モデル DL を含む）を記事に明記しています。free ≠ fast。

### 計測環境（記事の数値）

- 機種: Apple M5 Pro 48GB / macOS
- モデル: `mlx-community/Llama-3.2-1B-Instruct-4bit`（revision は lockfile / run-meta に記録）
- mlx-lm: （実行時の version を run-meta に記録）

## Part 2: attention_from_scratch

self-attention を numpy だけで最小実装し、系列長 T に対する計算量を実測します（MLX も外部モデルも不要）。

```bash
cd examples/part_02_attention

# causal attention 行列を表示（過去にだけ重みが分散・上三角=未来=0）
python attention_from_scratch.py --demo

# 系列長 T スイープ: 射影(線形)と attention コア(二乗)の wall-time を分けて計測
python attention_from_scratch.py --bench --reps 7 --out results.jsonl

# MLX 不要のロジック検証（softmax 行和=1・causal リーク無し）
python attention_from_scratch.py --selftest
```

### 記事の数値（Apple M5 Pro / numpy 2.5.0 / d=128 / 7 rep median）

- attention コア（`softmax(QKᵀ/√d)·V`）は系列長 T が 2 倍で約 4 倍（O(T²)。T=2048 で倍率 4.23）
- Q/K/V 射影は約 2 倍（O(T)・線形）
- スコア行列メモリは T×T×4byte（T=2048 で 16 MiB）。理論値
- 小さい T（例 128 で倍率 2.23）では、二乗項が定数項・線形項に隠れて見えにくい（O(T²) は漸近計算量）

## Part 3: kv_cache_demo

mlx_lm の内部 KV キャッシュ実装（`KVCache`）を直接 introspect し、キャッシュサイズが context 長に線形で
効くこと、そして decode 速度が計算量ではなくキャッシュの読み出し量で決まる（memory-bound）ことを実測します。

```bash
cd examples/part_03_kv_cache

# 256 境界をまたぐ細かいスイープで「罠」(raw nbytes の階段関数) を見せる
python kv_cache_demo.py --probe --around 256 --span 6

# context 1K-32K で KV バイト数を実測（理論式・論理長・確保容量の三点一致）
python kv_cache_demo.py --kv --contexts 1024,2048,4096,8192,16384,32768 --out results_kv.jsonl

# context 1K-32K で decode tok/s を実測（memory-bound の実測）
python kv_cache_demo.py --decode --contexts 1024,2048,4096,8192,16384,32768 --out results_decode.jsonl

# mlx 不要のロジック検証（理論式・padding 計算）
python kv_cache_demo.py --selftest
```

### 記事の数値（Apple M5 Pro / mlx-lm 0.31.3 / mlx 0.31.2 / `mlx-community/Llama-3.2-1B-Instruct-4bit`）

- モデル構成: n_layers=16, n_heads=32, **n_kv_heads=8**, head_dim=64（GQA group factor = 4）。KV キャッシュの
  dtype は fp16（重みが 4bit 量子化でも KV は既定で fp16 のまま）
- **「罠」**: mlx_lm の `KVCache` は `step=256` 単位でキャッシュ配列を事前確保する（`mlx_lm/models/cache.py`
  を直読して確認）。`.nbytes` は確保容量ベースなので、context が 256→257 と 1 トークン増えただけで
  8.0 MiB → 16.0 MiB（ちょうど2倍）にジャンプする。論理長（`offset`）でスライスした実データは
  8.000 → 8.031 MiB としか増えない
- context 1,024〜32,768（256 の倍数）では raw（確保容量）・state（論理長）・理論式の 3 つが完全一致し、
  context が2倍で KV バイト数も正確に2倍（32,768 context で 1,024 MiB = 1 GiB）
- decode 速度: context 1,024→32,768（32倍）で 292.1 → 119.0 tok/s（約 2.45 倍低下）。新規計算量は
  context 長によらず一定なので、この低下はキャッシュ読み出しコストが効いている証拠（memory-bound）

## Part 4: self_consistency

GSM8K（小学校算数文章題）で、greedy（1回だけ生成）と self-consistency（temperature>0 で N 回
サンプリングし多数決）の正答率を比較します。N=1 の基準点は greedy で別途生成し、N∈{3,5,10,20} は
M=20 本のサンプルプールから非復元抽出した部分集合で見積もる（生成コストを N 毎に払わない標準的な
評価トリック）。bootstrap CI で「有意な差か、ノイズ内か」を判定します。

```bash
cd examples/part_04_sampling_self_consistency

# greedy 1本 + サンプル M=20本を150問全てで生成（時間がかかる。1問あたり約20-25秒 × 150問）
python self_consistency.py --sample --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --m 20 --temperature 0.7 --top-p 0.95 --out results_samples.jsonl

# 保存済みサンプルから N∈{1,3,5,10,20} の maj@N 正答率を bootstrap CI 付きで算出（再生成なし）
python self_consistency.py --analyze --in results_samples.jsonl --out results_analysis.jsonl

# mlx 不要のロジック検証（answer抽出・多数決・bootstrap CI）
python self_consistency.py --selftest
```

### データセット

- `gsm8k_subset.jsonl`: 公式 [openai/grade-school-math](https://github.com/openai/grade-school-math) の
  `test.jsonl` 先頭 150 問（committed、決定論的選択）。MIT License。

### 記事の数値（Apple M5 Pro / `mlx-community/Qwen2.5-1.5B-Instruct-4bit` / mlx-lm 0.31.3 / temperature=0.7, top_p=0.95）

- GSM8K 150問での maj@N 正答率（bootstrap 95% CI, 2,000 resamples）:
  - N=1（greedy）: 54.0% [46.0%, 62.0%]
  - N=3: 62.2% [56.5%, 67.7%]（+8.2pp）
  - N=5: 67.9% [62.2%, 73.8%]（+5.7pp）
  - N=10: 72.8% [66.8%, 78.8%]（+4.9pp）
  - N=20: 76.7% [70.0%, 83.3%]（+3.9pp）— N を倍にするごとに伸びが縮む（diminishing returns）
- コスト: N=1 は150クエリ/183.6秒、N=20 は3,000クエリ/3,672.3秒（正確に20倍）。正答率は約1.42倍にしかならない
- 内訳: 34/150問が self-consistency で「救われた」（greedy不正解→maj@20正解）、0/150問が「悪化」（greedy正解→maj@20不正解、本実験では未観測）、35/150問はmaj@20でも不正解のまま（うち6問は20本中10本以上が同じ誤答=系統的、残り17問は票が分散する独立ノイズ寄り）
- 系統的誤りの例（Q41、ドラゴンと投槍の問題）: 20本中12本が中間計算「1,200 feet」で止まり最後の引き算を飛ばして誤答（同一設定で再生成したcompletion本文を読んで確認済み）。独立ノイズは多数決で均せるが、相関した系統誤りには効かない

## Part 5: tiny_gpt

Part 1-4 は既存モデルの重みを使った推論だけを扱ってきました。Part 5 では nanoGPT 級（Karpathy 標準
config: n_layer=6, n_head=6, n_embd=384, block_size=256, ≈10.6M params）の decoder-only Transformer を
TinyShakespeare で **ゼロから学習** し、train/val loss・perplexity・bits-per-char・所要時間・ピーク
メモリを実測します。

```bash
cd examples/part_05_train_tiny
# torch を追加導入（mlx extraと共存させる。--extra torch 単体だと mlx が入れ替わりで消えるため注意）
uv sync --extra mlx --extra torch --extra dev

# 標準config（Karpathy nanoGPT 相当, ≈10.6M params）で学習。fp32既定・MPS優先
python tiny_gpt.py --train --out results.jsonl

# 学習パラメータを変える場合の例（縮小config・短時間で様子を見る）
python tiny_gpt.py --train --n-layer 4 --n-head 4 --n-embd 128 --block-size 64 \
  --max-iters 2000 --out results_small.jsonl

# torch実装のロジック検証（実機=MPS実走なしでCPU上で通る。tokenizer roundtrip/forward shape/過学習でloss減少/PPL・BPC定義/causal mask整合）
python tiny_gpt.py --selftest
```

### データセット

- `input.txt`: TinyShakespeare（[karpathy/char-rnn](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
  由来、約40,000行・1.11MB）。**ライセンス表記は "More Information Needed"（不明）** —
  Hugging Face の dataset card でも明確なライセンスは確認できないため、出典と教育目的での広範な
  利用実績（nanoGPT/char-rnn 双方の公式サンプル）を明記した上で使用する。

### 再現性・計測方法の注記（このリポジトリの不変条件）

- **fp32 既定**: MPS の bf16 autocast は環境によって不安定という報告があるため（`01_research.md`
  参照）、`--dtype float32` を既定にする。`PYTORCH_ENABLE_MPS_FALLBACK=1` を既定で有効にし、未対応
  演算が発生した場合は `warnings` を捕捉して `run-meta.mps_fallback_triggered` に記録する（発火時は
  計測値の一部がCPU実行分を含む可能性があるフラグとして機能する）。
- **ピークメモリの近似**: `torch.mps` には CUDA の `max_memory_allocated()` に相当する peak tracker が
  無い。`torch.mps.current_allocated_memory()` を各評価チェックポイントでサンプリングし、その最大値を
  peak の近似値として `run-meta.peak_mem_bytes` に記録する（近似方法であることを `run-meta.peak_mem_method`
  に明記）。
- **MLX(Part1-4) と PyTorch(Part5-6) の指標は混ぜない**: メモリ・スループットの定義がライブラリ間で
  異なるため、別建てで報告する。

### 記事の数値（実機実走待ち）

- **未確定**: 実際の学習実走はザワが Apple M5 Pro 48GB 上で行う（このリポジトリの数値は他者ベンチの
  引用ではなく自分で測ったものにする、という連載全体の方針に基づく）。実走完了後、train/val loss 曲線・
  perplexity・bits-per-char・所要時間・ピークメモリの実測値をここに追記する。
- 参考値（nanoGPT公式ベンチマーク、A100 GPU実測・自分の実測ではないため直接引用しない）: 標準config
  で validation loss 1.4697（`01_research.md` 参照）。GPU と MPS では収束曲線・所要時間が異なりうる点に
  留意する。

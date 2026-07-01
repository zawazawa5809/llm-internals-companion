# llm-internals-companion

連載「ローカルLLMで覗く 言語モデルの中身 — トークンから PHOTON まで」のコード。各 Part の数値は
このリポジトリのコードを Apple Silicon 上で自分で走らせて計測したものです（他者ベンチの引用ではありません）。

> 各 Part のコードは `part-NN` タグで固定しています（例: Part 1 → `part-01`）。記事の各回は、そのタグ時点のコードに対応します。

## レイアウト

```
llm_internals/                       # 共有ユーティリティ（最小から開始。measure 抽象化は Part3/5 で追加）
examples/part_01_token_loop/
  token_loop.py                      # 自己回帰ループを1トークンずつ覗く + 累積 wall-time の線形を実測
  selftest.py                        # MLX 非依存のロジック検証（実機なしで通る）
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

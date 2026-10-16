# Velo-Verify-Gemini — Hybrid Logic Compiler

> **『自転車の青切符をGeminiで判定しようとしたら、2008年度に書いた卒論に救われた話』検証用プロトタイプ**
>
> 2026年4月1日施行の**自転車交通反則通告制度（青切符）**（令和6年法律第34号）を題材に、最新LLM（Gemini / Claude / Gemma）が法規判定で陥る「もっともらしい嘘（ハルシネーション）」を、2008年当時の卒論技術（決定論的パース + コサイン類似度）で検出・矯正するハイブリッド判定システム。

---

## 📌 目次

- [概要 & 解決する課題](#概要--解決する課題)
- [🤖 対応LLM一覧](#-対応llm一覧)
- [🏛️ アーキテクチャ](#️-アーキテクチャ)
- [🚀 クイックスタート](#-クイックスタート)
  - [前提条件](#前提条件)
  - [インストール](#インストール)
  - [環境変数・認証設定](#環境変数認証設定)
  - [法令データの取得](#法令データの取得)
- [💻 使い方](#-使い方)
  - [CLI（コマンドライン）](#cliコマンドライン)
  - [Web UI（FastAPI + Next.js）](#web-ui-fastapi--nextjs)
  - [Layer 1 単体実行（APIキー不要）](#layer-1-単体実行apiキー不要)
- [📊 2×2 マトリクス検証（前処理 × 推論 Thinking）](#-22-マトリクス検証前処理--推論-thinking)
- [📁 プロジェクト構成](#-プロジェクト構成)
- [🧪 テスト](#-テスト)
- [📜 対象法令 & ライセンス](#-対象法令--ライセンス)

---

## 概要 & 解決する課題

2026年4月から自転車の交通違反に「青切符」が導入され、113種類もの違反に対して反則金が課されます。
この判定を最新LLM単体に任せると、条文構造の複雑さ（原則・例外・例外の例外・政令への委任など）により**もっともらしい誤答**が頻発します。

### LLMが陥る3つの失敗パターン

| パターン | 内容 | 具体例 |
|---|---|---|
| **階層無視** | 原則→例外→例外の例外を平坦化 | 「自転車は車道通行が原則」に囚われ、歩道通行可能な例外規定を見落とす |
| **数値捏造** | 条文にない反則金額や根拠条文を勝手に補完 | 存在しない「3,000円」や無関係な条文番号を出力 |
| **参照欠落** | 「政令で定める者」等の委任規定を追跡できない | 70歳以上の高齢者や13歳未満の歩道通行許可要件を無視 |

本プロジェクトでは、2008年式の**決定論的法令XMLパーサ（AST生成）**と**ベクトル空間モデル（TF-IDF cos類似度）**を前処理エンジン（Layer 1）とし、条文と反則金情報を「絶対的根拠」としてLLM（Layer 2）に注入することで、ハルシネーションを物理的に抑止します。

---

## 🤖 対応LLM一覧

本システムは Google Gemini だけでなく、Anthropic Claude や Ollama 経由のローカルLLMなど、複数ファミリーのモデルに対応しています。

| モデルキー (`--model`) | モデル名 / バージョン | プロバイダ / 実行基盤 | デフォルト識別子 | グラウンディング対応 | 必要な認証 / 設定 |
|---|---|---|---|:---:|---|
| **`flash`** *(デフォルト)* | **Gemini 3.7 Flash** | Google AI Studio / Vertex AI | `gemini-3.7-flash` | Layer 1 / Web Search / なし | `GEMINI_API_KEY` または GCP ADC |
| **`flash_3_8`** | **Gemini 3.8 Flash** | Google AI Studio / Vertex AI | `gemini-3.8-flash` | Layer 1 / Web Search / なし | `GEMINI_API_KEY` または GCP ADC |
| **`flash_lite`** | **Gemini 3.5 Flash-Lite** | Google AI Studio / Vertex AI | `gemini-3.5-flash-lite` | Layer 1 / Web Search / なし | `GEMINI_API_KEY` または GCP ADC |
| **`pro`** | **Gemini 3.1 Pro** | Google AI Studio / Vertex AI | `gemini-3.1-pro-preview` | Layer 1 / Web Search / なし | `GEMINI_API_KEY` または GCP ADC |
| **`claude`** | **Claude Opus 4.7** | Google Cloud Vertex AI | `claude-opus-4-7@default` | Layer 1 / なし | GCP ADC（Model Gardenで承諾要） |
| **`claude_sonnet`** | **Claude Sonnet 4.6** | Google Cloud Vertex AI | `claude-sonnet-4-6@default` | Layer 1 / なし | GCP ADC（Model Gardenで承諾要） |
| **`gemma3`** | **Gemma 3 4B** | Ollama (ローカルLLM) | `gemma3:4b` | Layer 1 / なし | なし（Ollama サーバー起動のみ） |
| *その他ローカル* | **Llama / Qwen / Mistral / Phi** | Ollama (ローカルLLM) | 各モデル名 | Layer 1 / なし | なし（Ollama サーバー起動のみ） |

> [!NOTE]
> - **モデルIDの上書き**: Geminiモデルは環境変数 `GEMINI_FLASH_MODEL`, `GEMINI_3_8_FLASH_MODEL`, `GEMINI_3_5_FLASH_LITE_MODEL`, `GEMINI_PRO_MODEL` で任意のバージョンに上書き可能です。
> - **Web Search グラウンディング**: Gemini の Google Search ツールを利用するため、Gemini 系モデル (`flash` / `flash_3_8` / `flash_lite` / `pro`) のみ対応しています。
> - **意味ベクトル検索（Embedding）**: Layer 1 の条文検索補助には `text-embedding-004` (AI Studio) または `text-multilingual-embedding-002` (Vertex AI) を自動使用します。


---

## 🏛️ アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                       ユーザークエリ                        │
│         例: 「75歳の高齢者が歩道を自転車で走行。違反？」      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: 2008年式 決定論的処理（卒論ロジック）             │
│  ┌─────────────────────────┐   ┌─────────────────────────┐  │
│  │ Legal Compiler (AST)    │   │ VSM Engine (検索)       │  │
│  │ ・e-Gov XML → 条文AST   │   │ ・TF-IDF cos類似度      │  │
│  │ ・論理フラグ抽出        │──>│ ・意味埋め込み補助      │  │
│  │  （「を除く」「政令…」）│   │ ・条文アドレス特定      │  │
│  │ ・委任規定解決          │   │ ・反則金テーブル照合    │  │
│  └─────────────────────────┘   └─────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               │ 条文AST + 反則金 + 委任規定解決済み情報
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: 2026年式 LLM (Gemini / Claude / Gemma)            │
│  条文を「絶対的根拠」としてプロンプトに注入                 │
│  → 推論の脱走を物理的に封じ、正確な法的判定を出力           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 クイックスタート

### 前提条件

- **Python 3.11+**
- **Node.js 18+**（Web UI を動かす場合）
- いずれかの実行環境:
  - **Google AI Studio API Key**（推奨・最も手軽）
  - **Google Cloud プロジェクト**（Vertex AI / Claude 利用時）
  - **Ollama**（ローカルLLM利用時）

### インストール

```bash
# 1. リポジトリのクローン & 移動
git clone https://github.com/shu-kob/gemini-law-compiler.git
cd gemini-law-compiler

# 2. バックエンド（Python パッケージ）のインストール
pip install -e ./backend

# 3. フロントエンド（Next.js）の依存関係インストール（Web UI利用時）
cd frontend && npm install && cd ..
```

### 環境変数・認証設定

利用するプロバイダに合わせて `.env` を設定します。

#### パターン A: Google AI Studio（デフォルト・推奨）

プロジェクト直下に `.env` を作成し、Gemini API キーを設定します:

```bash
echo "GEMINI_API_KEY=your-gemini-api-key" > .env
```

#### パターン B: Google Cloud Vertex AI（Gemini / Claude 利用時）

```bash
cat << 'EOF' > .env
GEMINI_PROVIDER=vertex
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global
EOF

# gcloud 認証の実行
gcloud auth application-default login --scopes="https://www.googleapis.com/auth/cloud-platform"
gcloud auth application-default set-quota-project your-gcp-project-id
```

> **Claude を利用する場合:** Vertex AI Model Garden で Claude (Anthropic) への利用承諾を済ませておく必要があります。

#### パターン C: Ollama（ローカルモデル利用時）

API キーは不要です。Ollama を起動し、モデルを pull しておきます:

```bash
ollama run gemma3:4b
```

### 法令データの取得

e-Gov法令APIから対象の道路交通法XMLを取得します:

```bash
curl -o backend/data/road_traffic_act_full.xml \
  "https://laws.e-gov.go.jp/api/1/lawdata/昭和三十五年法律第百五号"
```

---

## 💻 使い方

### CLI（コマンドライン）

```bash
cd backend
```

#### 1. 2×2 マトリクス検証（おすすめ）
前処理（あり/なし）× 推論Thinking（あり/なし）の4パターンを一括比較します:

```bash
# 全テストケースで検証
python -m src.main --matrix

# 件数を絞って高速実行（例: 最初の2件）
python -m src.main --matrix --limit 2

# モデルを指定して実行（例: Gemini Pro）
python -m src.main --matrix --model pro --limit 2
```

#### 2. ハイブリッド判定（Layer 1 + Layer 2）
各LLMモデルに前処理情報を注入して法規判定を実行します:

```bash
# Gemini 3.7 Flash（デフォルト）
python -m src.main --hybrid

# Gemini 3.8 Flash
python -m src.main --hybrid --model flash_3_8

# Gemini 3.5 Flash-Lite
python -m src.main --hybrid --model flash_lite

# Gemini 3.1 Pro
python -m src.main --hybrid --model pro

# Claude Opus 4.7（要 Vertex AI）
python -m src.main --hybrid --model claude

# Claude Sonnet 4.6（要 Vertex AI）
python -m src.main --hybrid --model claude_sonnet

# Gemma 3 4B（要 Ollama）
python -m src.main --hybrid --model gemma3

# 純 TF-IDF (2008年式) のみで実行（Embedding 補助を無効化）
python -m src.main --hybrid --no-embedding
```

#### 3. ベンチマーク & 比較実行
```bash
# LLM単体ベンチマーク（ハルシネーションの観測）
python -m src.main --benchmark

# Flash単体 vs ハイブリッド比較（ブログ用レポート出力）
python -m src.main --compare
```

---

### Web UI（FastAPI + Next.js）

ブラウザからインタラクティブに違反シナリオを入力し、判定結果・根拠条文・反則金・思考プロセスを確認できます。

```bash
# ターミナル 1: バックエンド API を起動 (FastAPI: port 8000)
cd backend
python -m src.api.server

# ターミナル 2: フロントエンド Web UI を起動 (Next.js: port 3000)
cd frontend
npm run dev
```

ブラウザで `http://localhost:3000` にアクセスします。

- **モデル選択**: Gemini 3.7 Flash / Gemini 3.8 Flash / Gemini 3.5 Flash-Lite / Gemini 3.1 Pro / Gemma 3 / Claude Opus 4.7 / Claude Sonnet 4.6
- **グラウンディングモード**:
  - `llm_only`: LLM単体（グラウンディングなし）
  - `layer1`: Layer 1 決定論的グラウンディング（本プロジェクトの本命構成）
  - `web_search`: Google Search グラウンディング（Gemini系モデルのみ対応）


---

### Layer 1 単体実行（APIキー不要）

外部APIを一切呼び出さず、純粋な2008年卒論ロジックのみをローカルで動かすことができます:

```bash
cd backend

# e-Gov XML パーサー（条文AST抽出と論理フラグ解析）
python -m src.parser.legal_compiler

# VSM エンジン（TF-IDF コサイン類似度による条文検索）
python -m src.matcher.vsm_engine
```

---

## 📊 2×2 マトリクス検証（前処理 × 推論 Thinking）

「決定論的前処理（Layer 1）」と「LLM推論（Thinking Budget）」の有無を掛け合わせた4パターンを同一テストケースでベンチマーク計測します。

| パターン | 入力（前処理） | LLM推論（Thinking） | 特徴と検証結果 |
|---|---|---|---|
| **① 生テキスト × 推論なし** | 生の条文 | OFF (Budget=0) | **【最弱】** 参照ジャンプや多段ネストを追いきれず、読み落とし・ハルシネーション多発。 |
| **② 生テキスト × 推論あり** | 生の条文 | ON (Budget=2048) | **【力技】** 推論で参照を自力解決しようとするが、思考トークンが爆発し論理破綻のリスクあり。 |
| **③ 前処理済み × 推論あり** | 構造化・参照解決済み | ON (Budget=2048) | **【過剰推論】** 正解情報はすでにあるのに自問自答を挟み、レイテンシとコストが増大。 |
| **④ 前処理済み × 推論なし** | 構造化・参照解決済み | OFF (Budget=0) | **【本命】最速・最安・最高精度。** LLMを確定的評価器として用いることで最も安定。 |

実行結果はコンソール表示のほか、`backend/results/matrix_benchmark_<timestamp>.md` および `.json` に自動出力されます。

---

## 📁 プロジェクト構成

```
gemini-law-compiler/
├── .env                             # 環境変数（APIキー・GCP設定など）
├── .spec/spec.md                    # 基本仕様書
├── backend/                         # バックエンド（CLI & FastAPI）
│   ├── pyproject.toml
│   ├── data/
│   │   ├── road_traffic_act_full.xml    # e-Gov法令XML（gitignore）
│   │   └── bicycle_fine_table.json      # 青切符反則金マスターテーブル
│   ├── results/                         # ベンチマーク・検証結果レポート
│   ├── src/
│   │   ├── config.py                    # 共通設定・クライアント初期化
│   │   ├── main.py                      # CLI エントリポイント
│   │   ├── api/
│   │   │   ├── server.py                # FastAPI サーバー
│   │   │   └── judge_service.py         # 判定サービス（3モードディスパッチャ）
│   │   ├── parser/
│   │   │   └── legal_compiler.py        # e-Gov XML → 条文AST変換
│   │   ├── matcher/
│   │   │   ├── vsm_engine.py            # TF-IDF cos類似度エンジン
│   │   │   └── embedding_engine.py      # Vertex AI/AI Studio Embedding
│   │   ├── benchmark/
│   │   │   ├── flash_only_judge.py      # LLM 単体ベンチマーク
│   │   │   └── matrix_benchmark.py      # 2×2 マトリクス検証
│   │   ├── judgement/
│   │   │   └── hybrid_judge.py          # ハイブリッド判定エンジン
│   │   └── llm/                         # Ollama / Anthropic アダプタ
│   └── tests/                           # 単体テスト群
└── frontend/                        # Web UI（Next.js App Router, TypeScript）
    └── src/
        ├── app/page.tsx                 # メイン画面
        ├── components/JudgeForm.tsx     # 判定フォーム & 結果表示
        └── lib/api.ts                   # バックエンド API クライアント
```

---

## 🧪 テスト

純ロジック（XMLパーサ / VSM / プロンプト生成 / ハルシネーション検出 / 設定処理）は pytest による単体テストでカバーされています（外部LLM API呼び出し部分はテスト対象外のためオフラインで実行可能）。

```bash
cd backend
pip install -e '.[dev]'
python -m pytest tests/
```

テストケースの振る舞い契約および詳細仕様は [`docs/unit_test_spec.md`](docs/unit_test_spec.md) をご参照ください。

---

## 📜 対象法令 & ライセンス

- **対象法令**: **道路交通法**（昭和35年法律第105号）
- **改正法**: 令和6年法律第34号（令和6年5月24日公布）
- **施行日**: 2026年4月1日 — 自転車への交通反則通告制度（青切符）適用開始
- **対象範囲**: 16歳以上の自転車運転者 / 113種類の違反 / 反則金3,000〜12,000円
- **ライセンス**: [MIT License](LICENSE)

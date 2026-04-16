# Velo-Verify-Gemini — Hybrid Logic Compiler

『自転車の青切符をGeminiで判定しようとしたら、2008年度に書いた卒論に救われた話』検証用プロトタイプ

## 概要

2026年4月1日施行の**自転車交通反則通告制度（青切符）**（令和6年法律第34号）を題材に、
最新LLM（Gemini Flash/Pro）が法規判定で陥る「もっともらしい嘘（ハルシネーション）」を、
2008年当時の卒論技術（決定論的パース + コサイン類似度）で検出・矯正するハイブリッドシステム。

## アーキテクチャ

```
┌─────────────────────────────────────────────────┐
│                 ユーザークエリ                      │
│  例: 「75歳の高齢者が歩道を自転車で走行。違反？」    │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  Layer 1: 2008年式 決定論的処理（卒論ロジック）      │
│  ┌───────────────────┐  ┌─────────────────────┐  │
│  │ Legal Compiler     │  │ VSM Engine          │  │
│  │ e-Gov XML → AST    │  │ TF-IDF cos類似度     │  │
│  │ 論理フラグ抽出      │→│ 条文アドレス特定      │  │
│  │ 「を除く」「政令…」 │  │ 反則金テーブル照合    │  │
│  └───────────────────┘  └─────────────────────┘  │
└─────────────┬───────────────────────────────────┘
              │ 条文AST + 反則金 + 委任規定解決済み情報
              ▼
┌─────────────────────────────────────────────────┐
│  Layer 2: 2026年式 Gemini Flash/Pro               │
│  条文を「絶対的根拠」としてプロンプトに注入         │
│  → 推論の脱走を物理的に封じて判定                  │
└─────────────────────────────────────────────────┘
```

## Flashが陥る3つの失敗パターン

| パターン | 内容 | 例 |
|---------|------|-----|
| **階層無視** | 原則→例外→例外の例外を平坦化 | 歩道通行の例外規定を見落とす |
| **数値捏造** | 条文にない反則金額を補完 | 存在しない「3,000円」を回答 |
| **参照欠落** | 「政令で定める者」を解決できない | 70歳以上の歩道通行許可を無視 |

## セットアップ

### 前提条件

- Python 3.11+
- Google Cloud プロジェクト（Vertex AI API有効）
- gcloud CLI（ADC認証済み）

### インストール

```bash
pip install -e .
```

### 認証設定

Vertex AI (ADC) 経由で接続します。Gemini 3系モデルは `location=global` で動作します。

```bash
gcloud auth application-default login --scopes="https://www.googleapis.com/auth/cloud-platform"
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

> **注意**: `GOOGLE_APPLICATION_CREDENTIALS` 環境変数にサービスアカウントキーが設定されていると、ADCよりも優先されて権限エラーになる場合があります。本ツールは起動時にこの環境変数を自動的に除外してADCを使用します。

### 法令データ取得

e-Gov法令APIから道路交通法XMLを取得:

```bash
curl -o data/road_traffic_act_full.xml \
  "https://laws.e-gov.go.jp/api/1/lawdata/昭和三十五年法律第百五号"
```

## 使い方

```bash
# Flash単体ベンチマーク（ハルシネーション観測）
python -m src.main --benchmark

# ハイブリッド判定
python -m src.main --hybrid

# Flash単体 vs ハイブリッド比較（ブログ用）
python -m src.main --compare

# Proモデル使用
python -m src.main --hybrid --model pro
```

### Layer 1 のみ（API不要）

```bash
# e-Gov XMLパーサー
python -m src.parser.legal_compiler

# cos類似度検索
python -m src.matcher.vsm_engine
```

## プロジェクト構成

```
gemini-law-compiler/
├── .spec/spec.md                    # 仕様書
├── data/
│   ├── road_traffic_act_full.xml    # e-Gov法令XML（gitignore）
│   └── bicycle_fine_table.json      # 青切符反則金テーブル
├── src/
│   ├── config.py                    # 共通設定・Vertex AI接続
│   ├── main.py                      # エントリポイント
│   ├── parser/
│   │   └── legal_compiler.py        # e-Gov XML → 条文AST変換
│   ├── matcher/
│   │   └── vsm_engine.py            # TF-IDF cos類似度エンジン
│   ├── benchmark/
│   │   └── flash_only_judge.py      # Flash単体ベンチマーク
│   └── judgement/
│       └── hybrid_judge.py          # ハイブリッド判定エンジン
└── results/                         # 実行結果出力先（gitignore）
```

## 対象法令

- **道路交通法**（昭和35年法律第105号）
- **改正法**: 令和6年法律第34号（令和6年5月24日公布）
- **施行日**: 2026年4月1日 — 自転車への交通反則通告制度（青切符）適用開始
- 対象: 16歳以上の自転車運転者 / 113種類の違反 / 反則金3,000〜12,000円

## ライセンス

MIT

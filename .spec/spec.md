Spec: Velo-Verify-Gemini (Hybrid Logic Compiler)
Project Title: 『自転車の青切符をGeminiで判定しようとしたら、2008年度に書いた卒論に救われた話』検証用プロトタイプ

1. Project Objective
e-Govの「道路交通法」をソースとし、最新LLM（Gemini-3-Flash）が陥る「論理の脱線（ハルシネーション）」を、2008年当時の卒論技術（決定論的NLP）で検出し、矯正するハイブリッド・アーキテクチャの構築。

1.1 対象法令
- 道路交通法（昭和35年法律第105号）
- 改正法: 令和6年法律第34号（令和6年5月24日公布）
- 施行日: 2026年4月1日 — 自転車への交通反則通告制度（青切符）の適用開始
- データソース: e-Gov法令API (https://laws.e-gov.go.jp/api/1/lawdata/昭和三十五年法律第百五号)
- ローカルキャッシュ: data/road_traffic_act_full.xml（e-Gov APIから取得した完全版XML、約1.5MB）

1.2 2026年4月1日施行「自転車青切符制度」の概要
従来、自転車の交通違反は「指導警告」か「赤切符（刑事罰）」の二択だったが、
2026年4月1日より「青切符（交通反則通告制度）」が導入された。

- 対象: 16歳以上の自転車運転者
- 対象違反: 113種類
- 反則金額: 3,000円〜12,000円
- 反則金を納付すれば刑事手続きに移行せず、前科もつかない
- 酒気帯び運転・酒酔い運転・妨害運転等は引き続き赤切符（刑事罰）

主要な反則金額:
| 金額 | 主な違反行為 |
|------|------------|
| 12,000円 | 携帯電話使用等（ながらスマホ） |
| 7,000円 | 遮断踏切立入り |
| 6,000円 | 信号無視、通行区分違反（逆走等）、安全運転義務違反、横断歩行者等妨害等 |
| 5,000円 | 一時不停止、無灯火、ブレーキ不良、公安委員会遵守事項違反（傘差し等） |
| 3,000円 | 並進禁止違反、歩道徐行等義務違反、二人乗り |

2. Failure Analysis Strategy (The "Hook" for Blog)
Gemini-3-Flash単体に法規を読み込ませた際、以下のバグが発生することを「仕様」として定義し、ベンチマークを行う。

階層無視: 条文のネスト（原則→例外→例外の例外）を平坦化して読み、誤った判定を下す。

数値捏造: 条文内にない反則金額を、Web上の不正確な知識から補完して回答する。

参照欠落: 「政令で定める者」等のポインタを解決できず、特定の属性（高齢者等）の免除規定を無視する。

3. Core Architecture
Layer 1 (The 2008 Saviors): * MeCab Parser: 条・項・号を正規表現で固定し、XMLを厳密なJSONツリー（AST）に変換。

VSM Matcher: 2008年式cos類似度を用い、クエリ（違反状況）に該当する「絶対的な条文アドレス」を特定。

Layer 2 (The 2026 Reasoning):

Gemini Controller: Layer 1が特定した「一箇所の条文」のみをコンテキストに与え、推論の脱走を物理的に封じる。

4. Claude Code Implementation Tasks
Task 1: Failure Reproduction (Benchmark Script)
src/benchmark/flash_only_judge.py

素のGemini-3-Flashに道交法を丸投げし、「75歳の老人が歩道を走行した」等のエッジケースを判定させる。

期待される「誤答」をログに記録し、ブログ用の比較データとする。

Task 2: 2008-Style Deterministic Preprocessor
src/parser/legal_compiler.py

e-Gov XMLから、条文のインデント（Hierarchy）を完全にパースするコード。

ポイント: 「〜を除く」「〜を準用する」というトークンを、論理フラグとして抽出。

Task 3: Saving with Cosine Similarity
src/matcher/vsm_engine.py

事務局長の卒論ロジックの実装。

入力事例をベクトル化し、全条文の中で最もcos類似度が高い条文を特定。

出力: 「Geminiは第60条と迷っていますが、cos類似度は第63条の4が正解だと言っていますｗ」というメタデータを出力させる。

Task 4: Hybrid Corrected Reasoner (PTE Standard)
src/judgement/hybrid_judge.py

FlashとProのパラメータ切り替えに対応。

Layer 1のパース結果をプロンプトの「絶対的根拠」としてインジェクションし、Flashに再判定させる。

5. Blog Narrative Support (Logging Specs)
プログラム実行時の標準出力に、ブログにそのままコピペできる「魂のログ」を出す。

[2026-AI-Logic]: 歩道走行は違反、反則金3,000円（※捏造）です。
[2008-Thesis-Logic]: 待て。MeCabパース結果、第63条の4第1項第2号に『高齢者免除』を検知。
[Hybrid-Result]: 判定を修正。本件は『合法』です。18年前のロジックがAIの過ちを正しました。

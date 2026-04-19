# Unit Test 仕様書

`tests/` 配下の Unit Test が固定化している「各モジュールの振る舞い契約（コントラクト）」を記述する。TDD の回帰テストとして、実装を変更した際は本仕様書とテストコードが同時に更新されることを期待する。

## 対象範囲 / 対象外

### 対象（純ロジック）

| モジュール | テストファイル | 説明 |
|---|---|---|
| `src/parser/legal_compiler.py` | `tests/test_legal_compiler.py` | e-Gov XML → AST 変換と論理フラグ抽出 |
| `src/matcher/vsm_engine.py` | `tests/test_vsm_engine.py` | TF-IDF cos類似度エンジン |
| `src/benchmark/flash_only_judge.py` | `tests/test_flash_only_judge.py` | 正答判定とハルシネーション検出（純関数のみ） |
| `src/judgement/hybrid_judge.py` | `tests/test_hybrid_judge.py` | プロンプト生成と反則金ルックアップ（API非依存部分） |

### 対象外（Gemini API 呼び出し部）

`run_flash_benchmark`（`flash_only_judge`）と `HybridJudge.judge`（`hybrid_judge`）は Vertex AI / Gemini API を叩くため、Unit Test の対象外。これらは `--benchmark` / `--hybrid` / `--compare` の End-to-End 実行によって検証される。

## 実行方法

```bash
pip install -e '.[dev]'         # pytest を含む dev 依存をインストール
python -m pytest tests/         # 全テスト実行（現在 101 件）
python -m pytest tests/ -v      # 詳細表示
```

## フィクスチャ

- `tests/fixtures/sample_law.xml` — e-Gov API 応答を模した最小 XML。以下を含む:
  - 自転車関連条文: 第2条（定義）、第7条（信号）、第63条の4（歩道通行）、第65条（酒気帯び）、第70条（安全運転義務）
  - 非自転車条文: 第999条（抽出フィルタの negative test 用）
  - `ArticleCaption` / `ArticleTitle` / `Paragraph` / `ParagraphSentence`（`Function="main"`/`"proviso"`）/ `Item` / `ItemTitle` / `ItemSentence` / `SupplNote` / `Ruby`+`Rt` の各タグ
  - 論理トークン: `にかかわらず` / `することができる` / `ただし` / `してはならない` / `政令で定める` / `を除く` / `しなければならない`

- `tests/conftest.py` — `sample_ast`, `sample_bicycle_articles` の共通フィクスチャを提供。

---

## モジュール別仕様

### 1. `legal_compiler.py`

#### 1.1 `_detect_logic_flags(text) -> list[str]`

論理トークン正規表現 `LOGIC_PATTERNS` に対する literal マッチ。以下 10 種のフラグのどれにマッチしたかを返す。

| flag | 正規表現 | 想定用途 |
|---|---|---|
| `prohibition` | `してはならない` | 禁止規定 |
| `obligation` | `しなければならない` | 義務規定 |
| `permission` | `することができる` | 許容規定 |
| `proviso` | `ただし[、,]` | ただし書き |
| `exception` | `を除[くき]` | 除外規定 |
| `delegation` | `政令で定める` | 委任規定 |
| `application_mutatis` | `を準用する` | 準用 |
| `notwithstanding` | `にかかわらず` | 原則の例外 |
| `effort_obligation` | `よう努めなければならない` | 努力義務 |
| `penalty_ref` | `罰則` | 罰則条項参照 |

**固定化した挙動:**
- パターンは literal（活用違いでマッチしない）。「従**わ**なければならない」は `obligation` にマッチしない。これは道交法本文が文末で「〜しなければならない」に収束する傾向に基づく実用上の割り切り。
- `effort_obligation` と `obligation` は互いに排他（"よう努めなければならない" は後者にマッチしない）。
- 空文字列に対しては `[]` を返す。

#### 1.2 `_extract_text(elem) -> str`

XML 要素から再帰的にテキストを抽出。`<Rt>` 子要素（ルビ読み）は除外し、ベース漢字と `tail` テキストを連結する。

#### 1.3 `parse_egov_xml(xml_path) -> LawAST`

- e-Gov XML を読み `LawBody` 配下の全 `Article` をフラットに収集する（`Chapter` / `Section` を跨いで `iter("Article")`）。
- `LawBody` が見つからない場合 `ValueError("LawBody element not found in XML")`。
- 各 `Article` の `Paragraph` → `Item` → `Subitem1/2/3` の階層を `ArticleNode` / `ParagraphNode` / `ItemNode` に保持する。
- 各 `Sentence` は `Function` 属性（`"main"` / `"proviso"` / `""`）と検出済み `logic_flags` を持つ。

#### 1.4 `LawAST.find_article(num)`

指定の `Num` 属性と一致する最初の条文を返す。見つからなければ `None`。

#### 1.5 `LawAST.to_dict` / `to_json`

`dataclasses.asdict` ベースのシリアライズ。`to_json` は `ensure_ascii=False` で UTF-8 の日本語を保持。

#### 1.6 `extract_bicycle_articles(ast) -> list[ArticleNode]`

ハードコードされた「自転車関連条文番号セット」（第2条 / 第7条 / 第17条 / 第63条の3〜11 / 第65条 / 第70条 / 第71条 ほか）に一致する条文のみを返す。

#### 1.7 `flatten_article_text(article) -> str`

`ArticleNode` を平文テキストに変換。構成: `「{title} {caption}」\n{全 Paragraph 本文}\n{Item title + 本文}\n  {Subitem title + 本文}`。VSM ベクトル化用の一次ソース。

---

### 2. `vsm_engine.py`

#### 2.1 `tokenize(text) -> list[str]`

- 句読点・括弧（`[、。,.\s（）()「」『』\[\]]`）でチャンク分割。
- 各チャンク内で文字 bi-gram を生成。
- `_PARTICLE_RE` にフルマッチする bi-gram は除外。**ただし `_PARTICLE_RE` は単一文字クラスのため `fullmatch` は 2 文字 bi-gram に反応しない**。つまり現行実装では助詞×助詞の bi-gram はそのまま残る（既知の挙動、テストで固定化済）。
- 法規特有キーワード 35 個（`自転車` / `歩道` / `携帯電話` 等）がテキストに含まれていれば、別途トークンとして追加。

#### 2.2 `VSMEngine`

**インデックス構築** (`_build_index`)
- 対象文書 = コンストラクタで渡した `article_filter`（省略時は AST 全条文）。
- DF: 各トークンを含む文書数。
- IDF: `log(N / df) + 1`（smoothed）。値は常に `>= 1.0`。
- 各文書の TF-IDF: `(count / total) * IDF`。

**検索** (`search(query, top_k)`)
- 空クエリ / 記号のみクエリ → 空 list。
- 返却は `VSMMatch`（`article`, `score`, `rank`）の list。
- `score` は float、`rank` は 1 から始まる昇順。
- スコアは降順に並ぶ（monotonically non-increasing）。
- `score ∈ [0.0, 1.0]`。

**cos類似度** (`_cosine_similarity(a, b)`)
- 同一ベクトル → 1.0。
- 直交（共通キーなし）→ 0.0。
- 空 dict / 全要素 0 → 0.0。
- 定数倍ベクトルは同一視（スケール不変）。

**ランキング期待値（フィクスチャ）**
- 「普通自転車で歩道を走行」→ 第63条の4 が top 1。
- 「酒気帯び運転」→ 第65条 が top 3 以内。

---

### 3. `flash_only_judge.py`

#### 3.1 `_check_answer(answer, tc) -> bool`

回答文が以下の両方を満たすと `True`:
1. `tc.expected_article` が answer に含まれる（または `第`除去版 / `条の→-` 代替表記を含む）。
2. 判定一致: 期待=合法なら answer に「違反」のみはNG、期待=違反なら「合法」のみはNG。両方含む場合は通過。

#### 3.2 `_detect_hallucination(answer, tc) -> str`

`tc.failure_type` ごとに異なる検出ロジック。検出された issue を `"; "` で連結して返す（なしなら `""`）。

| failure_type | 検出条件 |
|---|---|
| `number_fabrication` | answer 内の `数字円` 表記を抽出し、反則金テーブルに存在しない額を検出。ただし `500,000`・`1,000,000` は刑事罰金の許可リスト。 |
| `reference_missing` | 「政令 / 施行令 / 70歳 / 高齢者 / 児童」のいずれにも言及なしで `参照欠落` 検出。 |
| `hierarchy_ignore` | 「ただし / 例外 / やむを得ない」のいずれにも言及なしで `階層無視` 検出。 |

#### 3.3 `TEST_CASES`

- 全 case が必須フィールド (`id`, `scenario`, `expected_answer`, `expected_article`, `failure_type`, `description`) を埋める。
- `id` は重複なし。
- `failure_type` は 3 種全てをカバー（`hierarchy_ignore`, `number_fabrication`, `reference_missing`）。

#### 3.4 `print_summary(results)`

`stdout` に `正答率 {correct}/{total} ({pct}%)` と `ハルシネーション検知 {n}/{total}` を出力。

---

### 4. `hybrid_judge.py`

#### 4.1 `HYBRID_SYSTEM_PROMPT`

- 「青切符」「2026」「補完または学習」の語を含む（LLM に学習データからの推測を禁じるため）。

#### 4.2 `HybridJudge._build_prompt(query, article_text, fine_info, logic_flags) -> str`

- `query` / `article_text` / `fine_info` を全て文字列埋め込み。
- `logic_flags` が空なら `なし`、複数なら `", "` で連結。
- `"delegation"` が flags に含まれる場合のみ、`【委任規定の解決済み情報】` ブロック（児童・70歳以上・障害者の歩道通行許可）を注入。
- 回答スキーマ（`judgement` / `article` / `fine` / `reasoning`）を JSON 形式で明示。

#### 4.3 `HybridJudge._lookup_fine(query, article) -> str`

1. `query` に対するキーワードマッチ（`歩道` → `[通行区分違反, 歩道徐行等義務違反]` など 16 エントリ）で違反名の候補を収集。
2. 候補名に一致する `fines` と `criminal_only` を検索。`criminal_only` は `反則金対象外・刑事罰` 表記で出力。
3. 1-2 でヒットなしの場合、`article.num` を `fine.article` に部分一致させて逆引き。
4. それでもヒットなしなら `"該当する反則金情報なし（条文を確認して判断してください）"`。
5. 複数ヒット時は `"\n"` 連結。

期待ケース:
| 入力クエリ | 条文 | 期待結果 |
|---|---|---|
| `スマホを持って走行` | 任意 | `携帯電話` + `12,000` |
| `信号無視した` | 任意 | `信号無視` + `6,000` |
| `酒気帯びで運転` | 任意 | `反則金対象外` + `刑事罰` |
| `二人乗りをした` | 任意 | `軽車両乗車積載制限違反` + `3,000` |
| `踏切を通過した` | 任意 | `遮断踏切立入り` + `踏切不停止等` を改行連結 |
| `説明不能な状況` | 第7条 | 条文番号逆引きで `信号無視` + `6,000` |
| `完全に無関係な内容` | 第9999条 | `該当する反則金情報なし` |

#### 4.4 `HybridJudge._load_fine_table() -> dict`

- `data/bicycle_fine_table.json` を UTF-8 で読み込み、dict を返す。
- トップレベルキーに `metadata` / `fines` / `criminal_only` を含む。
- `fines[*]` は `amount`（int）/ `violation`（str）/ `article`（str）を必ず持つ。

---

## テスト結果サマリ

| テストファイル | ケース数 |
|---|---|
| `tests/test_legal_compiler.py` | 39 |
| `tests/test_vsm_engine.py` | 24 |
| `tests/test_flash_only_judge.py` | 17 |
| `tests/test_hybrid_judge.py` | 21 |
| **合計** | **101** |

最後の実行時の `pytest` 出力:

```
============================= 101 passed in 0.07s ==============================
```

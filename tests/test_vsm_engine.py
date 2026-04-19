"""Unit tests for src/matcher/vsm_engine.py

Layer 1 の TF-IDF cos類似度エンジンの振る舞いを固定化する。
外部ライブラリ不使用の純Python実装であるため、決定論的に検証可能。
"""

from __future__ import annotations

import math

import pytest

from src.matcher.vsm_engine import VSMEngine, VSMMatch, tokenize
from src.parser.legal_compiler import LawAST


# ---------------------------------------------------------------------------
# tokenize: bi-gram + 法規キーワード
# ---------------------------------------------------------------------------
class TestTokenize:
    def test_returns_list_of_strings(self) -> None:
        tokens = tokenize("自転車で歩道を走行する")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_empty_string_returns_empty_list(self) -> None:
        assert tokenize("") == []

    def test_short_chunk_is_filtered(self) -> None:
        # 1文字以下のチャンクは除外される
        tokens = tokenize("a")
        # キーワードにもbi-gramにも該当しない
        assert tokens == []

    def test_legal_keyword_is_added(self) -> None:
        # 「自転車」は法規キーワードリストに含まれるため、トークンに現れる
        tokens = tokenize("自転車で走行した")
        assert "自転車" in tokens

    def test_multiple_keywords(self) -> None:
        tokens = tokenize("高齢者が自転車で歩道を走行")
        assert "自転車" in tokens
        assert "歩道" in tokens
        assert "高齢者" in tokens

    def test_bigrams_are_generated(self) -> None:
        tokens = tokenize("徐行")
        # 文字 bi-gram 「徐行」が含まれる
        assert "徐行" in tokens

    def test_particle_filter_only_matches_single_char_bigrams(self) -> None:
        # 現行実装: _PARTICLE_RE はシングル文字クラスのため、
        # `fullmatch` は 2 文字の bi-gram には反応しない。
        # つまり「のを」「にで」のような助詞×助詞の bi-gram はそのまま残る。
        # これは実装の既知の性質であり、本テストで挙動を固定化する。
        tokens = tokenize("のを")
        assert "のを" in tokens

    def test_punctuation_splits_chunks(self) -> None:
        # 句読点・括弧でチャンクが分割され、チャンクを跨いだ bi-gram は生成されない
        tokens = tokenize("走行、運転")
        # 「行、」や「、運」のようなbi-gramは含まれない
        assert "行、" not in tokens
        assert "、運" not in tokens


# ---------------------------------------------------------------------------
# VSMEngine._cosine_similarity
# ---------------------------------------------------------------------------
class TestCosineSimilarity:
    def test_identical_vectors_return_one(self) -> None:
        v = {"a": 1.0, "b": 2.0}
        assert VSMEngine._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self) -> None:
        a = {"a": 1.0}
        b = {"b": 1.0}
        assert VSMEngine._cosine_similarity(a, b) == 0.0

    def test_empty_vector_returns_zero(self) -> None:
        assert VSMEngine._cosine_similarity({}, {"a": 1.0}) == 0.0
        assert VSMEngine._cosine_similarity({"a": 1.0}, {}) == 0.0

    def test_zero_norm_returns_zero(self) -> None:
        # 全要素が0の場合 norm が 0 → 0.0 返却
        assert VSMEngine._cosine_similarity({"a": 0.0}, {"a": 0.0}) == 0.0

    def test_partial_overlap_is_between_zero_and_one(self) -> None:
        a = {"x": 1.0, "y": 1.0}
        b = {"x": 1.0, "z": 1.0}
        sim = VSMEngine._cosine_similarity(a, b)
        assert 0.0 < sim < 1.0
        # 理論値: 1 / (sqrt(2) * sqrt(2)) = 0.5
        assert sim == pytest.approx(0.5)

    def test_scaled_vector_same_similarity(self) -> None:
        a = {"x": 1.0, "y": 2.0}
        b = {"x": 2.0, "y": 4.0}  # aの定数倍
        assert VSMEngine._cosine_similarity(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# VSMEngine: 検索ランキング
# ---------------------------------------------------------------------------
class TestVSMEngineSearch:
    @pytest.fixture(scope="class")
    def engine(self, sample_ast: LawAST, sample_bicycle_articles) -> VSMEngine:
        return VSMEngine(sample_ast, article_filter=sample_bicycle_articles)

    def test_returns_vsm_matches(self, engine: VSMEngine) -> None:
        results = engine.search("自転車で歩道を走行", top_k=3)
        assert all(isinstance(m, VSMMatch) for m in results)

    def test_top_k_respected(self, engine: VSMEngine) -> None:
        results = engine.search("自転車", top_k=2)
        assert len(results) <= 2

    def test_empty_query_returns_empty(self, engine: VSMEngine) -> None:
        assert engine.search("", top_k=3) == []
        # 記号のみも、tokenize後に空になるため空結果
        assert engine.search(",,,", top_k=3) == []

    def test_ranks_are_sequential(self, engine: VSMEngine) -> None:
        results = engine.search("自転車 歩道", top_k=3)
        ranks = [m.rank for m in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_scores_are_monotonically_nonincreasing(self, engine: VSMEngine) -> None:
        results = engine.search("自転車 歩道", top_k=5)
        scores = [m.score for m in results]
        for i in range(1, len(scores)):
            assert scores[i - 1] >= scores[i]

    def test_scores_are_bounded(self, engine: VSMEngine) -> None:
        results = engine.search("歩道を自転車で走行", top_k=3)
        for m in results:
            assert 0.0 <= m.score <= 1.0

    def test_sidewalk_query_matches_63_4(self, engine: VSMEngine) -> None:
        # 「歩道」「普通自転車」というキーワードから、
        # 第六十三条の四（普通自転車の歩道通行）が最上位にランクされるはず。
        results = engine.search("普通自転車で歩道を走行した", top_k=3)
        assert results, "検索結果が空"
        assert results[0].article.num == "63_4"

    def test_drinking_query_matches_65(self, engine: VSMEngine) -> None:
        # 「酒気」で第六十五条（酒気帯び運転等の禁止）がヒット
        results = engine.search("自転車で酒気帯び運転をした", top_k=3)
        assert results
        top_nums = [m.article.num for m in results]
        assert "65" in top_nums

    def test_empty_article_filter_produces_empty_engine(self, sample_ast: LawAST) -> None:
        engine = VSMEngine(sample_ast, article_filter=[])
        # インデックスが空の場合でも例外を出さず、空結果を返す
        assert engine.search("自転車", top_k=3) == []

    def test_idf_smoothing_is_log_n_over_df_plus_one(
        self, sample_ast: LawAST, sample_bicycle_articles
    ) -> None:
        # IDF = log(N / df) + 1 が仕様。
        engine = VSMEngine(sample_ast, article_filter=sample_bicycle_articles)
        n = len(sample_bicycle_articles)
        for token, idf in engine._idf.items():
            # 最低でも log(N/N) + 1 = 1.0 以上
            assert idf >= 1.0
            # 最大値も妥当な範囲
            assert idf <= math.log(n) + 1.0 + 1e-9

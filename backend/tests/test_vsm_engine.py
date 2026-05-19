"""Unit tests for src/matcher/vsm_engine.py

Layer 1 の TF-IDF cos類似度エンジンの振る舞いを固定化する。
外部ライブラリ不使用の純Python実装であるため、決定論的に検証可能。
"""

from __future__ import annotations

import math

import pytest

from src.matcher.vsm_engine import (
    VSMEngine,
    VSMMatch,
    _hybrid_combine,
    _minmax_normalize,
    tokenize,
)
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


# ---------------------------------------------------------------------------
# Hybrid score helpers: _minmax_normalize / _hybrid_combine
# ---------------------------------------------------------------------------
class TestMinMaxNormalize:
    def test_empty_list_returns_empty(self) -> None:
        assert _minmax_normalize([]) == []

    def test_all_same_values_returns_zeros(self) -> None:
        # min == max のとき span=0 で zero ベクトルにフォールバック
        assert _minmax_normalize([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]

    def test_scales_to_unit_interval(self) -> None:
        out = _minmax_normalize([1.0, 2.0, 3.0])
        assert out == [0.0, 0.5, 1.0]

    def test_preserves_relative_order(self) -> None:
        values = [0.14, 0.27, 0.18, 0.09]
        out = _minmax_normalize(values)
        # 順序は不変
        original_rank = sorted(range(len(values)), key=lambda i: values[i])
        normalized_rank = sorted(range(len(out)), key=lambda i: out[i])
        assert original_rank == normalized_rank


class TestHybridCombine:
    def test_alpha_one_keeps_tfidf_ranking(self) -> None:
        # α=1.0 では embedding スコアは無視され、TF-IDF と同順位
        tfidf = [0.3, 0.1, 0.2]
        emb = [0.9, 0.8, 0.7]
        combined = _hybrid_combine(tfidf, emb, alpha=1.0)
        # TF-IDF の rank: 0 > 2 > 1
        assert combined[0] > combined[2] > combined[1]

    def test_alpha_zero_keeps_embedding_ranking(self) -> None:
        tfidf = [0.3, 0.1, 0.2]
        emb = [0.5, 0.9, 0.7]
        combined = _hybrid_combine(tfidf, emb, alpha=0.0)
        # embedding の rank: 1 > 2 > 0
        assert combined[1] > combined[2] > combined[0]

    def test_normalization_protects_against_scale_difference(self) -> None:
        # TF-IDF は 0.1〜0.3、embedding は 0.6〜0.8 で全体スケールが違う。
        # 単純加重和なら常に embedding が支配するが、min-max 正規化により
        # 「相対的に低スコアな embedding」<「相対的に高スコアな TF-IDF」を実現可能。
        tfidf = [0.3, 0.1]  # idx0 が圧倒的に強い (TF-IDF 視点)
        emb = [0.6, 0.7]   # idx1 がやや強い (embedding 視点)
        # α=0.7 なら TF-IDF 寄り → idx0 勝ち
        combined = _hybrid_combine(tfidf, emb, alpha=0.7)
        assert combined[0] > combined[1]

    def test_mismatched_lengths_fall_back_to_tfidf(self) -> None:
        # embedding 側が落ちている異常系では TF-IDF にフォールバック
        tfidf = [0.3, 0.1, 0.2]
        emb = [0.5]
        combined = _hybrid_combine(tfidf, emb, alpha=0.5)
        assert combined == tfidf


# ---------------------------------------------------------------------------
# VSMEngine: ハイブリッドモード (embedding_engine 注入)
# ---------------------------------------------------------------------------
class _StubEmbeddingEngine:
    """テスト用の固定 cos 類似度を返すスタブ。"""

    def __init__(self, scores_per_query: dict[str, list[float]]):
        self._scores = scores_per_query

    def cosine_similarities(self, query: str) -> list[float]:
        return self._scores.get(query, [])


class TestHybridSearch:
    def test_is_hybrid_flag_reflects_embedding_engine(
        self, sample_ast: LawAST, sample_bicycle_articles
    ) -> None:
        vsm_tfidf = VSMEngine(sample_ast, article_filter=sample_bicycle_articles)
        assert vsm_tfidf.is_hybrid is False

        n = len(sample_bicycle_articles)
        stub = _StubEmbeddingEngine({"q": [0.0] * n})
        vsm_hybrid = VSMEngine(
            sample_ast,
            article_filter=sample_bicycle_articles,
            embedding_engine=stub,
            alpha=0.5,
        )
        assert vsm_hybrid.is_hybrid is True

    def test_alpha_one_disables_hybrid(
        self, sample_ast: LawAST, sample_bicycle_articles
    ) -> None:
        # alpha=1.0 では embedding は使われず、is_hybrid は False
        n = len(sample_bicycle_articles)
        stub = _StubEmbeddingEngine({"q": [0.0] * n})
        vsm = VSMEngine(
            sample_ast,
            article_filter=sample_bicycle_articles,
            embedding_engine=stub,
            alpha=1.0,
        )
        assert vsm.is_hybrid is False

    def test_embedding_boost_changes_ranking(
        self, sample_ast: LawAST, sample_bicycle_articles
    ) -> None:
        # TF-IDF では 1 位にならないが embedding では 1 位になる条文を、
        # ハイブリッド合成で top に押し上げられることを確認する。
        query = "ながらスマホ運転"

        vsm_tfidf = VSMEngine(sample_ast, article_filter=sample_bicycle_articles)
        tfidf_top = vsm_tfidf.search(query, top_k=1)
        assert tfidf_top

        # スタブ embedding は TF-IDF top と別の条文に最高スコアを付ける
        target_idx = (0 if sample_bicycle_articles[0].num != tfidf_top[0].article.num
                      else 1)
        n = len(sample_bicycle_articles)
        emb_scores = [0.5] * n
        emb_scores[target_idx] = 0.99  # 強い意味マッチをシミュレート

        stub = _StubEmbeddingEngine({query: emb_scores})
        vsm_hybrid = VSMEngine(
            sample_ast,
            article_filter=sample_bicycle_articles,
            embedding_engine=stub,
            alpha=0.0,  # 完全に embedding 主導
        )
        hybrid_top = vsm_hybrid.search(query, top_k=1)
        assert hybrid_top[0].article.num == sample_bicycle_articles[target_idx].num

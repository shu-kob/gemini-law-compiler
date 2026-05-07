"""Unit tests for src/judgement/hybrid_judge.py

Gemini API を叩かない純ロジック（_build_prompt / _lookup_fine / _load_fine_table）
のみ検証する。HybridJudge のインスタンスは `__new__` で生成し、
必要な属性のみを手動でセットする（ネットワーク不要）。
"""

from __future__ import annotations

import json

import pytest

from src.judgement.hybrid_judge import HYBRID_SYSTEM_PROMPT, HybridJudge
from src.parser.legal_compiler import ArticleNode


@pytest.fixture
def judge_no_api() -> HybridJudge:
    """API クライアントを持たない純ロジック検証用の HybridJudge。"""
    instance = HybridJudge.__new__(HybridJudge)
    instance._fine_table = HybridJudge._load_fine_table()
    return instance


@pytest.fixture
def dummy_article() -> ArticleNode:
    return ArticleNode(num="7", caption="（信号）", title="第七条")


# ---------------------------------------------------------------------------
# HYBRID_SYSTEM_PROMPT: システムプロンプトの内容
# ---------------------------------------------------------------------------
class TestSystemPrompt:
    def test_mentions_blue_ticket_system(self) -> None:
        assert "青切符" in HYBRID_SYSTEM_PROMPT

    def test_forbids_hallucination(self) -> None:
        # 学習データからの補完を禁じる文言が含まれているべき
        assert "補完" in HYBRID_SYSTEM_PROMPT or "学習" in HYBRID_SYSTEM_PROMPT

    def test_effective_date_mentioned(self) -> None:
        assert "2026" in HYBRID_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# _build_prompt: Layer 1 の結果をプロンプトに注入
# ---------------------------------------------------------------------------
class TestBuildPrompt:
    def test_query_is_embedded(self, judge_no_api: HybridJudge) -> None:
        query = "75歳が歩道を自転車で走行"
        prompt = judge_no_api._build_prompt(query, "条文", "反則金情報", [])
        assert query in prompt

    def test_article_text_is_embedded(self, judge_no_api: HybridJudge) -> None:
        article_text = "第六十三条の四 ... 歩道を通行することができる"
        prompt = judge_no_api._build_prompt("Q", article_text, "F", [])
        assert article_text in prompt

    def test_fine_info_is_embedded(self, judge_no_api: HybridJudge) -> None:
        fine_info = "・信号無視: 反則金6,000円"
        prompt = judge_no_api._build_prompt("Q", "A", fine_info, [])
        assert fine_info in prompt

    def test_empty_flags_shown_as_none(self, judge_no_api: HybridJudge) -> None:
        prompt = judge_no_api._build_prompt("Q", "A", "F", [])
        assert "なし" in prompt

    def test_flags_joined_with_comma(self, judge_no_api: HybridJudge) -> None:
        prompt = judge_no_api._build_prompt(
            "Q", "A", "F", ["exception", "proviso"]
        )
        assert "exception, proviso" in prompt

    def test_delegation_injects_note(self, judge_no_api: HybridJudge) -> None:
        prompt = judge_no_api._build_prompt("Q", "A", "F", ["delegation"])
        # 「政令で定める者」の解決済み情報が注入される
        assert "委任規定の解決済み情報" in prompt
        assert "七十歳以上" in prompt
        assert "児童" in prompt

    def test_no_delegation_no_note(self, judge_no_api: HybridJudge) -> None:
        prompt = judge_no_api._build_prompt("Q", "A", "F", ["proviso"])
        assert "委任規定の解決済み情報" not in prompt

    def test_json_schema_is_specified(self, judge_no_api: HybridJudge) -> None:
        prompt = judge_no_api._build_prompt("Q", "A", "F", [])
        # Gemini に JSON 形式を強制するスキーマフィールドが含まれる
        assert "judgement" in prompt
        assert "article" in prompt
        assert "fine" in prompt
        assert "reasoning" in prompt


# ---------------------------------------------------------------------------
# _lookup_fine: クエリ × 条文から反則金テーブルを検索
# ---------------------------------------------------------------------------
class TestLookupFine:
    def test_sidewalk_matches_passage_violations(
        self, judge_no_api: HybridJudge, dummy_article: ArticleNode
    ) -> None:
        info = judge_no_api._lookup_fine("歩道を走行した", dummy_article)
        # 「歩道」は {通行区分違反, 歩道徐行等義務違反} にマップされる
        assert "通行区分違反" in info or "歩道徐行" in info

    def test_signal_matches_signal_ignore(
        self, judge_no_api: HybridJudge, dummy_article: ArticleNode
    ) -> None:
        info = judge_no_api._lookup_fine("信号無視した", dummy_article)
        assert "信号無視" in info
        assert "6,000" in info

    def test_phone_matches_12000(
        self, judge_no_api: HybridJudge, dummy_article: ArticleNode
    ) -> None:
        info = judge_no_api._lookup_fine("スマホを持って走行", dummy_article)
        assert "携帯電話" in info
        assert "12,000" in info

    def test_alcohol_marked_as_criminal_only(
        self, judge_no_api: HybridJudge, dummy_article: ArticleNode
    ) -> None:
        info = judge_no_api._lookup_fine("酒気帯びで運転", dummy_article)
        # 酒気帯びは criminal_only → 反則金対象外・刑事罰 表示
        assert "反則金対象外" in info
        assert "刑事罰" in info

    def test_two_ride_maps_to_vehicle_load_violation(
        self, judge_no_api: HybridJudge, dummy_article: ArticleNode
    ) -> None:
        info = judge_no_api._lookup_fine("二人乗りをした", dummy_article)
        assert "軽車両乗車積載制限違反" in info
        assert "3,000" in info

    def test_unmatched_query_falls_back_to_article_num(
        self, judge_no_api: HybridJudge
    ) -> None:
        # クエリキーワードにヒットしないが、第7条 → 信号無視6,000円 が逆引きで出る
        article_7 = ArticleNode(num="7", caption="(信号)", title="第七条")
        info = judge_no_api._lookup_fine("説明不能な状況", article_7)
        assert "信号無視" in info
        assert "6,000" in info

    def test_no_match_at_all_returns_fallback(
        self, judge_no_api: HybridJudge
    ) -> None:
        # クエリキーワードにも条文番号にもヒットしない
        orphan = ArticleNode(num="9999", caption="", title="第九千九百九十九条")
        info = judge_no_api._lookup_fine("完全に無関係な内容", orphan)
        assert "該当する反則金情報なし" in info

    def test_result_is_multiline_when_multiple_hits(
        self, judge_no_api: HybridJudge, dummy_article: ArticleNode
    ) -> None:
        # 「踏切」→ {遮断踏切立入り, 踏切不停止等} の2件がヒット → 改行区切り
        info = judge_no_api._lookup_fine("踏切を通過した", dummy_article)
        assert "遮断踏切立入り" in info
        assert "踏切不停止等" in info
        assert "\n" in info


# ---------------------------------------------------------------------------
# _load_fine_table: JSON ファイル読み込み
# ---------------------------------------------------------------------------
class TestLoadFineTable:
    def test_returns_expected_top_level_keys(self) -> None:
        data = HybridJudge._load_fine_table()
        assert "fines" in data
        assert "criminal_only" in data
        assert "metadata" in data

    def test_fines_entries_have_required_fields(self) -> None:
        data = HybridJudge._load_fine_table()
        for fine in data["fines"]:
            assert "amount" in fine
            assert "violation" in fine
            assert "article" in fine
            assert isinstance(fine["amount"], int)

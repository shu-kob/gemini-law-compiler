"""Unit tests for src/benchmark/flash_only_judge.py

Gemini API を呼び出さない純関数部分（_check_answer / _detect_hallucination /
print_summary / TEST_CASES の妥当性）のみを対象とする。
"""

from __future__ import annotations

import pytest

from src.benchmark.flash_only_judge import (
    BenchmarkResult,
    TEST_CASES,
    TestCase,
    _check_answer,
    _detect_hallucination,
    print_summary,
)


# ---------------------------------------------------------------------------
# _check_answer: 回答文に期待条文と判定（合法/違反）が含まれるか
# ---------------------------------------------------------------------------
class TestCheckAnswer:
    @pytest.fixture
    def tc_violation(self) -> TestCase:
        return TestCase(
            id="TC-X",
            scenario="テスト用シナリオ",
            expected_answer="違反。反則金6,000円。",
            expected_article="第7条",
            failure_type="number_fabrication",
            description="",
        )

    @pytest.fixture
    def tc_legal(self) -> TestCase:
        return TestCase(
            id="TC-Y",
            scenario="",
            expected_answer="合法。第63条の4第1項第2号により認められる。",
            expected_article="第63条の4",
            failure_type="reference_missing",
            description="",
        )

    def test_correct_answer_with_article_and_judgement(self, tc_violation: TestCase) -> None:
        answer = "これは違反です。根拠は第7条に基づく。反則金は6,000円です。"
        assert _check_answer(answer, tc_violation) is True

    def test_missing_article_fails(self, tc_violation: TestCase) -> None:
        # 条文番号が答えにない場合は False（代替表記にも該当しない）
        answer = "これは違反です。反則金は6,000円です。"
        assert _check_answer(answer, tc_violation) is False

    def test_wrong_judgement_fails_when_expected_violation(self, tc_violation: TestCase) -> None:
        # 期待=違反 なのに回答が合法のみ → False
        answer = "合法です。第7条に基づけば問題ありません。"
        assert _check_answer(answer, tc_violation) is False

    def test_wrong_judgement_fails_when_expected_legal(self, tc_legal: TestCase) -> None:
        # 期待=合法 なのに回答が違反のみ → False
        answer = "違反です。第63条の4第1項第2号を適用。"
        assert _check_answer(answer, tc_legal) is False

    def test_both_judgements_mentioned_is_accepted(self, tc_violation: TestCase) -> None:
        # 「違反にあたるが合法とも読める」等、両語を含む場合は判定チェックを通過する
        answer = "第7条を参照。原則違反だが一定条件では合法の可能性も言及される。"
        assert _check_answer(answer, tc_violation) is True

    def test_alternative_article_notation_accepted(self) -> None:
        # 「条の」→「-」表記で代替が効く仕様
        tc = TestCase(
            id="TC-Z",
            scenario="",
            expected_answer="違反。",
            expected_article="第63条の4",
            failure_type="hierarchy_ignore",
            description="",
        )
        answer = "違反です。根拠は63-4第1項第2号。"
        # "第63条の4" は含まれないが、"63-4" に代替で一致
        assert _check_answer(answer, tc) is True


# ---------------------------------------------------------------------------
# _detect_hallucination: ハルシネーション種別ごとの検出
# ---------------------------------------------------------------------------
class TestDetectHallucination:
    def test_no_issues_returns_empty_string(self) -> None:
        tc = TestCase(
            id="TC-A",
            scenario="",
            expected_answer="",
            expected_article="",
            failure_type="number_fabrication",
            description="",
        )
        # 正規の反則金額 6,000 円のみを含む回答
        answer = "反則金6,000円です。"
        assert _detect_hallucination(answer, tc) == ""

    def test_fabricated_number_detected(self) -> None:
        tc = TestCase(
            id="TC-B",
            scenario="",
            expected_answer="",
            expected_article="",
            failure_type="number_fabrication",
            description="",
        )
        # 2,500 円は反則金テーブルに存在しないため検出
        issues = _detect_hallucination("反則金は2,500円です。", tc)
        assert "数値捏造" in issues
        assert "2500" in issues or "2,500" in issues

    def test_criminal_penalty_amounts_are_allowlisted(self) -> None:
        tc = TestCase(
            id="TC-C",
            scenario="",
            expected_answer="",
            expected_article="",
            failure_type="number_fabrication",
            description="",
        )
        # 50万円 / 100万円は刑事罰の罰金として許可リストに含まれる
        assert _detect_hallucination("50万円の罰金です。", tc) == ""
        # 数値は「500,000円」として抽出される
        assert _detect_hallucination("500,000円の罰金です。", tc) == ""
        assert _detect_hallucination("1,000,000円の罰金です。", tc) == ""

    def test_reference_missing_detected(self) -> None:
        tc = TestCase(
            id="TC-D",
            scenario="",
            expected_answer="",
            expected_article="",
            failure_type="reference_missing",
            description="",
        )
        # 「政令/施行令/70歳/高齢者/児童」のいずれにも言及なし
        answer = "歩道通行は違反です。"
        assert "参照欠落" in _detect_hallucination(answer, tc)

    def test_reference_missing_skipped_when_seniors_mentioned(self) -> None:
        tc = TestCase(
            id="TC-E",
            scenario="",
            expected_answer="",
            expected_article="",
            failure_type="reference_missing",
            description="",
        )
        # 「70歳」が言及されていれば参照欠落は検出されない
        answer = "70歳以上は歩道通行可。"
        assert "参照欠落" not in _detect_hallucination(answer, tc)

    def test_hierarchy_ignore_detected(self) -> None:
        tc = TestCase(
            id="TC-F",
            scenario="",
            expected_answer="",
            expected_article="",
            failure_type="hierarchy_ignore",
            description="",
        )
        # 「ただし/例外/やむを得ない」のいずれにも言及なし
        answer = "歩道通行は原則違反。"
        assert "階層無視" in _detect_hallucination(answer, tc)

    def test_hierarchy_ignore_skipped_when_exception_mentioned(self) -> None:
        tc = TestCase(
            id="TC-G",
            scenario="",
            expected_answer="",
            expected_article="",
            failure_type="hierarchy_ignore",
            description="",
        )
        answer = "原則違反だが、ただしやむを得ない場合は例外として許容。"
        assert "階層無視" not in _detect_hallucination(answer, tc)


# ---------------------------------------------------------------------------
# TEST_CASES の構造的整合性
# ---------------------------------------------------------------------------
class TestTestCasesDataset:
    def test_all_cases_have_required_fields(self) -> None:
        for tc in TEST_CASES:
            assert tc.id
            assert tc.scenario
            assert tc.expected_answer
            assert tc.expected_article
            assert tc.failure_type in {
                "hierarchy_ignore",
                "number_fabrication",
                "reference_missing",
            }

    def test_ids_are_unique(self) -> None:
        ids = [tc.id for tc in TEST_CASES]
        assert len(ids) == len(set(ids))

    def test_covers_all_failure_types(self) -> None:
        types = {tc.failure_type for tc in TEST_CASES}
        # ベンチマークは3種のハルシネーション全てを網羅すべき
        assert types == {
            "hierarchy_ignore",
            "number_fabrication",
            "reference_missing",
        }


# ---------------------------------------------------------------------------
# print_summary: 出力の形式（副作用）
# ---------------------------------------------------------------------------
class TestPrintSummary:
    def test_prints_total_and_correct_counts(self, capsys) -> None:
        tc = TEST_CASES[0]
        results = [
            BenchmarkResult(
                test_case=tc,
                flash_answer="...",
                is_correct=True,
                hallucination_detected="",
                response_time_ms=100,
            ),
            BenchmarkResult(
                test_case=tc,
                flash_answer="...",
                is_correct=False,
                hallucination_detected="数値捏造: 2500円",
                response_time_ms=200,
            ),
        ]
        print_summary(results)
        out = capsys.readouterr().out
        assert "1/2" in out  # 正答率
        assert "50%" in out

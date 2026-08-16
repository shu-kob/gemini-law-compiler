"""Unit tests for src/benchmark/matrix_benchmark.py"""

from __future__ import annotations

import types
import pytest

from src.benchmark.matrix_benchmark import (
    MatrixBenchmarkRunner,
    PatternExecutionResult,
    CaseMatrixResult,
    save_matrix_results,
)
from src.benchmark.flash_only_judge import TestCase


class FakeUsageMetadata:
    def __init__(self, prompt: int = 100, candidates: int = 50, total: int = 150):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = total


class FakeResponse:
    def __init__(self, text: str, usage=None):
        self.text = text
        self.usage_metadata = usage or FakeUsageMetadata()


class FakeClient:
    def __init__(self):
        self.models = self
        self.calls = []

    def generate_content(self, model: str, contents: str, config: dict):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return FakeResponse(text='{"judgement": "合法", "fine": "0円", "reasoning": "テスト"}')


@pytest.fixture
def mock_runner(monkeypatch: pytest.MonkeyPatch):
    fake_client = FakeClient()
    runner = MatrixBenchmarkRunner.__new__(MatrixBenchmarkRunner)
    runner.model = "fake-gemini-model"
    runner.thinking_budget = 2048
    runner.client = fake_client
    runner.raw_law_text = "第63条の4 歩道通行"
    runner.fine_table = {
        "通行区分違反": {
            "fine_yen": 6000,
            "article": "第17条第1項",
            "penalty_standard": "3月以下の懲役又は5万円以下の罰金",
            "blue_ticket_eligible": True,
        }
    }

    # vsm モック
    class DummyArticle:
        title = "第63条の4"
        caption = "（普通自転車の歩道通行）"
        paragraphs = []

    class DummyMatch:
        article = DummyArticle()

    class DummyVSM:
        def search(self, query: str, top_k: int = 3):
            return [DummyMatch()]

    runner.vsm = DummyVSM()
    return runner, fake_client



def test_matrix_runner_patterns(mock_runner):
    runner, fake_client = mock_runner

    case = TestCase(
        id="TC-TEST",
        scenario="テストケース: 70歳以上高齢者の歩道通行",
        expected_answer="合法",
        expected_article="第63条の4",
        failure_type="reference_missing",
        description="テスト説明",
    )

    result = runner.evaluate_case(case, verbose=False)
    assert isinstance(result, CaseMatrixResult)
    assert len(result.patterns) == 4

    p1 = result.patterns["P1"]
    assert p1.pattern_id == "P1_RAW_NOTHINK"
    assert p1.thinking_enabled is False
    assert p1.has_preprocessing is False

    p2 = result.patterns["P2"]
    assert p2.pattern_id == "P2_RAW_THINK"
    assert p2.thinking_enabled is True
    assert p2.has_preprocessing is False

    p3 = result.patterns["P3"]
    assert p3.pattern_id == "P3_PREP_THINK"
    assert p3.thinking_enabled is True
    assert p3.has_preprocessing is True

    p4 = result.patterns["P4"]
    assert p4.pattern_id == "P4_PREP_NOTHINK"
    assert p4.thinking_enabled is False
    assert p4.has_preprocessing is True


def test_save_matrix_results(tmp_path, monkeypatch):
    monkeypatch.setattr("src.benchmark.matrix_benchmark.RESULTS_DIR", tmp_path)

    dummy_pres = PatternExecutionResult(
        pattern_id="P1_RAW_NOTHINK",
        pattern_name="① Raw × Thinking OFF",
        has_preprocessing=False,
        thinking_enabled=False,
        thinking_budget=0,
        response_text="test response",
        response_time_ms=120,
        prompt_tokens=100,
        candidates_tokens=50,
        total_tokens=150,
    )

    case = TestCase(
        id="TC-001",
        scenario="75歳の高齢者",
        expected_answer="合法",
        expected_article="第63条の4",
        failure_type="reference_missing",
        description="テスト",
    )

    case_res = CaseMatrixResult(
        test_case=case,
        patterns={
            "P1": dummy_pres,
            "P2": dummy_pres,
            "P3": dummy_pres,
            "P4": dummy_pres,
        },
    )

    json_p, md_p = save_matrix_results([case_res], model="test-model")
    assert json_p.exists()
    assert md_p.exists()

    md_content = md_p.read_text(encoding="utf-8")
    assert "2×2 マトリクス検証レポート" in md_content
    assert "TC-001" in md_content

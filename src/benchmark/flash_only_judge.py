"""
Task 1: Failure Reproduction — Gemini Flash単体ベンチマーク

素のGemini-3-Flashに道交法を丸投げし、エッジケースを判定させる。
期待される「誤答」をログに記録し、ブログ用の比較データとする。

想定されるFlashの失敗パターン:
  1. 階層無視: 原則→例外→例外の例外のネストを平坦化
  2. 数値捏造: 条文にない反則金額をWeb知識から補完
  3. 参照欠落: 「政令で定める者」のポインタを解決できない
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from src.config import GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL, FINE_TABLE_PATH, get_genai_client


@dataclass
class TestCase:
    id: str
    scenario: str
    expected_answer: str
    expected_article: str
    failure_type: str  # "hierarchy_ignore" | "number_fabrication" | "reference_missing"
    description: str


@dataclass
class BenchmarkResult:
    test_case: TestCase
    flash_answer: str
    is_correct: bool
    hallucination_detected: str
    response_time_ms: int


# --- ベンチマーク用テストケース ---
TEST_CASES: list[TestCase] = [
    TestCase(
        id="TC-001",
        scenario="75歳の高齢者が普通自転車で歩道を走行しています。これは違反ですか？反則金はいくらですか？",
        expected_answer="合法。第63条の4第1項第2号により、70歳以上の者は歩道通行が認められる（施行令第26条）。反則金は発生しない。",
        expected_article="第63条の4",
        failure_type="reference_missing",
        description="Flashは「政令で定める者」を解決できず、歩道通行=違反と判定する可能性が高い。"
                    "また、存在しない反則金額を捏造する可能性がある。",
    ),
    TestCase(
        id="TC-002",
        scenario="成人（30歳）が普通自転車で、自転車通行可の標識がない歩道を走行しています。これは違反ですか？反則金はいくらですか？",
        expected_answer="原則違反。通行区分違反で反則金6,000円。ただし車道の状況がやむを得ない場合は第63条の4第1項第3号により合法。",
        expected_article="第63条の4、第17条",
        failure_type="hierarchy_ignore",
        description="原則（車道通行義務）→例外（歩道通行可の条件）→例外の例外（やむを得ない場合）"
                    "の三層構造をFlashが正しくハンドルできるか。",
    ),
    TestCase(
        id="TC-003",
        scenario="自転車でスマートフォンを手に持ちながら運転しました。反則金はいくらですか？",
        expected_answer="違反。携帯電話使用等（ながらスマホ）で反則金12,000円。",
        expected_article="第71条第5号の5",
        failure_type="number_fabrication",
        description="2026年4月施行の青切符制度の正確な反則金額(12,000円)を回答できるか。"
                    "Flashは古い情報や不正確な金額を回答する可能性がある。",
    ),
    TestCase(
        id="TC-004",
        scenario="10歳の児童が普通自転車で歩道を走行しています。これは違反ですか？",
        expected_answer="合法。第63条の4第1項第2号により、児童は歩道通行が認められる。"
                       "また、16歳未満なので青切符の対象外。",
        expected_article="第63条の4",
        failure_type="reference_missing",
        description="児童の歩道通行許可 + 青切符の年齢制限（16歳以上）の二重の条件を正しく処理できるか。",
    ),
    TestCase(
        id="TC-005",
        scenario="自転車で酒気帯び運転をしました。反則金はいくらですか？",
        expected_answer="酒気帯び運転は青切符（反則金）の対象外。赤切符による刑事罰（3年以下の懲役又は50万円以下の罰金）。",
        expected_article="第65条、第117条の2の2",
        failure_type="number_fabrication",
        description="酒気帯び運転は反則金対象外（刑事罰）であることを正しく区別できるか。"
                    "Flashは反則金額を捏造する可能性が高い。",
    ),
    TestCase(
        id="TC-006",
        scenario="自転車で歩道を通行中、徐行せずに時速20kmで走行しました。反則金はいくらですか？",
        expected_answer="違反。歩道徐行等義務違反で反則金3,000円。第63条の4第2項。",
        expected_article="第63条の4第2項",
        failure_type="number_fabrication",
        description="歩道徐行義務違反の正確な反則金額(3,000円)を回答できるか。",
    ),
    TestCase(
        id="TC-007",
        scenario="自転車で赤信号を無視して交差点に進入しました。反則金はいくらですか？",
        expected_answer="違反。信号無視で反則金6,000円。",
        expected_article="第7条",
        failure_type="number_fabrication",
        description="最も基本的な違反の反則金額を正しく回答できるか。",
    ),
]


SYSTEM_PROMPT = """\
あなたは日本の道路交通法の専門家です。
2026年4月1日施行の自転車交通反則通告制度（青切符制度）に基づいて回答してください。

以下のルールを厳守してください:
1. 根拠となる条文番号を必ず明示すること
2. 反則金額を回答する場合は正確な金額を示すこと
3. 反則金対象外（刑事罰）の場合はその旨を明示すること
4. 「政令で定める」等の委任規定がある場合は、その内容も含めて回答すること

回答はJSON形式で以下の構造で返してください:
{
  "judgement": "合法" or "違反",
  "article": "根拠条文",
  "fine": "反則金額（該当する場合）",
  "reasoning": "判定理由の説明"
}
"""


def run_flash_benchmark(
    test_cases: list[TestCase] | None = None,
    verbose: bool = True,
    model: str = GEMINI_FLASH_MODEL,
) -> list[BenchmarkResult]:
    """Gemini単体でベンチマークを実行する（デフォルトはFlash、Pro等も指定可）"""
    client = get_genai_client()

    cases = test_cases or TEST_CASES
    results: list[BenchmarkResult] = []

    for tc in cases:
        if verbose:
            print(f"\n{'='*60}")
            print(f"[2026-AI-Logic]: テストケース {tc.id} 実行中...")
            print(f"[2026-AI-Logic]: シナリオ: {tc.scenario}")

        start = time.monotonic_ns()
        response = client.models.generate_content(
            model=model,
            contents=tc.scenario,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.0,
            },
        )
        elapsed_ms = (time.monotonic_ns() - start) // 1_000_000

        flash_answer = response.text or "(empty response)"

        # 正解判定（簡易: 期待される条文が含まれるか）
        is_correct = _check_answer(flash_answer, tc)
        hallucination = _detect_hallucination(flash_answer, tc)

        result = BenchmarkResult(
            test_case=tc,
            flash_answer=flash_answer,
            is_correct=is_correct,
            hallucination_detected=hallucination,
            response_time_ms=elapsed_ms,
        )
        results.append(result)

        if verbose:
            marker = "✓" if is_correct else "✗"
            print(f"[2026-AI-Logic]: {marker} 回答 ({elapsed_ms}ms):")
            print(f"  {flash_answer[:300]}")
            if hallucination:
                print(f"[2026-AI-Logic]: ⚠ ハルシネーション検知: {hallucination}")
            print(f"[2026-AI-Logic]: 期待される正解: {tc.expected_answer}")

    return results


def _check_answer(answer: str, tc: TestCase) -> bool:
    """回答が正しいかの簡易チェック"""
    answer_lower = answer.lower()

    # 期待される条文が言及されているか
    if tc.expected_article not in answer:
        # 条文番号の表記揺れに対応
        alt = tc.expected_article.replace("第", "").replace("条の", "-")
        if alt not in answer and tc.expected_article.replace("第", "") not in answer:
            return False

    # 合法/違反の判定が合っているか
    if "合法" in tc.expected_answer:
        if "違反" in answer and "合法" not in answer:
            return False
    elif "違反" in tc.expected_answer:
        if "合法" in answer and "違反" not in answer:
            return False

    return True


def _detect_hallucination(answer: str, tc: TestCase) -> str:
    """ハルシネーションの種類を検出する"""
    issues: list[str] = []

    fine_table = _load_fine_table()

    if tc.failure_type == "number_fabrication":
        # 回答中の金額を抽出
        import re
        amounts = re.findall(r"([\d,]+)円", answer)
        for amount_str in amounts:
            amount = int(amount_str.replace(",", ""))
            # 正規の反則金テーブルに存在するか確認
            valid_amounts = {f["amount"] for f in fine_table.get("fines", [])}
            if amount not in valid_amounts and amount not in {500000, 1000000}:
                issues.append(f"数値捏造: {amount}円は正規の反則金額に存在しない")

    if tc.failure_type == "reference_missing":
        if "政令" not in answer and "施行令" not in answer:
            if "70歳" not in answer and "高齢者" not in answer and "児童" not in answer:
                issues.append("参照欠落: 政令委任規定を解決していない")

    if tc.failure_type == "hierarchy_ignore":
        if "ただし" not in answer and "例外" not in answer and "やむを得ない" not in answer:
            issues.append("階層無視: 例外規定への言及がない")

    return "; ".join(issues) if issues else ""


_fine_table_cache: dict | None = None


def _load_fine_table() -> dict:
    global _fine_table_cache
    if _fine_table_cache is None:
        with open(FINE_TABLE_PATH, encoding="utf-8") as f:
            _fine_table_cache = json.load(f)
    return _fine_table_cache


def print_summary(results: list[BenchmarkResult]) -> None:
    """ベンチマーク結果のサマリーを出力"""
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    hallucinated = sum(1 for r in results if r.hallucination_detected)

    print(f"\n{'='*60}")
    print("[2026-AI-Logic]: === Flash単体ベンチマーク結果 ===")
    print(f"[2026-AI-Logic]: 正答率: {correct}/{total} ({100*correct/total:.0f}%)")
    print(f"[2026-AI-Logic]: ハルシネーション検知: {hallucinated}/{total}")
    print()

    for r in results:
        marker = "✓" if r.is_correct else "✗"
        hal = f" ⚠{r.hallucination_detected}" if r.hallucination_detected else ""
        print(f"  {marker} {r.test_case.id}: {r.test_case.failure_type}{hal}")


if __name__ == "__main__":
    print("[2026-AI-Logic]: Gemini Flash単体ベンチマーク開始")
    print("[2026-AI-Logic]: モデル:", GEMINI_FLASH_MODEL)
    print("[2026-AI-Logic]: 青切符制度の知識なしで法規判定を試みます...")

    results = run_flash_benchmark()
    print_summary(results)

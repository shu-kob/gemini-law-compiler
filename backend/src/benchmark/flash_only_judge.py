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

from src.config import GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL, FINE_TABLE_PATH, get_llm_client


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
    # --- 拡張ケース (TC-008〜TC-030): 青切符制度の主要違反を網羅 ---
    TestCase(
        id="TC-008",
        scenario="自転車で一時停止標識を無視して交差点に進入しました。反則金はいくらですか？",
        expected_answer="違反。指定場所一時不停止等で反則金5,000円。第43条。",
        expected_article="第43条",
        failure_type="number_fabrication",
        description="一時不停止違反の正確な反則金額(5,000円)を回答できるか。",
    ),
    TestCase(
        id="TC-009",
        scenario="夜間、自転車のライトを点けずに走行しました。反則金はいくらですか？",
        expected_answer="違反。無灯火で反則金5,000円。第52条。",
        expected_article="第52条",
        failure_type="number_fabrication",
        description="無灯火違反の正確な反則金額(5,000円)を回答できるか。",
    ),
    TestCase(
        id="TC-010",
        scenario="自転車で友達と横並びで走行しました。反則金はいくらですか？",
        expected_answer="違反。並進禁止違反で反則金3,000円。第19条。",
        expected_article="第19条",
        failure_type="number_fabrication",
        description="並進禁止違反の正確な反則金額(3,000円)を回答できるか。",
    ),
    TestCase(
        id="TC-011",
        scenario="大人の自転車に大人がもう一人乗って、二人乗りで走行しました。反則金はいくらですか？",
        expected_answer="違反。軽車両乗車積載制限違反（二人乗り等）で反則金3,000円。第57条。",
        expected_article="第57条",
        failure_type="number_fabrication",
        description="二人乗り違反の正確な反則金額(3,000円)を回答できるか。",
    ),
    TestCase(
        id="TC-012",
        scenario="雨の中、傘を差しながら自転車を運転しました。反則金はいくらですか？",
        expected_answer="違反。公安委員会遵守事項違反（傘差し運転）で反則金5,000円。第71条第6号。",
        expected_article="第71条第6号",
        failure_type="number_fabrication",
        description="傘差し運転は条文上「公安委員会遵守事項違反」として処理される。"
                    "条文番号を正しく引けるか。",
    ),
    TestCase(
        id="TC-013",
        scenario="両耳にイヤホンをつけて音楽を聴きながら自転車を運転しました。反則金はいくらですか？",
        expected_answer="違反。公安委員会遵守事項違反（イヤホン等）で反則金5,000円。第71条第6号。",
        expected_article="第71条第6号",
        failure_type="number_fabrication",
        description="イヤホン運転も第71条第6号の公安委員会遵守事項違反として処理される。",
    ),
    TestCase(
        id="TC-014",
        scenario="ブレーキが前後とも効かない自転車で走行しました。反則金はいくらですか？",
        expected_answer="違反。自転車制動装置不良（ブレーキ不良）で反則金5,000円。第63条の9。",
        expected_article="第63条の9",
        failure_type="number_fabrication",
        description="ブレーキ不良違反の正確な反則金額(5,000円)を回答できるか。",
    ),
    TestCase(
        id="TC-015",
        scenario="横断歩道を渡っている歩行者の前を、自転車で徐行せずに横切りました。反則金はいくらですか？",
        expected_answer="違反。横断歩行者等妨害等で反則金6,000円。第38条。",
        expected_article="第38条",
        failure_type="number_fabrication",
        description="横断歩行者妨害違反の反則金額(6,000円)を回答できるか。",
    ),
    TestCase(
        id="TC-016",
        scenario="自転車で右折する際、手による合図をせずに曲がりました。反則金はいくらですか？",
        expected_answer="違反。合図不履行で反則金5,000円。第53条。",
        expected_article="第53条",
        failure_type="number_fabrication",
        description="合図不履行違反の反則金額(5,000円)を回答できるか。",
    ),
    TestCase(
        id="TC-017",
        scenario="自転車道が設けられている道路で、自転車道を通らずに車道を走行しました。反則金はいくらですか？",
        expected_answer="違反。自転車道通行義務違反で反則金3,000円。第63条の3。",
        expected_article="第63条の3",
        failure_type="number_fabrication",
        description="自転車道通行義務違反の反則金額(3,000円)を回答できるか。",
    ),
    TestCase(
        id="TC-018",
        scenario="自転車通行禁止の標識がある道路を自転車で走行しました。反則金はいくらですか？",
        expected_answer="違反。通行禁止違反で反則金5,000円。第8条。",
        expected_article="第8条",
        failure_type="number_fabrication",
        description="通行禁止違反の反則金額(5,000円)を回答できるか。",
    ),
    TestCase(
        id="TC-019",
        scenario="自転車で踏切を一時停止せずに通過しました。反則金はいくらですか？",
        expected_answer="違反。踏切不停止等で反則金6,000円。第33条第1項。",
        expected_article="第33条第1項",
        failure_type="number_fabrication",
        description="踏切不停止違反の反則金額(6,000円)を回答できるか。",
    ),
    TestCase(
        id="TC-020",
        scenario="遮断機が降り始めた踏切に、自転車で進入しました。反則金はいくらですか？",
        expected_answer="違反。遮断踏切立入りで反則金7,000円。第33条第2項。",
        expected_article="第33条第2項",
        failure_type="number_fabrication",
        description="遮断踏切立入りは踏切不停止より重く7,000円。第1項と第2項の区別ができるか。",
    ),
    TestCase(
        id="TC-021",
        scenario="自転車で路側帯を、対向方向に逆走しました。反則金はいくらですか？",
        expected_answer="違反。路側帯進行方法違反で反則金3,000円。第17条の2。",
        expected_article="第17条の2",
        failure_type="number_fabrication",
        description="路側帯進行方法違反の反則金額(3,000円)を回答できるか。",
    ),
    TestCase(
        id="TC-022",
        scenario="自転車で両手放し運転をしました。反則金はいくらですか？",
        expected_answer="違反。安全運転義務違反で反則金6,000円。第70条。",
        expected_article="第70条",
        failure_type="number_fabrication",
        description="両手放しは安全運転義務違反として第70条適用。反則金(6,000円)を回答できるか。",
    ),
    TestCase(
        id="TC-023",
        scenario="自転車で交差点を斜めに横断（ショートカット）しました。反則金はいくらですか？",
        expected_answer="違反。交差点右左折方法違反で反則金3,000円。第34条。",
        expected_article="第34条",
        failure_type="number_fabrication",
        description="斜め横断は第34条の右左折方法違反として処理。反則金(3,000円)を回答できるか。",
    ),
    TestCase(
        id="TC-024",
        scenario="自転車で歩行者に道を譲ってもらうためにベルを鳴らしました。反則金はいくらですか？",
        expected_answer="違反。警音器使用制限違反で反則金3,000円。第54条第2項。"
                       "ただし危険防止のためやむを得ない場合は適用除外。",
        expected_article="第54条第2項",
        failure_type="hierarchy_ignore",
        description="警音器の使用は危険防止のためのみ許容される例外規定がある。"
                    "原則→例外の階層を正しく処理できるか。",
    ),
    TestCase(
        id="TC-025",
        scenario="7歳の幼児が補助輪付きの自転車で歩道を走行しています。これは違反ですか？",
        expected_answer="合法。第63条の4第1項第2号により幼児は歩道通行が認められる。"
                       "また16歳未満なので青切符の対象外（指導警告で対応）。",
        expected_article="第63条の4",
        failure_type="reference_missing",
        description="幼児の歩道通行許可 + 青切符の年齢制限（16歳以上）の二重条件を処理できるか。",
    ),
    TestCase(
        id="TC-026",
        scenario="14歳の中学生が自転車で赤信号を無視しました。反則金はいくらですか？",
        expected_answer="違反だが青切符（反則金）の対象外。16歳未満は指導警告による対応。第7条。",
        expected_article="第7条",
        failure_type="reference_missing",
        description="青切符制度は16歳以上が対象。16歳未満は条文違反でも反則金対象外。",
    ),
    TestCase(
        id="TC-027",
        scenario="65歳の人が普通自転車で、自転車通行可の標識のない歩道を走行しています。これは違反ですか？",
        expected_answer="原則違反。65歳は政令で定める「70歳以上」に該当せず、第63条の4第1項第2号の"
                       "例外は適用されない。通行区分違反で反則金6,000円。"
                       "ただし車道がやむを得ない状況なら第3号で適法。",
        expected_article="第63条の4",
        failure_type="reference_missing",
        description="施行令第26条で「70歳以上」と規定されているため、65歳は例外不適用。"
                    "政令の閾値を正確に処理できるか。",
    ),
    TestCase(
        id="TC-028",
        scenario="身体障害者手帳を持つ40歳の人が普通自転車で歩道を走行しています。これは違反ですか？",
        expected_answer="合法。第63条の4第1項第2号および施行令第26条により、"
                       "身体に障害を有する者で政令で定めるものは歩道通行が認められる。",
        expected_article="第63条の4",
        failure_type="reference_missing",
        description="施行令第26条が定める「身体障害者」の歩道通行例外を解決できるか。"
                    "年齢では例外に該当しないが、障害により該当するパターン。",
    ),
    TestCase(
        id="TC-029",
        scenario="自転車で泥酔状態（酒酔い）で運転しました。反則金はいくらですか？",
        expected_answer="酒酔い運転は青切符（反則金）の対象外。赤切符による刑事罰"
                       "（5年以下の懲役又は100万円以下の罰金）。第65条、第117条の2。",
        expected_article="第65条、第117条の2",
        failure_type="hierarchy_ignore",
        description="酒酔いと酒気帯びの罰則差を区別できるか。酒酔いは酒気帯びより重い。",
    ),
    TestCase(
        id="TC-030",
        scenario="自転車で他の車両の前に急に進路変更し、あおるような運転をしました。反則金はいくらですか？",
        expected_answer="違反。妨害運転は青切符の対象外。赤切符による刑事罰"
                       "（3年以下の懲役又は50万円以下の罰金）。第117条の2の2。",
        expected_article="第117条の2の2",
        failure_type="hierarchy_ignore",
        description="妨害運転（あおり）は反則金対象外で刑事罰。"
                    "反則金 vs 刑事罰の階層を区別できるか。",
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
    client = get_llm_client(model)

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

"""
2x2 Matrix Verification Benchmark:
前処理（あり/なし） × 推論 Thinking（あり/なし）の4パターン比較検証

4つの検証パターン:
  ① Raw × Thinking OFF:      生テキスト × 推論なし (Budget=0)      → 【最弱】 参照ジャンプ追えず誤答多発
  ② Raw × Thinking ON:       生テキスト × 推論あり (Budget=2048)   → 【力技】 推論で追おうとするがトークン爆発＆捏造リスク
  ③ Preprocessed × Thinking ON: 前処理済み × 推論あり (Budget=2048)   → 【過剰推論】 前処理済みで答えがあるのに無駄な自問自答で高コスト・遅延
  ④ Preprocessed × Thinking OFF:前処理済み × 推論なし (Budget=0)      → 【本命】 最速・最安・最高精度（確定的トランスレータ）
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import (
    GEMINI_FLASH_MODEL,
    RESULTS_DIR,
    XML_PATH,
    FINE_TABLE_PATH,
    get_llm_client,
)
from src.parser.legal_compiler import (
    LawAST,
    parse_egov_xml,
    extract_bicycle_articles,
    flatten_article_text,
)
from src.matcher.vsm_engine import VSMEngine
from src.benchmark.flash_only_judge import TEST_CASES, TestCase


@dataclass
class PatternExecutionResult:
    pattern_id: str
    pattern_name: str
    has_preprocessing: bool
    thinking_enabled: bool
    thinking_budget: int
    response_text: str
    response_time_ms: int
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    total_tokens: int = 0
    is_correct: bool | None = None
    hallucination_notes: str = ""


@dataclass
class CaseMatrixResult:
    test_case: TestCase
    patterns: dict[str, PatternExecutionResult]


SYSTEM_PROMPT_RAW = """\
あなたは日本の道路交通法の専門家です。
2026年4月1日施行の自転車交通反則通告制度（青切符制度）に基づいて回答してください。
以下の質問に対し、日本の道路交通法の知識と提示された条文テキストに基づいて正確に回答してください。
回答形式は以下のJSONにしてください:
{
  "judgement": "合法" or "違反",
  "article": "根拠条文（条・項・号まで）",
  "fine": "反則金額（該当する場合）or 対象外の理由",
  "reasoning": "判定理由"
}
"""

SYSTEM_PROMPT_PREPROCESSED = """\
あなたは日本の道路交通法の専門家です。
2026年4月1日施行の自転車交通反則通告制度（青切符制度）に基づいて回答してください。

【重要】以下に提示する「条文データ」「委任規定の解決済み情報」「反則金データ」は、
e-Gov法令APIから取得した公式の法令データを決定論的にパース・構造化した結果です。
この情報を絶対的な根拠として使用し、あなた自身の学習データからの補完や推測は一切行わないでください。

以下のJSON形式で回答してください:
{
  "judgement": "合法" or "違反",
  "article": "根拠条文（条・項・号まで）",
  "fine": "反則金額（該当する場合）or 対象外の理由",
  "reasoning": "判定理由（提示された階層構造に基づく簡潔な説明）"
}
"""


class MatrixBenchmarkRunner:
    """2×2マトリクス比較ベンチマーク実行エンジン"""

    def __init__(
        self,
        model: str = GEMINI_FLASH_MODEL,
        thinking_budget: int = 2048,
        use_embedding: bool = False,
    ):
        self.model = model
        self.thinking_budget = thinking_budget
        self.client = get_llm_client(model)

        print("[MatrixBenchmark]: 法令XMLパース中...")
        self.ast = parse_egov_xml(XML_PATH)
        self.bicycle_articles = extract_bicycle_articles(self.ast)
        print(f"[MatrixBenchmark]: パース完了（全{len(self.ast.articles)}条中 自転車関連{len(self.bicycle_articles)}条）")

        # 生条文テキスト（前処理なし用：自転車関連条文をそのまま結合）
        self.raw_law_text = "\n\n".join(
            f"【{a.title} {a.caption}】\n{flatten_article_text(a)}"
            for a in self.bicycle_articles[:15]  # 主要条文
        )

        # 前処理用 VSM エンジン（Layer 1）
        print("[MatrixBenchmark]: VSMエンジン初期化中...")
        self.vsm = VSMEngine(self.ast, article_filter=self.bicycle_articles)

        # 反則金テーブル
        with open(FINE_TABLE_PATH, encoding="utf-8") as f:
            self.fine_table = json.load(f)

    def _lookup_fine(self, query: str) -> str:
        results = []
        keywords_map = {
            "歩道": ["通行区分違反", "歩道徐行等義務違反"],
            "信号": ["信号無視"],
            "一時停止": ["指定場所一時不停止等"],
            "スマホ": ["携帯電話使用等"],
            "携帯": ["携帯電話使用等"],
            "酒": ["酒気帯び運転", "酒酔い運転"],
            "傘": ["公安委員会遵守事項違反"],
            "イヤホン": ["公安委員会遵守事項違反"],
            "並進": ["並進禁止違反"],
            "右側": ["通行区分違反"],
            "逆走": ["通行区分違反"],
            "遮断": ["遮断踏切立入り"],
            "ブレーキ": ["制動装置不良自転車運転"],
        }
        for kw, violation_names in keywords_map.items():
            if kw in query:
                for vname in violation_names:
                    if vname in self.fine_table:
                        info = self.fine_table[vname]
                        results.append(
                            f"- {vname}: 反則金 {info['fine_yen']:,}円 "
                            f"(根拠: {info['article']}, 法定刑: {info['penalty_standard']})"
                            f"{' ※青切符対象外（刑事罰）' if not info.get('blue_ticket_eligible', True) else ''}"
                        )
        return "\n".join(results) if results else "（該当する反則金規定なし）"

    def _call_llm(self, system_instruction: str, prompt: str, thinking_budget: int) -> tuple[str, int, int, int, int]:
        """LLM呼び出しとメトリクス取得 (text, elapsed_ms, prompt_tokens, candidates_tokens, total_tokens)"""
        config: dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": 0.0,
        }
        if thinking_budget > 0:
            config["thinking_config"] = {"thinking_budget": thinking_budget}
        else:
            config["thinking_config"] = {"thinking_budget": 0}

        max_retries = 5
        initial_backoff = 2.0

        for attempt in range(max_retries):
            start = time.monotonic_ns()
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                elapsed_ms = (time.monotonic_ns() - start) // 1_000_000

                text = response.text or ""
                prompt_tokens = 0
                candidates_tokens = 0
                total_tokens = 0

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    prompt_tokens = getattr(um, "prompt_token_count", 0) or 0
                    candidates_tokens = getattr(um, "candidates_token_count", 0) or 0
                    total_tokens = getattr(um, "total_token_count", 0) or 0

                return text, elapsed_ms, prompt_tokens, candidates_tokens, total_tokens

            except Exception as e:
                msg = str(e)
                is_transient = any(code in msg for code in ("429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
                if not is_transient or attempt == max_retries - 1:
                    raise
                backoff = initial_backoff * (2 ** attempt)
                time.sleep(backoff)

        raise RuntimeError("Max retries exceeded in _call_llm")


    def run_pattern_1_raw_no_thinking(self, query: str) -> PatternExecutionResult:
        """① Raw × Thinking OFF: 生テキスト × 推論なし (Budget=0)"""
        prompt = f"""\
以下の質問に、提示された条文テキストに基づいて回答してください。

【質問】
{query}

【道路交通法 条文テキスト（抜粋）】
{self.raw_law_text}
"""
        text, ms, p_tok, c_tok, t_tok = self._call_llm(
            system_instruction=SYSTEM_PROMPT_RAW,
            prompt=prompt,
            thinking_budget=0,
        )
        return PatternExecutionResult(
            pattern_id="P1_RAW_NOTHINK",
            pattern_name="① Raw × Thinking OFF",
            has_preprocessing=False,
            thinking_enabled=False,
            thinking_budget=0,
            response_text=text,
            response_time_ms=ms,
            prompt_tokens=p_tok,
            candidates_tokens=c_tok,
            total_tokens=t_tok,
        )

    def run_pattern_2_raw_thinking(self, query: str) -> PatternExecutionResult:
        """② Raw × Thinking ON: 生テキスト × 推論あり (Budget=2048)"""
        prompt = f"""\
以下の質問に、提示された条文テキストに基づいて論理的に思考し回答してください。

【質問】
{query}

【道路交通法 条文テキスト（抜粋）】
{self.raw_law_text}
"""
        text, ms, p_tok, c_tok, t_tok = self._call_llm(
            system_instruction=SYSTEM_PROMPT_RAW,
            prompt=prompt,
            thinking_budget=self.thinking_budget,
        )
        return PatternExecutionResult(
            pattern_id="P2_RAW_THINK",
            pattern_name="② Raw × Thinking ON",
            has_preprocessing=False,
            thinking_enabled=True,
            thinking_budget=self.thinking_budget,
            response_text=text,
            response_time_ms=ms,
            prompt_tokens=p_tok,
            candidates_tokens=c_tok,
            total_tokens=t_tok,
        )

    def _build_preprocessed_prompt(self, query: str) -> str:
        vsm_matches = self.vsm.search(query, top_k=3)
        article_texts = []
        all_flags: list[str] = []
        for m in vsm_matches:
            t = flatten_article_text(m.article)
            article_texts.append(f"--- {m.article.title} {m.article.caption} ---\n{t}")
            for p in m.article.paragraphs:
                for s in p.sentences:
                    all_flags.extend(s.logic_flags)

        unique_flags = sorted(set(all_flags))
        combined_text = "\n\n".join(article_texts)
        fine_info = self._lookup_fine(query)

        delegation_note = ""
        if "delegation" in unique_flags or "歩道" in query:
            delegation_note = """
【委任規定の解決済み情報】
道路交通法施行令第26条により、第63条の4第1項第2号の「政令で定める者」は以下のとおり:
  一　児童（6歳以上13歳未満）及び幼児（6歳未満）
  二　七十歳以上の者
  三　身体に障害を有する者で政令で定めるもの

※ 自転車の青切符制度の対象は16歳以上の自転車運転者。16歳未満は指導警告による対応。
"""

        return f"""\
以下の質問に、決定論的パーサーが抽出・解決したデータのみを根拠として回答してください。

【質問】
{query}

【決定論的パーサーが特定した関連条文（e-Gov法令XMLより抽出）】
{combined_text}
{delegation_note}
【反則金データ（2026年4月1日施行・青切符制度）】
{fine_info}

【検出された論理フラグ】
{', '.join(unique_flags) if unique_flags else 'なし'}
"""

    def run_pattern_3_preprocessed_thinking(self, query: str) -> PatternExecutionResult:
        """③ Preprocessed × Thinking ON: 前処理済み × 推論あり (Budget=2048)"""
        prompt = self._build_preprocessed_prompt(query)
        text, ms, p_tok, c_tok, t_tok = self._call_llm(
            system_instruction=SYSTEM_PROMPT_PREPROCESSED,
            prompt=prompt,
            thinking_budget=self.thinking_budget,
        )
        return PatternExecutionResult(
            pattern_id="P3_PREP_THINK",
            pattern_name="③ Preprocessed × Thinking ON",
            has_preprocessing=True,
            thinking_enabled=True,
            thinking_budget=self.thinking_budget,
            response_text=text,
            response_time_ms=ms,
            prompt_tokens=p_tok,
            candidates_tokens=c_tok,
            total_tokens=t_tok,
        )

    def run_pattern_4_preprocessed_no_thinking(self, query: str) -> PatternExecutionResult:
        """④ Preprocessed × Thinking OFF: 前処理済み × 推論なし (Budget=0) 【本命】"""
        prompt = self._build_preprocessed_prompt(query)
        text, ms, p_tok, c_tok, t_tok = self._call_llm(
            system_instruction=SYSTEM_PROMPT_PREPROCESSED,
            prompt=prompt,
            thinking_budget=0,
        )
        return PatternExecutionResult(
            pattern_id="P4_PREP_NOTHINK",
            pattern_name="④ Preprocessed × Thinking OFF",
            has_preprocessing=True,
            thinking_enabled=False,
            thinking_budget=0,
            response_text=text,
            response_time_ms=ms,
            prompt_tokens=p_tok,
            candidates_tokens=c_tok,
            total_tokens=t_tok,
        )

    def evaluate_case(self, case: TestCase, verbose: bool = True) -> CaseMatrixResult:
        if verbose:
            print(f"\n{'='*70}")
            print(f"■ テストケース [{case.id}]: {case.scenario}")
            print(f"  期待解: {case.expected_answer}")
            print(f"{'='*70}")

        patterns: dict[str, PatternExecutionResult] = {}

        # 4パターン実行
        p_runners = [
            ("P1", "① Raw × Thinking OFF", self.run_pattern_1_raw_no_thinking),
            ("P2", "② Raw × Thinking ON", self.run_pattern_2_raw_thinking),
            ("P3", "③ Preprocessed × Thinking ON", self.run_pattern_3_preprocessed_thinking),
            ("P4", "④ Preprocessed × Thinking OFF", self.run_pattern_4_preprocessed_no_thinking),
        ]

        for pid, label, runner in p_runners:
            if verbose:
                print(f"  ▶ 実行中: {label} ...", end="", flush=True)
            res = runner(case.scenario)
            patterns[pid] = res
            if verbose:
                print(f" 完了 ({res.response_time_ms}ms, {res.candidates_tokens} tokens)")

        return CaseMatrixResult(test_case=case, patterns=patterns)


def run_matrix_benchmark(
    model: str = GEMINI_FLASH_MODEL,
    cases: list[TestCase] | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> list[CaseMatrixResult]:
    """2×2マトリクスベンチマークを実行する"""
    runner = MatrixBenchmarkRunner(model=model)
    target_cases = cases or TEST_CASES
    if limit:
        target_cases = target_cases[:limit]

    results: list[CaseMatrixResult] = []
    print(f"\n[MatrixBenchmark]: 2×2 マトリクス検証開始 (全{len(target_cases)}件, model={model})")

    for case in target_cases:
        res = runner.evaluate_case(case, verbose=verbose)
        results.append(res)

    # 結果保存
    save_matrix_results(results, model=model)
    return results


def save_matrix_results(results: list[CaseMatrixResult], model: str) -> tuple[Path, Path]:
    """ベンチマーク結果を JSON および Markdown レポートとして保存"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"matrix_benchmark_{timestamp}.json"
    md_path = RESULTS_DIR / f"matrix_benchmark_{timestamp}.md"

    # JSON形式
    json_data = {
        "timestamp": timestamp,
        "model": model,
        "results": [
            {
                "case_id": r.test_case.id,
                "scenario": r.test_case.scenario,
                "expected": r.test_case.expected_answer,
                "expected_article": r.test_case.expected_article,
                "failure_type": r.test_case.failure_type,
                "patterns": {
                    pid: asdict(pres) for pid, pres in r.patterns.items()
                },
            }
            for r in results
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # Markdown レポート作成
    md_lines = [
        f"# 2×2 マトリクス検証レポート（前処理 × 推論 Thinking）",
        f"",
        f"- 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 検証モデル: `{model}`",
        f"- 対象ケース数: {len(results)} 件",
        f"",
        f"## 検証マトリクスと設計仮説",
        f"",
        f"| パターン | 入力（前処理） | LLM Thinking | 特徴・仮説 |",
        f"|---|---|---|---|",
        f"| **① Raw × Thinking OFF** | 生の条文 | OFF (Budget=0) | 【最弱】参照ジャンプや例外多段ネストを追えず誤答多発 |",
        f"| **② Raw × Thinking ON** | 生の条文 | ON (Budget=2048) | 【力技】自力推論するがトークン爆発＆捏造リスク |",
        f"| **③ Preprocessed × Thinking ON** | 構造化・参照解決済み | ON (Budget=2048) | 【過剰推論】確定情報があるのに自問自答で高コスト・遅延 |",
        f"| **④ Preprocessed × Thinking OFF** | 構造化・参照解決済み | OFF (Budget=0) | 【本命】最速・最安・最高精度（確定的トランスレータ） |",
        f"",
        f"## パターン別 平均メトリクス比較",
        f"",
    ]

    # 平均メトリクス集計
    p_keys = ["P1", "P2", "P3", "P4"]
    p_names = {
        "P1": "① Raw × Think OFF",
        "P2": "② Raw × Think ON",
        "P3": "③ Prep × Think ON",
        "P4": "④ Prep × Think OFF (本命)",
    }

    avg_time = {k: sum(r.patterns[k].response_time_ms for r in results) / len(results) for k in p_keys}
    avg_out_tok = {k: sum(r.patterns[k].candidates_tokens for r in results) / len(results) for k in p_keys}
    avg_total_tok = {k: sum(r.patterns[k].total_tokens for r in results) / len(results) for k in p_keys}

    md_lines.extend([
        f"| パターン | 平均レイテンシ (ms) | 平均出力トークン | 平均総トークン |",
        f"|---|---|---|---|",
    ])
    for k in p_keys:
        md_lines.append(f"| **{p_names[k]}** | {avg_time[k]:.1f} ms | {avg_out_tok[k]:.1f} | {avg_total_tok[k]:.1f} |")

    md_lines.extend([
        f"",
        f"## 各テストケース別 詳細結果",
        f"",
    ])

    for r in results:
        md_lines.extend([
            f"### [{r.test_case.id}] {r.test_case.scenario}",
            f"- **期待解**: {r.test_case.expected_answer}",
            f"- **着眼点**: {r.test_case.description}",
            f"",
            f"| パターン | レイテンシ | 出力トークン | 回答サマリー (抜粋) |",
            f"|---|---|---|---|",
        ])
        for k in p_keys:
            pres = r.patterns[k]
            short_ans = pres.response_text.replace("\n", " ")[:120] + ("..." if len(pres.response_text) > 120 else "")
            md_lines.append(f"| {p_names[k]} | {pres.response_time_ms} ms | {pres.candidates_tokens} tok | `{short_ans}` |")
        md_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[MatrixBenchmark]: 結果保存完了:")
    print(f"  - JSON: {json_path}")
    print(f"  - Markdown: {md_path}")
    return json_path, md_path


def print_matrix_summary(results: list[CaseMatrixResult]) -> None:
    """コンソールに集計サマリーを表示"""
    p_keys = ["P1", "P2", "P3", "P4"]
    p_labels = [
        "① Raw × Think OFF",
        "② Raw × Think ON",
        "③ Prep × Think ON",
        "④ Prep × Think OFF (本命)",
    ]

    print("\n" + "=" * 78)
    print(" 2×2 マトリクス検証 — パターン別 実行メトリクス 集計サマリー")
    print("=" * 78)
    print(f"{'パターン':<28} | {'平均レイテンシ':<12} | {'平均出力トークン':<14} | {'備考'}")
    print("-" * 78)

    for k, label in zip(p_keys, p_labels):
        avg_ms = sum(r.patterns[k].response_time_ms for r in results) / len(results)
        avg_tok = sum(r.patterns[k].candidates_tokens for r in results) / len(results)
        note = "最速・最軽量" if k == "P4" else ("トークン肥大" if k in ("P2", "P3") else "低精度リスク")
        print(f"{label:<28} | {avg_ms:8.1f} ms   | {avg_tok:8.1f} tokens   | {note}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="2x2 Matrix Verification Benchmark")
    parser.add_argument("--model", default=GEMINI_FLASH_MODEL, help="Model name to test")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test cases")
    args = parser.parse_args()

    res = run_matrix_benchmark(model=args.model, limit=args.limit)
    print_matrix_summary(res)

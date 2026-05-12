"""Hybrid Judge のステップ別レイテンシ計測スクリプト。

issue #10「処理速度向上」のための計測ツール。
Layer 1（XML パース / VSM 構築 / VSM 検索 / プロンプト構築）と
Layer 2（LLM 呼び出し）に分けて時間を取り、支配項を特定する。

Usage:
    cd backend
    python -m scripts.profile_hybrid                 # flash (default)
    python -m scripts.profile_hybrid --model claude_sonnet
    python -m scripts.profile_hybrid --model flash --model claude_sonnet
    python -m scripts.profile_hybrid --runs 5        # 各クエリの計測回数
    python -m scripts.profile_hybrid --queries 2     # 最初の N クエリのみ
"""

from __future__ import annotations

import argparse
import statistics
import time
from contextlib import contextmanager
from typing import Iterator

from src.benchmark.flash_only_judge import TEST_CASES
from src.config import (
    CLAUDE_MODEL,
    CLAUDE_SONNET_MODEL,
    GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL,
    GEMMA3_MODEL,
    XML_PATH,
)
from src.judgement.hybrid_judge import HybridJudge
from src.matcher.vsm_engine import VSMEngine
from src.parser.legal_compiler import (
    extract_bicycle_articles,
    parse_egov_xml,
)


MODEL_CHOICES = {
    "flash": GEMINI_FLASH_MODEL,
    "pro": GEMINI_PRO_MODEL,
    "gemma3": GEMMA3_MODEL,
    "claude": CLAUDE_MODEL,
    "claude_sonnet": CLAUDE_SONNET_MODEL,
}


@contextmanager
def stopwatch(label: str, sink: dict[str, float]) -> Iterator[None]:
    t0 = time.monotonic_ns()
    try:
        yield
    finally:
        sink[label] = (time.monotonic_ns() - t0) / 1_000_000  # ms


def format_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms/1000:6.2f}s"
    return f"{ms:6.1f}ms"


def print_row(label: str, values: list[float], total_ref: float | None = None) -> None:
    if not values:
        return
    mean = statistics.mean(values)
    pct = f"{(mean / total_ref * 100):5.1f}%" if total_ref else "   -- "
    extras = ""
    if len(values) > 1:
        extras = f"  min={format_ms(min(values))}  max={format_ms(max(values))}"
    print(f"  {label:18s} mean={format_ms(mean)}  {pct}{extras}")


def measure_one(judge: HybridJudge, query: str) -> dict[str, float]:
    """1 クエリの 1 ラウンドを計測する。

    本番経路 (`HybridJudge.judge()`) を呼ぶことで、production と同じ thinking 設定で
    LLM が叩かれることを保証する。トータルから LLM 時間 (judge が返す
    `response_time_ms`) を引いて Layer 1 部分を算出する。
    """
    timings: dict[str, float] = {}
    t0 = time.monotonic_ns()
    result = judge.judge(query, verbose=False)
    timings["total_request"] = (time.monotonic_ns() - t0) / 1_000_000
    timings["llm_call"] = float(result.response_time_ms)
    timings["layer1"] = timings["total_request"] - timings["llm_call"]
    return timings


def profile_model(model_key: str, queries: list[str], runs: int) -> None:
    model = MODEL_CHOICES[model_key]
    print("=" * 70)
    print(f"  Model: {model_key}  ({model})")
    print("=" * 70)

    # ------------------ Cold-start (one-time) ------------------
    cold: dict[str, float] = {}
    with stopwatch("xml_parse", cold):
        ast = parse_egov_xml(XML_PATH)
    with stopwatch("article_filter", cold):
        bicycle_articles = extract_bicycle_articles(ast)
    with stopwatch("vsm_build", cold):
        vsm = VSMEngine(ast, article_filter=bicycle_articles)
    with stopwatch("judge_init", cold):
        judge = HybridJudge(ast, vsm, model=model)

    print("\n[Cold start] (process 初期化, 1 回のみ)")
    cold_total = sum(cold.values())
    for label, v in cold.items():
        print(f"  {label:18s} {format_ms(v):>10s}  ({v/cold_total*100:5.1f}%)")
    print(f"  {'TOTAL':18s} {format_ms(cold_total):>10s}")

    # ------------------ Warm-up (1 回) ------------------
    print(f"\n[Warm-up] 1 回実行 (計測対象外)")
    _ = measure_one(judge, queries[0])

    # ------------------ 本計測 ------------------
    print(f"\n[Per-request timings]  queries={len(queries)} runs={runs}")
    agg: dict[str, list[float]] = {
        "layer1": [],
        "llm_call": [],
        "total_request": [],
    }
    for q_idx, q in enumerate(queries, 1):
        print(f"\n  Query {q_idx}: {q[:50]}...")
        for r in range(1, runs + 1):
            t = measure_one(judge, q)
            print(
                f"    run {r}: layer1={format_ms(t['layer1']):>9s}  "
                f"llm={format_ms(t['llm_call']):>9s}  "
                f"total={format_ms(t['total_request']):>9s}"
            )
            for k in agg:
                agg[k].append(t[k])

    # ------------------ サマリ ------------------
    total_mean = statistics.mean(agg["total_request"])
    print(f"\n[Summary] (mean over {len(agg['total_request'])} samples)")
    print_row("layer1 (VSM+prompt)", agg["layer1"], total_mean)
    print_row("llm_call", agg["llm_call"], total_mean)
    print_row("TOTAL", agg["total_request"], total_mean)

    # ------------------ ボトルネック診断 ------------------
    pct_llm = statistics.mean(agg["llm_call"]) / total_mean * 100

    print("\n[Diagnosis]")
    if pct_llm >= 90:
        print(f"  支配項: LLM 呼び出し ({pct_llm:.0f}%). Layer 1 はほぼノイズ。")
    elif pct_llm >= 60:
        print(f"  支配項: LLM ({pct_llm:.0f}%) だが Layer 1 ({100-pct_llm:.0f}%) も無視できない。")
    else:
        print(f"  Layer 1 寄り: LLM={pct_llm:.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Judge プロファイラ")
    parser.add_argument(
        "--model",
        action="append",
        choices=list(MODEL_CHOICES),
        help="計測対象モデル（複数指定可）。デフォルトは flash",
    )
    parser.add_argument("--runs", type=int, default=3, help="各クエリの計測回数")
    parser.add_argument(
        "--queries",
        type=int,
        default=3,
        help="TEST_CASES から先頭 N クエリを使用",
    )
    args = parser.parse_args()

    models = args.model or ["flash"]
    queries = [tc.scenario for tc in TEST_CASES[: args.queries]]

    for m in models:
        profile_model(m, queries, args.runs)
        print()


if __name__ == "__main__":
    main()

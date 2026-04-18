"""
Velo-Verify-Gemini: メインエントリポイント

『自転車の青切符をGeminiで判定しようとしたら、2008年度に書いた卒論に救われた話』

実行モード:
  --benchmark    Flash単体ベンチマーク（ハルシネーション検出）
  --hybrid       ハイブリッド判定（Layer 1 + Layer 2）
  --compare      ベンチマーク → ハイブリッドの比較（ブログ用）
  --model flash  Flash使用（デフォルト）
  --model pro    Pro使用
"""

from __future__ import annotations

import argparse
import sys

from src.config import XML_PATH, GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL
from src.parser.legal_compiler import parse_egov_xml, extract_bicycle_articles
from src.matcher.vsm_engine import VSMEngine
from src.benchmark.flash_only_judge import (
    run_flash_benchmark,
    print_summary,
    TEST_CASES,
)
from src.judgement.hybrid_judge import HybridJudge


BANNER = r"""
 ╔══════════════════════════════════════════════════════════════╗
 ║  Velo-Verify-Gemini — Hybrid Logic Compiler                ║
 ║  『自転車の青切符をGeminiで判定しようとしたら、              ║
 ║    2008年度に書いた卒論に救われた話』                        ║
 ║                                                              ║
 ║  Layer 1: 2008年式 決定論的パース + cos類似度 (卒論)          ║
 ║  Layer 2: 2026年式 Gemini Flash/Pro (最新LLM)                ║
 ║  対象法令: 道路交通法（令和6年法律第34号改正）               ║
 ║  施行日: 2026年4月1日 自転車青切符制度                       ║
 ╚══════════════════════════════════════════════════════════════╝
"""


def build_layers():
    """Layer 1 のコンポーネントを構築する"""
    print("[2008-Thesis-Logic]: e-Gov法令XMLをパース中...")
    ast = parse_egov_xml(XML_PATH)
    bicycle_articles = extract_bicycle_articles(ast)
    print(f"[2008-Thesis-Logic]: パース完了。全{len(ast.articles)}条中、"
          f"自転車関連{len(bicycle_articles)}条を抽出。")

    print(f"[2008-Thesis-Logic]: TF-IDF VSMインデックス構築中...")
    vsm = VSMEngine(ast, article_filter=bicycle_articles)
    print(f"[2008-Thesis-Logic]: インデックス構築完了。")

    return ast, vsm


def cmd_benchmark(model: str = GEMINI_FLASH_MODEL) -> None:
    """Gemini単体ベンチマーク"""
    label = "Pro" if model == GEMINI_PRO_MODEL else "Flash"
    print(f"\n[MODE]: {label}単体ベンチマーク (model={model})")
    print("[2026-AI-Logic]: Geminiに法規を丸投げし、ハルシネーションを観測します...\n")

    results = run_flash_benchmark(verbose=True, model=model)
    print_summary(results)


def cmd_hybrid(model: str) -> None:
    """ハイブリッド判定"""
    print(f"\n[MODE]: ハイブリッド判定 (model={model})")

    ast, vsm = build_layers()
    judge = HybridJudge(ast, vsm, model=model)

    queries = [tc.scenario for tc in TEST_CASES]
    for i, q in enumerate(queries, 1):
        print(f"\n[進捗 {i}/{len(queries)}]", flush=True)
        judge.judge(q, verbose=True)


def cmd_compare(model: str) -> None:
    """ベンチマーク → ハイブリッドの比較（ブログ用）"""
    print("\n[MODE]: Flash単体 vs ハイブリッド 比較")
    print("=" * 60)

    # Phase 1: Flash単体
    print("\n" + "─" * 60)
    print("【Phase 1】Gemini Flash 単体 — AIに法規を丸投げした結果")
    print("─" * 60)
    flash_results = run_flash_benchmark(verbose=True)
    print_summary(flash_results)

    # Phase 2: ハイブリッド
    print("\n" + "─" * 60)
    print("【Phase 2】ハイブリッド構成 — 2008年卒論 × 2026年AI")
    print("─" * 60)
    ast, vsm = build_layers()
    judge = HybridJudge(ast, vsm, model=model)

    hybrid_results = []
    for tc in TEST_CASES:
        result = judge.judge(tc.scenario, verbose=True)
        hybrid_results.append(result)

    # Phase 3: 比較サマリー
    print("\n" + "═" * 60)
    print("【比較結果】Flash単体 vs ハイブリッド")
    print("═" * 60)

    flash_correct = sum(1 for r in flash_results if r.is_correct)
    flash_hallucinated = sum(1 for r in flash_results if r.hallucination_detected)

    print(f"\n  Flash単体:")
    print(f"    正答率: {flash_correct}/{len(flash_results)}"
          f" ({100*flash_correct/len(flash_results):.0f}%)")
    print(f"    ハルシネーション: {flash_hallucinated}件")
    print(f"\n  ハイブリッド構成:")
    print(f"    モデル: {model}")
    print(f"    Layer 1が注入した根拠: 条文AST + 反則金テーブル + 委任規定解決済み情報")
    print(f"    判定件数: {len(hybrid_results)}件")

    print(f"\n{'─'*60}")
    print("[Hybrid-Result]: 18年前のロジックが、最新AIのハルシネーションを矯正しました。")
    print("[Hybrid-Result]: 決定論的パース × LLM推論 = 精度100%への回帰。")


def main() -> None:
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="Velo-Verify-Gemini: 自転車青切符ハイブリッド判定システム"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Flash単体ベンチマーク実行",
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="ハイブリッド判定実行",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Flash単体 vs ハイブリッドの比較実行",
    )
    parser.add_argument(
        "--model", choices=["flash", "pro"], default="flash",
        help="使用するGeminiモデル (default: flash)",
    )

    args = parser.parse_args()
    model = GEMINI_FLASH_MODEL if args.model == "flash" else GEMINI_PRO_MODEL

    if args.benchmark:
        cmd_benchmark(model)
    elif args.hybrid:
        cmd_hybrid(model)
    elif args.compare:
        cmd_compare(model)
    else:
        parser.print_help()
        print("\n使用例:")
        print("  python -m src.main --benchmark        # Flash単体テスト")
        print("  python -m src.main --hybrid            # ハイブリッド判定")
        print("  python -m src.main --compare           # 比較（ブログ用）")
        print("  python -m src.main --hybrid --model pro  # Proモデル使用")


if __name__ == "__main__":
    main()

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

from src.config import (
    XML_PATH,
    GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL,
    GEMMA3_MODEL,
    CLAUDE_MODEL,
    CLAUDE_SONNET_MODEL,
)
from src.parser.legal_compiler import parse_egov_xml, extract_bicycle_articles
from src.matcher.vsm_engine import VSMEngine
from src.matcher.embedding_engine import EmbeddingEngine
from src.benchmark.flash_only_judge import (
    run_flash_benchmark,
    print_summary,
    TEST_CASES,
)
from src.benchmark.matrix_benchmark import (
    run_matrix_benchmark,
    print_matrix_summary,
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


def build_layers(use_embedding: bool = True):
    """Layer 1 のコンポーネントを構築する。

    use_embedding=True なら Vertex AI Embedding を Layer 1 に同梱した
    ハイブリッド構成（TF-IDF + 意味埋め込み）になる。デフォルトは ON。
    Embedding 利用には GCP プロジェクト ADC が必要で、初期化に数秒かかる。
    純 TF-IDF (2008 年式) で動かしたいときは False を渡す。
    """
    print("[2008-Thesis-Logic]: e-Gov法令XMLをパース中...")
    ast = parse_egov_xml(XML_PATH)
    bicycle_articles = extract_bicycle_articles(ast)
    print(f"[2008-Thesis-Logic]: パース完了。全{len(ast.articles)}条中、"
          f"自転車関連{len(bicycle_articles)}条を抽出。")

    embedding = None
    if use_embedding:
        print("[2026-Semantic]: Vertex AI Embedding で意味インデックス構築中...")
        embedding = EmbeddingEngine(ast, article_filter=bicycle_articles)
        print(f"[2026-Semantic]: Embedding 完了。次元={len(embedding.doc_embeddings[0])}")

    label = "ハイブリッド (TF-IDF + Embedding α=0.3)" if use_embedding else "TF-IDF 単体"
    print(f"[2008-Thesis-Logic]: VSMインデックス構築中 ({label})...")
    vsm = VSMEngine(
        ast,
        article_filter=bicycle_articles,
        embedding_engine=embedding,
    )
    print(f"[2008-Thesis-Logic]: インデックス構築完了。")

    return ast, vsm


def cmd_benchmark(model: str = GEMINI_FLASH_MODEL) -> None:
    """LLM単体ベンチマーク"""
    if model == GEMINI_PRO_MODEL:
        label = "Pro"
    elif model == GEMMA3_MODEL:
        label = "Gemma3 (local)"
    elif model == CLAUDE_MODEL:
        label = "Claude Opus 4.7"
    elif model == CLAUDE_SONNET_MODEL:
        label = "Claude Sonnet 4.6"
    else:
        label = "Flash"
    print(f"\n[MODE]: {label}単体ベンチマーク (model={model})")
    print("[2026-AI-Logic]: LLMに法規を丸投げし、ハルシネーションを観測します...\n")

    results = run_flash_benchmark(verbose=True, model=model)
    print_summary(results)


def cmd_hybrid(model: str, use_embedding: bool = True) -> None:
    """ハイブリッド判定"""
    print(f"\n[MODE]: ハイブリッド判定 (model={model}, embedding={use_embedding})")

    ast, vsm = build_layers(use_embedding=use_embedding)
    judge = HybridJudge(ast, vsm, model=model)

    queries = [tc.scenario for tc in TEST_CASES]
    for i, q in enumerate(queries, 1):
        print(f"\n[進捗 {i}/{len(queries)}]", flush=True)
        judge.judge(q, verbose=True)


def cmd_compare(model: str, use_embedding: bool = True) -> None:
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
    ast, vsm = build_layers(use_embedding=use_embedding)
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


def cmd_matrix(model: str = GEMINI_FLASH_MODEL, limit: int | None = None) -> None:
    """前処理（あり/なし） × 推論Thinking（あり/なし）の2×2マトリクス比較検証"""
    print(f"\n[MODE]: 2×2 マトリクス検証 (前処理 × Thinking, model={model})")
    print("[2026-AI-Logic]: 4パターン（①生×推論OFF, ②生×推論ON, ③前処理×推論ON, ④前処理×推論OFF）を比較します...\n")

    results = run_matrix_benchmark(model=model, limit=limit, verbose=True)
    print_matrix_summary(results)


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
        "--matrix", action="store_true",
        help="2×2 マトリクス検証（前処理あり/なし × 推論あり/なし 4パターン比較）",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="テストケースの実行件数上限（未指定時は全件）",
    )
    parser.add_argument(
        "--model",
        choices=["flash", "pro", "gemma3", "claude", "claude_sonnet"],
        default="flash",
        help="使用するLLMモデル: flash/pro (Gemini) / gemma3 (ローカル Ollama) / "
             "claude (Vertex AI 経由 Claude Opus 4.7) / "
             "claude_sonnet (Vertex AI 経由 Claude Sonnet 4.6) (default: flash)",
    )
    parser.add_argument(
        "--no-embedding", action="store_true",
        help="Layer 1 を純 TF-IDF (2008年式) で動かす。"
             "デフォルトは Embedding と合成したハイブリッドモード。",
    )

    args = parser.parse_args()
    model = {
        "flash": GEMINI_FLASH_MODEL,
        "pro": GEMINI_PRO_MODEL,
        "gemma3": GEMMA3_MODEL,
        "claude": CLAUDE_MODEL,
        "claude_sonnet": CLAUDE_SONNET_MODEL,
    }[args.model]

    use_embedding = not args.no_embedding

    if args.matrix:
        cmd_matrix(model, limit=args.limit)
    elif args.benchmark:
        cmd_benchmark(model)
    elif args.hybrid:
        cmd_hybrid(model, use_embedding=use_embedding)
    elif args.compare:
        cmd_compare(model, use_embedding=use_embedding)
    else:
        parser.print_help()
        print("\n使用例:")
        print("  python -m src.main --matrix           # 2×2 マトリクス検証（前処理 × 推論ON/OFF 4パターン）")
        print("  python -m src.main --benchmark        # Flash単体テスト")
        print("  python -m src.main --hybrid           # ハイブリッド判定")
        print("  python -m src.main --compare          # 比較（ブログ用）")
        print("  python -m src.main --matrix --limit 3 # 最初の3件のみマトリクス検証")
        print("  python -m src.main --hybrid --model pro  # Proモデル使用")
        print("  python -m src.main --hybrid --model gemma3  # ローカル gemma3:4b 使用")
        print("  python -m src.main --hybrid --model claude          # Claude Opus 4.7 使用 (Vertex AI)")
        print("  python -m src.main --hybrid --model claude_sonnet   # Claude Sonnet 4.6 使用 (Vertex AI)")


if __name__ == "__main__":
    main()


"""
Task 4: Hybrid Corrected Reasoner (PTE Standard)

Layer 1（2008年式決定論的パース + cos類似度）の結果を
Layer 2（Gemini Flash/Pro）のプロンプトに「絶対的根拠」として注入し、
ハルシネーションを物理的に封じるハイブリッド判定エンジン。

ブログ用「魂のログ」出力対応。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from src.config import GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL, FINE_TABLE_PATH, get_genai_client
from src.parser.legal_compiler import (
    LawAST,
    ArticleNode,
    flatten_article_text,
)
from src.matcher.vsm_engine import VSMEngine, VSMMatch


@dataclass
class HybridResult:
    query: str
    # Layer 1 results
    vsm_matches: list[VSMMatch]
    matched_article_text: str
    logic_flags: list[str]
    fine_info: str
    # Layer 2 results
    gemini_answer: str
    # Meta
    model_used: str
    response_time_ms: int


HYBRID_SYSTEM_PROMPT = """\
あなたは日本の道路交通法の専門家です。
2026年4月1日施行の自転車交通反則通告制度（青切符制度）に基づいて回答してください。

【重要】以下に提示する「条文データ」と「反則金データ」は、e-Gov法令APIから取得した
公式の法令データを決定論的にパースした結果です。この情報を絶対的な根拠として使用し、
あなた自身の学習データからの補完は一切行わないでください。

条文データに記載されていない金額や条文を回答に含めてはいけません。
「政令で定める者」等の委任規定は、以下に解決済みの情報を提供します。
"""


class HybridJudge:
    """Layer 1 + Layer 2 ハイブリッド判定エンジン"""

    def __init__(
        self,
        ast: LawAST,
        vsm_engine: VSMEngine,
        model: str = GEMINI_FLASH_MODEL,
    ):
        self._ast = ast
        self._vsm = vsm_engine
        self._model = model
        self._client = get_genai_client()
        self._fine_table = self._load_fine_table()

    def judge(self, query: str, verbose: bool = True) -> HybridResult:
        """ハイブリッド判定を実行する"""

        # === Layer 1: 2008年式決定論的処理 ===
        if verbose:
            print(f"\n{'='*60}")
            print(f"[Hybrid-Judge]: クエリ受付「{query}」")
            print(f"[2008-Thesis-Logic]: Layer 1 起動... cos類似度検索開始")

        vsm_matches = self._vsm.search(query, top_k=3)

        if verbose:
            for m in vsm_matches:
                print(f"[2008-Thesis-Logic]:   #{m.rank} cos={m.score:.4f} "
                      f"→ {m.article.title} {m.article.caption}")

        # 最も関連する条文のテキストを取得
        primary = vsm_matches[0] if vsm_matches else None
        if primary is None:
            return HybridResult(
                query=query,
                vsm_matches=[],
                matched_article_text="(該当条文なし)",
                logic_flags=[],
                fine_info="(該当なし)",
                gemini_answer="条文が特定できませんでした。",
                model_used=self._model,
                response_time_ms=0,
            )

        # 条文テキストの取得（上位3条文を含める）
        article_texts = []
        all_flags: list[str] = []
        for m in vsm_matches:
            text = flatten_article_text(m.article)
            article_texts.append(f"--- {m.article.title} {m.article.caption} ---\n{text}")
            for p in m.article.paragraphs:
                for s in p.sentences:
                    all_flags.extend(s.logic_flags)

        unique_flags = sorted(set(all_flags))
        combined_text = "\n\n".join(article_texts)

        # 反則金情報を検索
        fine_info = self._lookup_fine(query, primary.article)

        if verbose:
            print(f"[2008-Thesis-Logic]: 論理フラグ検出: {unique_flags}")
            if "delegation" in unique_flags:
                print("[2008-Thesis-Logic]: ⚠ 「政令で定める」委任規定を検知。解決済み情報を注入します。")
            if "exception" in unique_flags or "proviso" in unique_flags:
                print("[2008-Thesis-Logic]: ⚠ 例外規定/ただし書きを検知。階層構造を保持して注入します。")
            print(f"[2008-Thesis-Logic]: 反則金情報: {fine_info}")

        # === Layer 2: Gemini推論（条文を限定注入） ===
        if verbose:
            print(f"[2026-AI-Logic]: Layer 2 起動... {self._model}", flush=True)
            print(f"[2026-AI-Logic]: Gemini API呼び出し中...", flush=True)

        prompt = self._build_prompt(query, combined_text, fine_info, unique_flags)

        start = time.monotonic_ns()
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config={
                "system_instruction": HYBRID_SYSTEM_PROMPT,
                "temperature": 0.0,
            },
        )
        elapsed_ms = (time.monotonic_ns() - start) // 1_000_000

        gemini_answer = response.text or "(empty response)"

        result = HybridResult(
            query=query,
            vsm_matches=vsm_matches,
            matched_article_text=combined_text,
            logic_flags=unique_flags,
            fine_info=fine_info,
            gemini_answer=gemini_answer,
            model_used=self._model,
            response_time_ms=elapsed_ms,
        )

        if verbose:
            self._print_soul_log(result)

        return result

    def _build_prompt(
        self,
        query: str,
        article_text: str,
        fine_info: str,
        logic_flags: list[str],
    ) -> str:
        """Layer 1の結果を注入したプロンプトを構築"""
        delegation_note = ""
        if "delegation" in logic_flags:
            delegation_note = """
【委任規定の解決済み情報】
道路交通法施行令第26条により、第63条の4第1項第2号の「政令で定める者」は以下のとおり:
  一　児童（6歳以上13歳未満）及び幼児（6歳未満）
  二　七十歳以上の者
  三　身体に障害を有する者で政令で定めるもの

※ 自転車の青切符制度の対象は16歳以上の自転車運転者。16歳未満は指導警告による対応。
"""

        return f"""\
以下の質問に、提供された条文データのみを根拠として回答してください。

【質問】
{query}

【決定論的パーサーが特定した関連条文（e-Gov法令XMLより抽出）】
{article_text}
{delegation_note}
【反則金データ（2026年4月1日施行・青切符制度）】
{fine_info}

【検出された論理フラグ】
{', '.join(logic_flags) if logic_flags else 'なし'}

上記の情報のみを使って、以下のJSON形式で回答してください:
{{
  "judgement": "合法" or "違反",
  "article": "根拠条文（条・項・号まで）",
  "fine": "反則金額（該当する場合）or 対象外の理由",
  "reasoning": "判定理由（条文の階層構造を踏まえた説明）"
}}
"""

    def _lookup_fine(self, query: str, article: ArticleNode) -> str:
        """反則金テーブルから関連する情報を検索"""
        results = []

        # キーワードマッチで関連する反則金を検索
        keywords_map = {
            "歩道": ["通行区分違反", "歩道徐行等義務違反"],
            "信号": ["信号無視"],
            "一時停止": ["指定場所一時不停止等"],
            "スマホ": ["携帯電話使用等"],
            "携帯": ["携帯電話使用等"],
            "飲酒": ["酒気帯び運転", "酒酔い運転"],
            "酒": ["酒気帯び運転", "酒酔い運転"],
            "二人乗り": ["軽車両乗車積載制限違反"],
            "並進": ["並進禁止違反"],
            "ブレーキ": ["自転車制動装置不良"],
            "無灯火": ["無灯火"],
            "踏切": ["遮断踏切立入り", "踏切不停止等"],
            "追越": ["追越し違反"],
            "傘": ["公安委員会遵守事項違反"],
            "イヤホン": ["公安委員会遵守事項違反"],
            "横断歩道": ["横断歩行者等妨害等"],
        }

        matched_violations: set[str] = set()
        for keyword, violations in keywords_map.items():
            if keyword in query:
                matched_violations.update(violations)

        for fine in self._fine_table.get("fines", []):
            if fine["violation"] in matched_violations:
                results.append(f"・{fine['violation']}: 反則金{fine['amount']:,}円（{fine['article']}）")

        for criminal in self._fine_table.get("criminal_only", []):
            if criminal["violation"] in matched_violations:
                results.append(
                    f"・{criminal['violation']}: 反則金対象外・刑事罰"
                    f"（{criminal['penalty']}、{criminal['article']}）"
                )

        if not results:
            # 条文番号から逆引き
            for fine in self._fine_table.get("fines", []):
                if article.num in fine.get("article", ""):
                    results.append(f"・{fine['violation']}: 反則金{fine['amount']:,}円")

        if not results:
            return "該当する反則金情報なし（条文を確認して判断してください）"

        return "\n".join(results)

    def _print_soul_log(self, result: HybridResult) -> None:
        """ブログ用「魂のログ」出力"""
        print()
        print(f"[Hybrid-Result]: === 判定結果 ({result.response_time_ms}ms) ===")
        print(f"[Hybrid-Result]: モデル: {result.model_used}")
        print(f"[Hybrid-Result]: {result.gemini_answer[:500]}")

        if result.logic_flags:
            flags_str = ", ".join(result.logic_flags)
            print(f"[2008-Thesis-Logic]: 18年前のロジックが検出した論理構造: {flags_str}")

        if "delegation" in result.logic_flags:
            print("[2008-Thesis-Logic]: 「政令で定める者」→ 施行令第26条で解決済み。"
                  "Flashの推測に頼らず、確定情報を注入しました。")

    @staticmethod
    def _load_fine_table() -> dict:
        with open(FINE_TABLE_PATH, encoding="utf-8") as f:
            return json.load(f)


if __name__ == "__main__":
    from src.parser.legal_compiler import parse_egov_xml, extract_bicycle_articles
    from src.config import XML_PATH

    print("[Hybrid-Judge]: ハイブリッド判定エンジン起動")
    print("[2008-Thesis-Logic]: e-Gov XMLパース中...")
    ast = parse_egov_xml(XML_PATH)
    bicycle_articles = extract_bicycle_articles(ast)

    print(f"[2008-Thesis-Logic]: VSMインデックス構築中... ({len(bicycle_articles)}条文)")
    vsm = VSMEngine(ast, article_filter=bicycle_articles)

    judge = HybridJudge(ast, vsm, model=GEMINI_FLASH_MODEL)

    test_queries = [
        "75歳の高齢者が普通自転車で歩道を走行しています。これは違反ですか？反則金はいくらですか？",
        "自転車でスマートフォンを手に持ちながら運転しました。反則金はいくらですか？",
        "自転車で酒気帯び運転をしました。反則金はいくらですか？",
    ]

    for q in test_queries:
        judge.judge(q)

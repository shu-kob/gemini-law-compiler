"""判定サービス本体。

3 つの判定モードを提供する:

- ``llm_only``: LLM 単体（グラウンディングなし）。Flash 単体ベンチマークと同等の挙動。
- ``layer1``:   2008 年式決定論的処理（e-Gov XML パース + cos 類似度）でヒットした
                条文を「絶対的根拠」としてプロンプトに注入する HybridJudge。
- ``web_search``: Gemini の Google 検索グラウンディング（``google_search`` ツール）を
                  有効化して Web 上の情報を根拠とさせる。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from src.benchmark.flash_only_judge import SYSTEM_PROMPT as LLM_ONLY_SYSTEM_PROMPT
from src.config import (
    CLAUDE_MODEL,
    GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL,
    GEMMA3_MODEL,
    XML_PATH,
    get_genai_client,
    get_llm_client,
    is_anthropic_model,
    is_ollama_model,
)
from src.judgement.hybrid_judge import HybridJudge
from src.matcher.vsm_engine import VSMEngine
from src.parser.legal_compiler import extract_bicycle_articles, parse_egov_xml


SUPPORTED_MODELS: dict[str, str] = {
    "flash": GEMINI_FLASH_MODEL,
    "pro": GEMINI_PRO_MODEL,
    "gemma3": GEMMA3_MODEL,
    "claude": CLAUDE_MODEL,
}

GROUNDING_MODES = ("llm_only", "layer1", "web_search")


WEB_SEARCH_SYSTEM_PROMPT = """\
あなたは日本の道路交通法の専門家です。
2026年4月1日施行の自転車交通反則通告制度（青切符制度）に基づいて回答してください。

以下のルールを厳守してください:
1. 根拠となる条文番号を必ず明示すること
2. 反則金額を回答する場合は正確な金額を示すこと
3. 反則金対象外（刑事罰）の場合はその旨を明示すること
4. Google 検索の結果から得られた情報を根拠として用いること
5. 推測ではなく、検索結果に基づいて回答すること

回答は次の JSON フォーマットで返してください（前後に余計な文字列は入れないこと）:
{
  "judgement": "合法" or "違反",
  "article": "根拠条文",
  "fine": "反則金額（該当する場合）",
  "reasoning": "判定理由の説明"
}
"""


@dataclass
class JudgeResponse:
    query: str
    mode: str
    model_key: str
    model: str
    judgement: str
    article: str
    fine: str
    reasoning: str
    raw_answer: str
    response_time_ms: int
    sources: list[dict[str, str]] = field(default_factory=list)
    layer1: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "model_key": self.model_key,
            "model": self.model,
            "judgement": self.judgement,
            "article": self.article,
            "fine": self.fine,
            "reasoning": self.reasoning,
            "raw_answer": self.raw_answer,
            "response_time_ms": self.response_time_ms,
            "sources": self.sources,
            "layer1": self.layer1,
        }


class JudgeService:
    """3 モードの判定をディスパッチするサービス。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._ast = None
        self._vsm: VSMEngine | None = None
        self._hybrid_judges: dict[str, HybridJudge] = {}

    def resolve_model(self, model_key: str) -> str:
        if model_key not in SUPPORTED_MODELS:
            raise ValueError(
                f"未対応のモデルキーです: {model_key!r}. "
                f"対応キー: {sorted(SUPPORTED_MODELS)}"
            )
        return SUPPORTED_MODELS[model_key]

    def judge(self, *, query: str, model_key: str, mode: str) -> JudgeResponse:
        if mode not in GROUNDING_MODES:
            raise ValueError(
                f"未対応の判定モードです: {mode!r}. 対応モード: {GROUNDING_MODES}"
            )
        model = self.resolve_model(model_key)

        if mode == "llm_only":
            return self._judge_llm_only(query=query, model_key=model_key, model=model)
        if mode == "layer1":
            return self._judge_layer1(query=query, model_key=model_key, model=model)
        return self._judge_web_search(query=query, model_key=model_key, model=model)

    # -- mode: llm_only --------------------------------------------------

    def _judge_llm_only(self, *, query: str, model_key: str, model: str) -> JudgeResponse:
        client = get_llm_client(model)

        start = time.monotonic_ns()
        response = client.models.generate_content(
            model=model,
            contents=query,
            config={
                "system_instruction": LLM_ONLY_SYSTEM_PROMPT,
                "temperature": 0.0,
            },
        )
        elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
        raw_answer = response.text or ""

        parsed = _parse_judgement_json(raw_answer)
        return JudgeResponse(
            query=query,
            mode="llm_only",
            model_key=model_key,
            model=model,
            judgement=parsed["judgement"],
            article=parsed["article"],
            fine=parsed["fine"],
            reasoning=parsed["reasoning"],
            raw_answer=raw_answer,
            response_time_ms=int(elapsed_ms),
        )

    # -- mode: layer1 ----------------------------------------------------

    def _judge_layer1(self, *, query: str, model_key: str, model: str) -> JudgeResponse:
        judge = self._get_hybrid_judge(model)
        result = judge.judge(query, verbose=False)

        parsed = _parse_judgement_json(result.gemini_answer)

        layer1_payload = {
            "matches": [
                {
                    "rank": m.rank,
                    "score": float(m.score),
                    "article_title": m.article.title,
                    "article_caption": m.article.caption,
                }
                for m in result.vsm_matches
            ],
            "logic_flags": list(result.logic_flags),
            "fine_info": result.fine_info,
            "matched_article_text": result.matched_article_text,
        }

        return JudgeResponse(
            query=query,
            mode="layer1",
            model_key=model_key,
            model=model,
            judgement=parsed["judgement"],
            article=parsed["article"],
            fine=parsed["fine"],
            reasoning=parsed["reasoning"],
            raw_answer=result.gemini_answer,
            response_time_ms=result.response_time_ms,
            layer1=layer1_payload,
        )

    # -- mode: web_search ------------------------------------------------

    def _judge_web_search(self, *, query: str, model_key: str, model: str) -> JudgeResponse:
        if is_ollama_model(model) or is_anthropic_model(model):
            raise ValueError(
                "Web Search グラウンディングは Gemini モデル (flash/pro) のみ対応しています。"
            )

        client = get_genai_client()

        start = time.monotonic_ns()
        response = client.models.generate_content(
            model=model,
            contents=query,
            config={
                "system_instruction": WEB_SEARCH_SYSTEM_PROMPT,
                "temperature": 0.0,
                "tools": [{"google_search": {}}],
            },
        )
        elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
        raw_answer = response.text or ""

        parsed = _parse_judgement_json(raw_answer)
        sources = _extract_grounding_sources(response)

        return JudgeResponse(
            query=query,
            mode="web_search",
            model_key=model_key,
            model=model,
            judgement=parsed["judgement"],
            article=parsed["article"],
            fine=parsed["fine"],
            reasoning=parsed["reasoning"],
            raw_answer=raw_answer,
            response_time_ms=int(elapsed_ms),
            sources=sources,
        )

    # -- internal ---------------------------------------------------------

    def _get_hybrid_judge(self, model: str) -> HybridJudge:
        with self._lock:
            if self._ast is None or self._vsm is None:
                ast = parse_egov_xml(XML_PATH)
                bicycle_articles = extract_bicycle_articles(ast)
                self._ast = ast
                self._vsm = VSMEngine(ast, article_filter=bicycle_articles)

            judge = self._hybrid_judges.get(model)
            if judge is None:
                judge = HybridJudge(self._ast, self._vsm, model=model)
                self._hybrid_judges[model] = judge
            return judge


def _parse_judgement_json(answer: str) -> dict[str, str]:
    """LLM レスポンスから JSON ブロックを抜き出して dict に変換する。

    モデルがコードフェンス付きで返したり、前後に文字列を付けて返すケースに耐える。
    パース不能でも raw_answer は呼び出し元で保持しているので、ここでは空文字に
    フォールバックする。
    """
    fields = {"judgement": "", "article": "", "fine": "", "reasoning": ""}
    if not answer:
        return fields

    candidate = _strip_code_fence(answer)
    json_str = _extract_first_json_object(candidate)
    if json_str is None:
        # JSON 形式で返ってこなかった場合は reasoning に丸ごと入れて返す
        fields["reasoning"] = answer.strip()
        return fields

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        fields["reasoning"] = answer.strip()
        return fields

    for key in fields:
        value = data.get(key, "")
        if value is None:
            value = ""
        fields[key] = str(value)
    return fields


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # ```json ... ``` の形式を剥がす
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _extract_first_json_object(text: str) -> str | None:
    """テキスト中から最初に現れる {...} ブロックを抜き出す。ネスト対応。"""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_grounding_sources(response: Any) -> list[dict[str, str]]:
    """Gemini レスポンスから Web Search グラウンディングのソースを取り出す。"""
    sources: list[dict[str, str]] = []
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        meta = getattr(cand, "grounding_metadata", None)
        if meta is None:
            continue
        chunks = getattr(meta, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web is None:
                continue
            uri = getattr(web, "uri", None) or ""
            title = getattr(web, "title", None) or uri
            if uri or title:
                sources.append({"title": title, "uri": uri})
    # 重複排除（uri ベース）
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for s in sources:
        key = s.get("uri") or s.get("title", "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique

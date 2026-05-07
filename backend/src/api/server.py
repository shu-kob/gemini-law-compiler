"""FastAPI エントリポイント。

Next.js フロントエンドからの POST /api/judge を受け、JudgeService にディスパッチする。
ローカル開発用に CORS を ``http://localhost:3000`` 向けに開けている。
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api.judge_service import (
    GROUNDING_MODES,
    SUPPORTED_MODELS,
    JudgeService,
)


app = FastAPI(
    title="Velo-Verify-Gemini API",
    description="自転車青切符ハイブリッド判定 API",
    version="0.1.0",
)

# CORS — 開発時は 3000 / 環境変数で上書き可能
_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
allowed_origins = [
    o.strip()
    for o in os.environ.get("API_ALLOWED_ORIGINS", _default_origins).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


_service = JudgeService()


class JudgeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    model: str = Field(default="flash", description="flash|pro|gemma3|claude")
    mode: str = Field(default="layer1", description="llm_only|layer1|web_search")


class ModelInfo(BaseModel):
    key: str
    model_id: str
    label: str
    supports_web_search: bool


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def list_models() -> dict[str, list[ModelInfo]]:
    labels = {
        "flash": "Gemini 3 Flash",
        "pro": "Gemini 3.1 Pro",
        "gemma3": "Gemma3 (Ollama / local)",
        "claude": "Claude on Vertex AI",
    }
    items = [
        ModelInfo(
            key=key,
            model_id=model_id,
            label=labels.get(key, key),
            supports_web_search=key in {"flash", "pro"},
        )
        for key, model_id in SUPPORTED_MODELS.items()
    ]
    return {"models": items}


@app.get("/api/modes")
def list_modes() -> dict[str, list[dict[str, str]]]:
    descriptions = {
        "llm_only": "LLM 単体（グラウンディングなし）",
        "layer1": "Layer 1 グラウンディング（決定論的パース + cos 類似度で条文注入）",
        "web_search": "Web Search グラウンディング（Gemini google_search ツール）",
    }
    return {
        "modes": [
            {"key": m, "label": descriptions[m]} for m in GROUNDING_MODES
        ]
    }


@app.post("/api/judge")
def judge(req: JudgeRequest) -> dict:
    import sys
    import time

    print(
        f"[judge] START model={req.model} mode={req.mode} query={req.query[:60]!r}",
        flush=True,
    )
    sys.stdout.flush()
    t0 = time.monotonic()
    try:
        result = _service.judge(query=req.query, model_key=req.model, mode=req.mode)
    except ValueError as e:
        print(f"[judge] BAD_REQUEST: {e}", flush=True)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        print(f"[judge] ERROR after {time.monotonic()-t0:.1f}s: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"判定中にエラーが発生しました: {e}") from e
    elapsed = time.monotonic() - t0
    print(
        f"[judge] DONE  ({elapsed:.1f}s) judgement={result.judgement!r} "
        f"article={result.article!r}",
        flush=True,
    )
    return result.to_dict()


def run() -> None:
    """`velo-verify-api` スクリプト用のエントリポイント。"""
    import uvicorn

    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("src.api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()

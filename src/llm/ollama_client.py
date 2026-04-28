"""Ollama HTTP クライアント（Gemini クライアント互換インタフェース）。

`google.genai.Client` の `client.models.generate_content(model, contents, config)`
と同じ呼び出し形で使えるよう、薄いダックタイプアダプタを提供する。
レスポンスオブジェクトは `.text` 属性を持つ。

ネットワーク依存を増やさないため urllib のみで実装。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class OllamaResponse:
    text: str


class _OllamaModels:
    def __init__(self, host: str) -> None:
        self._host = host.rstrip("/")

    def generate_content(
        self,
        model: str,
        contents: str,
        config: dict | None = None,
    ) -> OllamaResponse:
        config = config or {}
        body: dict = {
            "model": model,
            "prompt": contents,
            "stream": False,
            "options": {
                "temperature": float(config.get("temperature", 0.0)),
            },
        }
        system_instruction = config.get("system_instruction")
        if system_instruction:
            body["system"] = system_instruction

        req = urllib.request.Request(
            f"{self._host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama への接続に失敗しました ({self._host}/api/generate): {e}. "
                f"`ollama serve` が起動しているか確認してください。"
            ) from e

        return OllamaResponse(text=data.get("response", ""))


class OllamaClient:
    """Gemini クライアントと同じ呼び出し形を提供する Ollama ラッパー。"""

    def __init__(self, host: str = "http://localhost:11434") -> None:
        self.models = _OllamaModels(host)

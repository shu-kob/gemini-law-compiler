"""Anthropic Claude on Vertex AI クライアント（Gemini クライアント互換インタフェース）。

`google.genai.Client` の `client.models.generate_content(model, contents, config)`
と同じ呼び出し形で Vertex AI 経由の Claude を叩けるよう、薄いアダプタを提供する。
レスポンスオブジェクトは `.text` 属性を持つ（Gemini / Ollama アダプタと同じ）。

認証は Google Cloud ADC（`gcloud auth application-default login`）を前提とする。
プロジェクト ID とロケーションは Gemini と同じ環境変数（GOOGLE_CLOUD_PROJECT /
GOOGLE_CLOUD_LOCATION）を使い回す。

Opus 4.7 / Sonnet 4.6 では `temperature` / `top_p` / `top_k` が削除されているため、
config に渡された temperature は無視する（送ると 400）。

thinking は呼び出し側が `config["thinking"]` で明示した場合のみ有効化する。
Layer 1 グラウンディング済みのタスクでは thinking の効果が小さい一方でレイテンシが
3〜4 倍に膨らむため、デフォルトでは無効。詳細は issue #10。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnthropicResponse:
    text: str


# これらのモデルは temperature / top_p / top_k を受け付けないため除外する。
_NO_TEMPERATURE_MODELS = (
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
)


class _AnthropicVertexModels:
    def __init__(self, client) -> None:
        self._client = client

    def generate_content(
        self,
        model: str,
        contents: str,
        config: dict | None = None,
    ) -> AnthropicResponse:
        config = config or {}
        kwargs: dict = {
            "model": model,
            "max_tokens": int(config.get("max_tokens", 16000)),
            "messages": [{"role": "user", "content": contents}],
        }

        system_instruction = config.get("system_instruction")
        if system_instruction:
            kwargs["system"] = system_instruction

        if not model.startswith(_NO_TEMPERATURE_MODELS) and "temperature" in config:
            kwargs["temperature"] = float(config["temperature"])

        thinking_cfg = config.get("thinking")
        if thinking_cfg:
            kwargs["thinking"] = thinking_cfg

        response = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)

        return AnthropicResponse(text="".join(text_parts))


class AnthropicClient:
    """Vertex AI 経由 Claude を Gemini クライアント互換で呼び出すラッパー。"""

    def __init__(self, project_id: str, region: str = "global") -> None:
        try:
            from anthropic import AnthropicVertex
        except ImportError as e:
            raise RuntimeError(
                "anthropic[vertex] パッケージがインストールされていません。"
                "`pip install -e .` で再インストールするか、"
                "`pip install 'anthropic[vertex]' google-cloud-aiplatform` を実行してください。"
            ) from e

        if not project_id:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT が未設定です。"
                "プロジェクト直下の .env に GOOGLE_CLOUD_PROJECT=<your-project-id> を記載してください。"
            )

        self._client = AnthropicVertex(project_id=project_id, region=region)
        self.models = _AnthropicVertexModels(self._client)

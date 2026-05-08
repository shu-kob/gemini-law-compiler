"""Unit tests for src/config.py

LLM クライアント振り分けロジック (`is_ollama_model` / `is_anthropic_model` /
`get_llm_client`) の検証。Vertex AI / Ollama / Anthropic の実体は import
しないよう、`get_llm_client` のテストでは monkeypatch でモジュールを差し替える。
"""

from __future__ import annotations

import sys
import types

import pytest

from src import config
from src.config import is_anthropic_model, is_ollama_model


# ---------------------------------------------------------------------------
# is_ollama_model
# ---------------------------------------------------------------------------
class TestIsOllamaModel:
    @pytest.mark.parametrize(
        "model",
        [
            "gemma3:4b",
            "gemma3:12b",
            "gemma2:9b",
            "llama3.3:70b",
            "qwen2.5:7b",
            "mistral:7b",
            "phi4:14b",
        ],
    )
    def test_local_model_prefixes_are_recognized(self, model: str) -> None:
        assert is_ollama_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
            "claude-sonnet-4-6@default",
            "claude-opus-4-7@default",
            "gpt-4o",
        ],
    )
    def test_cloud_model_prefixes_are_rejected(self, model: str) -> None:
        assert is_ollama_model(model) is False


# ---------------------------------------------------------------------------
# is_anthropic_model
# ---------------------------------------------------------------------------
class TestIsAnthropicModel:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-6@default",
            "claude-opus-4-7@default",
            "claude-opus-4-6@default",
            "claude-3-5-sonnet@default",
        ],
    )
    def test_claude_prefixes_are_recognized(self, model: str) -> None:
        assert is_anthropic_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
            "gemma3:4b",
            "llama3.3:70b",
            "gpt-4o",
            "Claude-Sonnet",  # 大文字始まりは prefix 不一致 → 弾かれる
        ],
    )
    def test_non_claude_models_are_rejected(self, model: str) -> None:
        assert is_anthropic_model(model) is False


# ---------------------------------------------------------------------------
# get_llm_client: モデル名で 3 系統に振り分け
# ---------------------------------------------------------------------------
class TestGetLlmClient:
    def test_ollama_model_returns_ollama_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # OllamaClient の本体は urllib に触らないため、そのまま import OK
        client = config.get_llm_client("gemma3:4b")
        from src.llm.ollama_client import OllamaClient

        assert isinstance(client, OllamaClient)

    def test_anthropic_model_returns_anthropic_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # AnthropicClient.__init__ は `from anthropic import AnthropicVertex` を
        # 走らせるので、anthropic モジュールを fake で差し替える
        fake_module = types.ModuleType("anthropic")

        def fake_factory(project_id: str, region: str = "global"):
            obj = types.SimpleNamespace(project_id=project_id, region=region)
            obj.messages = types.SimpleNamespace()
            return obj

        fake_module.AnthropicVertex = fake_factory  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)
        monkeypatch.setattr(config, "VERTEX_PROJECT", "fake-project")

        client = config.get_llm_client("claude-sonnet-4-6@default")
        from src.llm.anthropic_client import AnthropicClient

        assert isinstance(client, AnthropicClient)

    def test_gemini_model_falls_through_to_genai_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # google.genai を呼ばないよう get_genai_client() ごと差し替える
        sentinel = object()

        def fake_get_genai_client():
            return sentinel

        monkeypatch.setattr(config, "get_genai_client", fake_get_genai_client)

        result = config.get_llm_client("gemini-3-flash-preview")
        assert result is sentinel

    def test_anthropic_path_requires_project_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # VERTEX_PROJECT が空のときは AnthropicClient.__init__ が RuntimeError を投げる
        fake_module = types.ModuleType("anthropic")
        fake_module.AnthropicVertex = lambda project_id, region="global": None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)
        monkeypatch.setattr(config, "VERTEX_PROJECT", "")

        with pytest.raises(RuntimeError) as ei:
            config.get_llm_client("claude-sonnet-4-6@default")
        assert "GOOGLE_CLOUD_PROJECT" in str(ei.value)


# ---------------------------------------------------------------------------
# get_genai_client: VERTEX_PROJECT 未設定時のガード
# ---------------------------------------------------------------------------
class TestGetGenaiClient:
    def test_missing_project_id_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "VERTEX_PROJECT", "")
        with pytest.raises(RuntimeError) as ei:
            config.get_genai_client()
        assert "GOOGLE_CLOUD_PROJECT" in str(ei.value)


# ---------------------------------------------------------------------------
# モデル定数の整合性
# ---------------------------------------------------------------------------
class TestModelConstants:
    def test_gemini_constants_have_expected_prefix(self) -> None:
        assert config.GEMINI_FLASH_MODEL.startswith("gemini-")
        assert config.GEMINI_PRO_MODEL.startswith("gemini-")

    def test_gemma_constant_routes_to_ollama(self) -> None:
        assert is_ollama_model(config.GEMMA3_MODEL) is True

    def test_claude_constant_routes_to_anthropic(self) -> None:
        assert is_anthropic_model(config.CLAUDE_MODEL) is True

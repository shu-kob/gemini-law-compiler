"""Unit tests for src/llm/ollama_client.py

ネットワークを叩かないため `urllib.request.urlopen` を monkeypatch で差し替え、
リクエストボディの構築・レスポンス整形・例外ラップを検証する。
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from src.llm.ollama_client import OllamaClient, OllamaResponse


class _FakeResponse:
    """`urlopen()` の戻り値を模した最小 fake (with 文で使えるようにする)。"""

    def __init__(self, payload: dict) -> None:
        self._buf = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._buf.close()
        return False

    def read(self) -> bytes:
        return self._buf.read()


@pytest.fixture
def captured_request() -> dict:
    """urlopen に渡された Request を保存する箱。"""
    return {}


@pytest.fixture
def patch_urlopen(monkeypatch: pytest.MonkeyPatch, captured_request: dict):
    """成功レスポンスを返す fake urlopen に差し替えるファクトリ。"""

    def _install(payload: dict) -> None:
        def fake_urlopen(req, timeout=None):
            captured_request["req"] = req
            captured_request["timeout"] = timeout
            return _FakeResponse(payload)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    return _install


# ---------------------------------------------------------------------------
# generate_content: 正常系のリクエスト構築
# ---------------------------------------------------------------------------
class TestGenerateContentRequest:
    def test_returns_response_text(self, patch_urlopen) -> None:
        patch_urlopen({"response": "hello"})
        client = OllamaClient("http://localhost:11434")
        result = client.models.generate_content("gemma3:4b", "ping")
        assert isinstance(result, OllamaResponse)
        assert result.text == "hello"

    def test_request_targets_generate_endpoint(
        self, patch_urlopen, captured_request: dict
    ) -> None:
        patch_urlopen({"response": "ok"})
        client = OllamaClient("http://localhost:11434")
        client.models.generate_content("gemma3:4b", "q")
        assert captured_request["req"].full_url == "http://localhost:11434/api/generate"

    def test_request_uses_post(self, patch_urlopen, captured_request: dict) -> None:
        patch_urlopen({"response": "ok"})
        client = OllamaClient("http://localhost:11434")
        client.models.generate_content("gemma3:4b", "q")
        assert captured_request["req"].get_method() == "POST"

    def test_request_sends_json_content_type(
        self, patch_urlopen, captured_request: dict
    ) -> None:
        patch_urlopen({"response": "ok"})
        client = OllamaClient("http://localhost:11434")
        client.models.generate_content("gemma3:4b", "q")
        assert captured_request["req"].get_header("Content-type") == "application/json"

    def test_request_body_contains_model_and_prompt(
        self, patch_urlopen, captured_request: dict
    ) -> None:
        patch_urlopen({"response": "ok"})
        client = OllamaClient("http://localhost:11434")
        client.models.generate_content("gemma3:4b", "ハロー")
        body = json.loads(captured_request["req"].data.decode("utf-8"))
        assert body["model"] == "gemma3:4b"
        assert body["prompt"] == "ハロー"
        assert body["stream"] is False

    def test_default_temperature_is_zero(
        self, patch_urlopen, captured_request: dict
    ) -> None:
        patch_urlopen({"response": "ok"})
        client = OllamaClient("http://localhost:11434")
        client.models.generate_content("gemma3:4b", "q")
        body = json.loads(captured_request["req"].data.decode("utf-8"))
        assert body["options"]["temperature"] == 0.0

    def test_custom_temperature_is_passed(
        self, patch_urlopen, captured_request: dict
    ) -> None:
        patch_urlopen({"response": "ok"})
        client = OllamaClient("http://localhost:11434")
        client.models.generate_content(
            "gemma3:4b", "q", config={"temperature": 0.7}
        )
        body = json.loads(captured_request["req"].data.decode("utf-8"))
        assert body["options"]["temperature"] == pytest.approx(0.7)

    def test_system_instruction_is_propagated(
        self, patch_urlopen, captured_request: dict
    ) -> None:
        patch_urlopen({"response": "ok"})
        client = OllamaClient("http://localhost:11434")
        client.models.generate_content(
            "gemma3:4b",
            "q",
            config={"system_instruction": "あなたは法律の専門家です"},
        )
        body = json.loads(captured_request["req"].data.decode("utf-8"))
        assert body["system"] == "あなたは法律の専門家です"

    def test_no_system_key_when_instruction_absent(
        self, patch_urlopen, captured_request: dict
    ) -> None:
        patch_urlopen({"response": "ok"})
        client = OllamaClient("http://localhost:11434")
        client.models.generate_content("gemma3:4b", "q")
        body = json.loads(captured_request["req"].data.decode("utf-8"))
        assert "system" not in body

    def test_long_timeout_is_set(
        self, patch_urlopen, captured_request: dict
    ) -> None:
        # ローカル LLM の長時間推論を許容するため timeout は十分長く取られている
        patch_urlopen({"response": "ok"})
        client = OllamaClient("http://localhost:11434")
        client.models.generate_content("gemma3:4b", "q")
        assert captured_request["timeout"] >= 60

    def test_missing_response_field_returns_empty_string(
        self, patch_urlopen
    ) -> None:
        # Ollama がエラーレスポンスを返した場合でも text は空文字に正規化
        patch_urlopen({"error": "model not found"})
        client = OllamaClient("http://localhost:11434")
        result = client.models.generate_content("gemma3:4b", "q")
        assert result.text == ""


# ---------------------------------------------------------------------------
# host 正規化: 末尾スラッシュ吸収
# ---------------------------------------------------------------------------
class TestHostNormalization:
    def test_trailing_slash_is_stripped(
        self, patch_urlopen, captured_request: dict
    ) -> None:
        patch_urlopen({"response": "ok"})
        client = OllamaClient("http://localhost:11434/")
        client.models.generate_content("gemma3:4b", "q")
        # 末尾スラッシュを含むホストでも `//api/generate` にならない
        assert (
            captured_request["req"].full_url == "http://localhost:11434/api/generate"
        )

    def test_default_host(self) -> None:
        client = OllamaClient()
        # デフォルトホストが localhost:11434 で生成されること (副作用なし)
        assert client.models._host == "http://localhost:11434"


# ---------------------------------------------------------------------------
# 例外ラップ: ネットワークエラーは RuntimeError に包む
# ---------------------------------------------------------------------------
class TestNetworkErrorWrapping:
    def test_url_error_wrapped_in_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        client = OllamaClient("http://localhost:11434")
        with pytest.raises(RuntimeError) as ei:
            client.models.generate_content("gemma3:4b", "q")
        # 元の例外が __cause__ にぶら下がる
        assert isinstance(ei.value.__cause__, urllib.error.URLError)
        # ユーザーへの誘導メッセージが含まれる
        assert "ollama serve" in str(ei.value)

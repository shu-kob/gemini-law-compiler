"""JudgeService および API サーバーのモデル解決テスト。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.judge_service import SUPPORTED_MODELS, JudgeService
from src.api.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_supported_models_contain_gemini_3_8_and_3_5_lite() -> None:
    assert "flash_3_8" in SUPPORTED_MODELS
    assert "flash_lite" in SUPPORTED_MODELS
    assert SUPPORTED_MODELS["flash_3_8"] == "gemini-3.8-flash"
    assert SUPPORTED_MODELS["flash_lite"] == "gemini-3.5-flash-lite"


def test_judge_service_resolves_new_models() -> None:
    service = JudgeService()
    assert service.resolve_model("flash_3_8") == "gemini-3.8-flash"
    assert service.resolve_model("flash_lite") == "gemini-3.5-flash-lite"


def test_api_models_endpoint_includes_new_models(client: TestClient) -> None:
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    models = {m["key"]: m for m in data["models"]}

    assert "flash_3_8" in models
    assert models["flash_3_8"]["model_id"] == "gemini-3.8-flash"
    assert models["flash_3_8"]["supports_web_search"] is True

    assert "flash_lite" in models
    assert models["flash_lite"]["model_id"] == "gemini-3.5-flash-lite"
    assert models["flash_lite"]["supports_web_search"] is True

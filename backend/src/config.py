"""共通設定"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # backend/
REPO_ROOT = PROJECT_ROOT.parent

# リポジトリ直下の .env からGCP設定などを読み込む（.env はGit管理外）
load_dotenv(REPO_ROOT / ".env")

# GOOGLE_APPLICATION_CREDENTIALS がサービスアカウントキーを指していると
# Vertex AI ADCの認証が上書きされるため、プロセス起動時に除外する。
# ADC (application_default_credentials.json) を優先して使う。
if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

DATA_DIR = PROJECT_ROOT / "data"
XML_PATH = DATA_DIR / "road_traffic_act_full.xml"
FINE_TABLE_PATH = DATA_DIR / "bicycle_fine_table.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# Gemini API Key（Google AI Studio）
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# プロバイダ設定: "ai_studio" (デフォルト) または "vertex"
GEMINI_PROVIDER = os.environ.get("GEMINI_PROVIDER")
if not GEMINI_PROVIDER:
    if GEMINI_API_KEY:
        GEMINI_PROVIDER = "ai_studio"
    elif os.environ.get("GOOGLE_CLOUD_PROJECT"):
        GEMINI_PROVIDER = "vertex"
    else:
        GEMINI_PROVIDER = "ai_studio"

# Gemini モデル設定（Gemini 3.7 Flash / Gemini 3.1 Pro / 環境変数で上書き可能）
if GEMINI_PROVIDER == "vertex":
    DEFAULT_FLASH = "gemini-3-flash-preview"
    DEFAULT_PRO = "gemini-3.1-pro-preview"
else:
    DEFAULT_FLASH = "gemini-3.7-flash"
    DEFAULT_PRO = "gemini-3.1-pro-preview"

GEMINI_FLASH_MODEL = os.environ.get("GEMINI_FLASH_MODEL", DEFAULT_FLASH)
GEMINI_PRO_MODEL = os.environ.get("GEMINI_PRO_MODEL", DEFAULT_PRO)

# ローカルLLM設定（Ollama 経由）
GEMMA3_MODEL = "gemma3:4b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


# Anthropic Claude（Vertex AI 経由）設定
CLAUDE_MODEL = "claude-opus-4-7@default"
CLAUDE_SONNET_MODEL = "claude-sonnet-4-6@default"

# Google Cloud設定（Vertex AI利用時）
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")


def is_ollama_model(model: str) -> bool:
    """ローカル（Ollama）で動かすモデルかどうか。"""
    return model.startswith(("gemma", "llama", "qwen", "mistral", "phi"))


def is_anthropic_model(model: str) -> bool:
    """Anthropic Claude API で動かすモデルかどうか。"""
    return model.startswith("claude-")


def get_genai_client():
    """GenAIクライアントを生成する（デフォルト: Google AI Studio, オプション: Vertex AI）。"""
    from google import genai

    if GEMINI_PROVIDER == "vertex":
        if not VERTEX_PROJECT:
            raise RuntimeError(
                "GEMINI_PROVIDER=vertex ですが GOOGLE_CLOUD_PROJECT が未設定です。"
                ".env に GOOGLE_CLOUD_PROJECT=<your-project-id> を記載するか、"
                "GEMINI_API_KEY を設定して Google AI Studio モードをご利用ください。"
            )
        return genai.Client(
            vertexai=True,
            project=VERTEX_PROJECT,
            location=VERTEX_LOCATION,
        )

    # デフォルト: Google AI Studio
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY が未設定です。\n"
            "プロジェクト直下の .env に GEMINI_API_KEY=<your-api-key> を記載してください。\n"
            "（Vertex AI を使用する場合は .env に GEMINI_PROVIDER=vertex と GOOGLE_CLOUD_PROJECT=<id> を記載してください）"
        )
    return genai.Client(api_key=GEMINI_API_KEY)


def get_llm_client(model: str):
    """モデル名に応じて LLM クライアントを返す。

    Gemini / Ollama / Anthropic クライアントは同じ
    `client.models.generate_content(model, contents, config)` インタフェースを持つ。
    """
    if is_ollama_model(model):
        from src.llm.ollama_client import OllamaClient
        return OllamaClient(OLLAMA_HOST)
    if is_anthropic_model(model):
        from src.llm.anthropic_client import AnthropicClient
        return AnthropicClient(project_id=VERTEX_PROJECT, region=VERTEX_LOCATION)
    return get_genai_client()


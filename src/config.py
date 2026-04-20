"""共通設定"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# プロジェクト直下の .env からGCP設定などを読み込む（.env はGit管理外）
load_dotenv(PROJECT_ROOT / ".env")

# GOOGLE_APPLICATION_CREDENTIALS がサービスアカウントキーを指していると
# Vertex AI ADCの認証が上書きされるため、プロセス起動時に除外する。
# ADC (application_default_credentials.json) を優先して使う。
if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

DATA_DIR = PROJECT_ROOT / "data"
XML_PATH = DATA_DIR / "road_traffic_act_full.xml"
FINE_TABLE_PATH = DATA_DIR / "bicycle_fine_table.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# Gemini API設定
GEMINI_FLASH_MODEL = "gemini-3-flash-preview"
GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"

# Google Cloud設定（.env で管理。リポジトリには含めない）
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")


def get_genai_client():
    """Vertex AI ADC経由のGenAIクライアントを生成する。"""
    if not VERTEX_PROJECT:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT が未設定です。"
            "プロジェクト直下の .env に GOOGLE_CLOUD_PROJECT=<your-project-id> を記載してください。"
        )
    from google import genai
    return genai.Client(
        vertexai=True,
        project=VERTEX_PROJECT,
        location=VERTEX_LOCATION,
    )

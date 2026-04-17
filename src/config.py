"""共通設定"""

import os
from pathlib import Path

# GOOGLE_APPLICATION_CREDENTIALS がサービスアカウントキーを指していると
# Vertex AI ADCの認証が上書きされるため、プロセス起動時に除外する。
# ADC (application_default_credentials.json) を優先して使う。
if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
XML_PATH = DATA_DIR / "road_traffic_act_full.xml"
FINE_TABLE_PATH = DATA_DIR / "bicycle_fine_table.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# Gemini API設定
GEMINI_FLASH_MODEL = "gemini-3-flash-preview"
GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"

# Google Cloud設定
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "test-2163-kobuchi-shu")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")


def get_genai_client():
    """Vertex AI ADC経由のGenAIクライアントを生成する。"""
    from google import genai
    return genai.Client(
        vertexai=True,
        project=VERTEX_PROJECT,
        location=VERTEX_LOCATION,
    )

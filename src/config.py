"""共通設定"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
XML_PATH = DATA_DIR / "road_traffic_act_full.xml"
FINE_TABLE_PATH = DATA_DIR / "bicycle_fine_table.json"

# Gemini API設定
GEMINI_FLASH_MODEL = "gemini-2.5-flash-preview-05-20"
GEMINI_PRO_MODEL = "gemini-2.5-pro-preview-05-06"

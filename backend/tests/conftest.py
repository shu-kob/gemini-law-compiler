"""共通テストフィクスチャ。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmark.flash_only_judge import TestCase as _BenchTestCase
from src.parser.legal_compiler import (
    LawAST,
    extract_bicycle_articles,
    parse_egov_xml,
)

# pytest が `TestCase` (dataclass) をテストクラスとして収集しないようにする
_BenchTestCase.__test__ = False


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_XML = FIXTURES_DIR / "sample_law.xml"


@pytest.fixture(scope="session")
def sample_xml_path() -> Path:
    return SAMPLE_XML


@pytest.fixture(scope="session")
def sample_ast(sample_xml_path: Path) -> LawAST:
    return parse_egov_xml(sample_xml_path)


@pytest.fixture(scope="session")
def sample_bicycle_articles(sample_ast: LawAST):
    return extract_bicycle_articles(sample_ast)

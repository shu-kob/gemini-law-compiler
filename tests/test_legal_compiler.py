"""Unit tests for src/parser/legal_compiler.py

Layer 1 (2008年式決定論的パーサー) の契約を固定化する。
"""

from __future__ import annotations

import pytest

from src.parser.legal_compiler import (
    ArticleNode,
    LawAST,
    _detect_logic_flags,
    _extract_text,
    extract_bicycle_articles,
    flatten_article_text,
    parse_egov_xml,
)
from xml.etree import ElementTree as ET


# ---------------------------------------------------------------------------
# _detect_logic_flags: 条文文字列 → 論理トークンフラグ
# ---------------------------------------------------------------------------
class TestDetectLogicFlags:
    @pytest.mark.parametrize(
        "text,expected_flag",
        [
            ("何人も酒気を帯びて車両等を運転してはならない。", "prohibition"),
            ("歩道の中央を徐行しなければならない。", "obligation"),
            ("歩道を通行することができる。", "permission"),
            ("ただし、この限りでない。", "proviso"),
            ("身体障害者用の車いす等を除く車両。", "exception"),
            ("政令で定める者であるとき。", "delegation"),
            ("前条の規定を準用する。", "application_mutatis"),
            ("第十七条第一項の規定にかかわらず", "notwithstanding"),
            ("他人に危害を及ぼさないよう努めなければならない。", "effort_obligation"),
            ("罰則は別表のとおりとする。", "penalty_ref"),
        ],
    )
    def test_single_pattern_detection(self, text: str, expected_flag: str) -> None:
        flags = _detect_logic_flags(text)
        assert expected_flag in flags, f"{expected_flag} not found in {flags}"

    def test_multiple_flags_coexist(self) -> None:
        text = (
            "普通自転車は、第十七条第一項の規定にかかわらず、"
            "歩道を通行することができる。"
            "ただし、政令で定める場合は、この限りでない。"
        )
        flags = set(_detect_logic_flags(text))
        assert {"notwithstanding", "permission", "proviso", "delegation"} <= flags

    def test_no_match_returns_empty(self) -> None:
        assert _detect_logic_flags("赤い自転車が走っていた。") == []

    def test_empty_string_returns_empty(self) -> None:
        assert _detect_logic_flags("") == []

    def test_effort_obligation_is_disjoint_from_obligation(self) -> None:
        # 「よう努めなければならない」は effort_obligation のみにマッチし、
        # obligation（"しなければならない" literal）にはマッチしない。
        # 実装は literal match のため末尾活用違いで両立しない仕様を固定化。
        flags = _detect_logic_flags("他人に危害を及ぼさないよう努めなければならない。")
        assert "effort_obligation" in flags
        assert "obligation" not in flags

    def test_obligation_pattern_is_literal(self) -> None:
        # パターンは literal 「しなければならない」である。
        # 活用違い（「従わなければならない」等）ではマッチしない。
        # これは現行実装の仕様 — 法規テキストは文末が「しなければならない」に
        # 収束する傾向があるため実運用上は十分とされる。
        assert _detect_logic_flags("信号に従わなければならない。") == []


# ---------------------------------------------------------------------------
# _extract_text: Ruby/Rt を除外した再帰的テキスト抽出
# ---------------------------------------------------------------------------
class TestExtractText:
    def test_plain_text(self) -> None:
        elem = ET.fromstring("<S>hello</S>")
        assert _extract_text(elem) == "hello"

    def test_ruby_rt_is_stripped(self) -> None:
        elem = ET.fromstring("<S>道<Ruby>路<Rt>ろ</Rt></Ruby>を渡る</S>")
        # Rt の「ろ」が除外され、ベーステキスト「路」が残る
        result = _extract_text(elem)
        assert "ろ" not in result
        assert "路" in result
        assert "道" in result
        assert "を渡る" in result

    def test_nested_text_and_tail(self) -> None:
        elem = ET.fromstring("<S>A<B>X</B>C<D>Y</D>E</S>")
        assert _extract_text(elem) == "AXCYE"


# ---------------------------------------------------------------------------
# parse_egov_xml: e-Gov XML → LawAST
# ---------------------------------------------------------------------------
class TestParseEgovXML:
    def test_returns_law_ast(self, sample_ast: LawAST) -> None:
        assert isinstance(sample_ast, LawAST)

    def test_law_title_is_extracted(self, sample_ast: LawAST) -> None:
        assert "道路交通法" in sample_ast.title

    def test_all_articles_parsed(self, sample_ast: LawAST) -> None:
        # フィクスチャに含まれる Article は 5 つ: 2, 7, 63_4, 65, 70, 999
        nums = [a.num for a in sample_ast.articles]
        assert nums == ["2", "7", "63_4", "65", "70", "999"]

    def test_article_caption_and_title(self, sample_ast: LawAST) -> None:
        art = sample_ast.find_article("63_4")
        assert art is not None
        assert art.caption == "（普通自転車の歩道通行）"
        assert art.title == "第六十三条の四"
        assert art.address == "第六十三条の四"  # addressプロパティはtitleのエイリアス

    def test_paragraph_structure(self, sample_ast: LawAST) -> None:
        art = sample_ast.find_article("63_4")
        assert art is not None
        assert len(art.paragraphs) == 2

    def test_main_and_proviso_sentences(self, sample_ast: LawAST) -> None:
        art = sample_ast.find_article("63_4")
        para1 = art.paragraphs[0]
        assert len(para1.sentences) == 2
        funcs = [s.function for s in para1.sentences]
        assert funcs == ["main", "proviso"]

    def test_items_parsed(self, sample_ast: LawAST) -> None:
        art = sample_ast.find_article("63_4")
        para1 = art.paragraphs[0]
        assert len(para1.items) == 2
        assert para1.items[0].title == "一"
        assert para1.items[1].title == "二"

    def test_logic_flags_on_sentences(self, sample_ast: LawAST) -> None:
        art = sample_ast.find_article("63_4")
        para1 = art.paragraphs[0]
        # main: にかかわらず + することができる
        main_flags = set(para1.sentences[0].logic_flags)
        assert "notwithstanding" in main_flags
        assert "permission" in main_flags
        # proviso: ただし + してはならない
        proviso_flags = set(para1.sentences[1].logic_flags)
        assert "proviso" in proviso_flags
        assert "prohibition" in proviso_flags

    def test_delegation_flag_on_item(self, sample_ast: LawAST) -> None:
        art = sample_ast.find_article("63_4")
        item2 = art.paragraphs[0].items[1]
        assert "delegation" in item2.sentences[0].logic_flags

    def test_suppl_note_preserved(self, sample_ast: LawAST) -> None:
        art = sample_ast.find_article("63_4")
        assert "罰則" in art.suppl_note

    def test_find_article_returns_none_for_missing(self, sample_ast: LawAST) -> None:
        assert sample_ast.find_article("99999") is None

    def test_missing_lawbody_raises(self, tmp_path) -> None:
        bad = tmp_path / "bad.xml"
        bad.write_text("<?xml version='1.0'?><Root><Other/></Root>", encoding="utf-8")
        with pytest.raises(ValueError, match="LawBody"):
            parse_egov_xml(bad)


# ---------------------------------------------------------------------------
# LawAST.to_dict / to_json
# ---------------------------------------------------------------------------
class TestLawASTSerialization:
    def test_to_dict_shape(self, sample_ast: LawAST) -> None:
        d = sample_ast.to_dict()
        assert "title" in d and "articles" in d
        assert isinstance(d["articles"], list)
        assert d["articles"][0]["num"] == "2"

    def test_to_json_is_valid(self, sample_ast: LawAST) -> None:
        import json

        parsed = json.loads(sample_ast.to_json())
        assert parsed["title"] == sample_ast.title


# ---------------------------------------------------------------------------
# extract_bicycle_articles: 自転車関連条文の抽出
# ---------------------------------------------------------------------------
class TestExtractBicycleArticles:
    def test_returns_only_bicycle_articles(self, sample_ast: LawAST) -> None:
        bicycle = extract_bicycle_articles(sample_ast)
        nums = {a.num for a in bicycle}
        # フィクスチャ中で自転車関連は 2, 7, 63_4, 65, 70
        assert nums == {"2", "7", "63_4", "65", "70"}

    def test_non_bicycle_article_excluded(self, sample_ast: LawAST) -> None:
        bicycle = extract_bicycle_articles(sample_ast)
        assert all(a.num != "999" for a in bicycle)

    def test_returns_article_nodes(self, sample_ast: LawAST) -> None:
        bicycle = extract_bicycle_articles(sample_ast)
        assert all(isinstance(a, ArticleNode) for a in bicycle)


# ---------------------------------------------------------------------------
# flatten_article_text: 条文 → 平文テキスト（VSM用）
# ---------------------------------------------------------------------------
class TestFlattenArticleText:
    def test_includes_title_and_caption(self, sample_ast: LawAST) -> None:
        art = sample_ast.find_article("63_4")
        text = flatten_article_text(art)
        assert "第六十三条の四" in text
        assert "普通自転車の歩道通行" in text

    def test_includes_all_sentences(self, sample_ast: LawAST) -> None:
        art = sample_ast.find_article("63_4")
        text = flatten_article_text(art)
        assert "歩道を通行することができる" in text
        assert "ただし" in text
        assert "徐行しなければならない" in text

    def test_includes_item_titles(self, sample_ast: LawAST) -> None:
        art = sample_ast.find_article("63_4")
        text = flatten_article_text(art)
        assert "一" in text
        assert "二" in text
        assert "政令で定める者" in text

    def test_empty_article_produces_header_only(self) -> None:
        art = ArticleNode(num="1", caption="(空)", title="第一条")
        text = flatten_article_text(art)
        assert text.startswith("第一条")

"""
Task 2: 2008-Style Deterministic Legal Compiler
e-Gov XML → 条文ASTへの決定論的変換器

条・項・号のヒエラルキーを完全にパースし、
「〜を除く」「〜を準用する」等の論理トークンをフラグとして抽出する。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


# --- 論理トークンパターン（2008年卒論由来） ---
LOGIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "exception": re.compile(r"を除[くき]"),
    "proviso": re.compile(r"ただし[、,]"),
    "delegation": re.compile(r"政令で定める"),
    "application_mutatis": re.compile(r"を準用する"),
    "notwithstanding": re.compile(r"にかかわらず"),
    "obligation": re.compile(r"しなければならない"),
    "prohibition": re.compile(r"してはならない"),
    "permission": re.compile(r"することができる"),
    "effort_obligation": re.compile(r"よう努めなければならない"),
    "penalty_ref": re.compile(r"罰則"),
}


@dataclass
class SentenceNode:
    text: str
    function: str  # "main", "proviso", or ""
    logic_flags: list[str] = field(default_factory=list)


@dataclass
class ItemNode:
    num: str
    title: str
    sentences: list[SentenceNode] = field(default_factory=list)
    subitems: list[ItemNode] = field(default_factory=list)


@dataclass
class ParagraphNode:
    num: str
    sentences: list[SentenceNode] = field(default_factory=list)
    items: list[ItemNode] = field(default_factory=list)


@dataclass
class ArticleNode:
    num: str
    caption: str
    title: str
    paragraphs: list[ParagraphNode] = field(default_factory=list)
    suppl_note: str = ""

    @property
    def address(self) -> str:
        """人間可読な条文アドレス (例: '第63条の4')"""
        return self.title


@dataclass
class LawAST:
    title: str
    articles: list[ArticleNode] = field(default_factory=list)

    def find_article(self, num: str) -> ArticleNode | None:
        for a in self.articles:
            if a.num == num:
                return a
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _extract_text(elem: ET.Element) -> str:
    """要素内の全テキストを再帰的に結合（Ruby/Rt対応）"""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        if child.tag == "Rt":
            continue  # ルビの読みは除外
        parts.append(_extract_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _detect_logic_flags(text: str) -> list[str]:
    """条文テキストから論理フラグを検出"""
    return [name for name, pat in LOGIC_PATTERNS.items() if pat.search(text)]


def _parse_sentence(elem: ET.Element) -> SentenceNode:
    text = _extract_text(elem)
    func = elem.get("Function", "")
    return SentenceNode(text=text, function=func, logic_flags=_detect_logic_flags(text))


def _parse_item(elem: ET.Element) -> ItemNode:
    title_el = elem.find("ItemTitle")
    title = _extract_text(title_el) if title_el is not None else ""

    sentences: list[SentenceNode] = []
    sent_container = elem.find("ItemSentence")
    if sent_container is not None:
        for s in sent_container.findall("Sentence"):
            sentences.append(_parse_sentence(s))

    subitems: list[ItemNode] = []
    for tag in ("Subitem1", "Subitem2", "Subitem3"):
        for sub in elem.findall(tag):
            subitems.append(_parse_item(sub))

    return ItemNode(
        num=elem.get("Num", ""),
        title=title,
        sentences=sentences,
        subitems=subitems,
    )


def _parse_paragraph(elem: ET.Element) -> ParagraphNode:
    num_el = elem.find("ParagraphNum")
    num = _extract_text(num_el) if num_el is not None else ""

    sentences: list[SentenceNode] = []
    sent_container = elem.find("ParagraphSentence")
    if sent_container is not None:
        for s in sent_container.findall("Sentence"):
            sentences.append(_parse_sentence(s))

    items = [_parse_item(it) for it in elem.findall("Item")]
    return ParagraphNode(num=num, sentences=sentences, items=items)


def _parse_article(elem: ET.Element) -> ArticleNode:
    caption_el = elem.find("ArticleCaption")
    caption = _extract_text(caption_el) if caption_el is not None else ""
    title_el = elem.find("ArticleTitle")
    title = _extract_text(title_el) if title_el is not None else ""
    suppl_el = elem.find("SupplNote")
    suppl = _extract_text(suppl_el) if suppl_el is not None else ""

    paragraphs = [_parse_paragraph(p) for p in elem.findall("Paragraph")]
    return ArticleNode(
        num=elem.get("Num", ""),
        caption=caption,
        title=title,
        paragraphs=paragraphs,
        suppl_note=suppl,
    )


def parse_egov_xml(xml_path: str | Path) -> LawAST:
    """e-Gov法令XMLをパースしてLawASTを構築する"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # e-Gov API応答の場合: DataRoot > ApplData > LawFullText > Law > LawBody
    law_body = root.find(".//LawBody")
    if law_body is None:
        raise ValueError("LawBody element not found in XML")

    title_el = law_body.find("LawTitle")
    title = _extract_text(title_el) if title_el is not None else ""

    # 全Articleを再帰的に収集（Chapter/Section内も含む）
    articles = [_parse_article(a) for a in law_body.iter("Article")]

    return LawAST(title=title, articles=articles)


def extract_bicycle_articles(ast: LawAST) -> list[ArticleNode]:
    """自転車関連条文を抽出する"""
    bicycle_nums = {
        "2",       # 定義（軽車両・自転車の定義）
        "7",       # 信号機の信号等に従う義務
        "8",       # 通行の禁止等
        "17",      # 通行区分
        "17_2",    # 路側帯の通行
        "19",      # 軽車両の並進の禁止
        "22",      # 最高速度
        "24",      # 急ブレーキの禁止
        "26",      # 車間距離の保持
        "33",      # 踏切の通過
        "34",      # 左折又は右折の方法
        "36",      # 交差点における他の車両等との関係等
        "38",      # 横断歩道等における歩行者等の優先
        "40",      # 緊急自動車の優先
        "43",      # 指定場所における一時停止
        "44",      # 停車及び駐車を禁止する場所
        "52",      # 車両等の灯火
        "53",      # 合図
        "57",      # 乗車又は積載の制限等
        "63_3",    # 自転車道の通行区分
        "63_4",    # 普通自転車の歩道通行
        "63_5",    # 普通自転車の並進
        "63_6",    # 自転車の横断の方法
        "63_7",    # 交差点における自転車の通行方法
        "63_8",    # 自転車の通行方法の指示
        "63_9",    # 自転車の制動装置等
        "63_10",   # 自転車の検査等
        "63_11",   # 自転車の運転者等の遵守事項
        "65",      # 酒気帯び運転等の禁止
        "70",      # 安全運転の義務
        "71",      # 運転者の遵守事項
    }
    return [a for a in ast.articles if a.num in bicycle_nums]


def flatten_article_text(article: ArticleNode) -> str:
    """条文を平文テキストに変換する（VSMベクトル化用）"""
    parts = [f"{article.title} {article.caption}"]
    for para in article.paragraphs:
        for sent in para.sentences:
            parts.append(sent.text)
        for item in para.items:
            for sent in item.sentences:
                parts.append(f"{item.title} {sent.text}")
            for sub in item.subitems:
                for sent in sub.sentences:
                    parts.append(f"  {sub.title} {sent.text}")
    return "\n".join(parts)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    xml_path = data_dir / "road_traffic_act_full.xml"

    print("[2008-Thesis-Logic]: e-Gov XMLをパース中...")
    ast = parse_egov_xml(xml_path)
    print(f"[2008-Thesis-Logic]: パース完了。全{len(ast.articles)}条を検出。")

    bicycle = extract_bicycle_articles(ast)
    print(f"[2008-Thesis-Logic]: 自転車関連条文 {len(bicycle)}条を抽出。")

    for a in bicycle[:5]:
        flags = set()
        for p in a.paragraphs:
            for s in p.sentences:
                flags.update(s.logic_flags)
        print(f"  {a.title} {a.caption} → 論理フラグ: {flags or '(なし)'}")

"""
Task 3: 2008年式 Vector Space Model (VSM) コサイン類似度エンジン

卒論ロジックの実装:
  入力事例をTF-IDFベクトル化し、全条文の中でcos類似度が最も高い条文を特定する。
  外部ライブラリ不使用 — 純粋なPythonとmathのみで2008年の精神を再現。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from src.parser.legal_compiler import ArticleNode, LawAST, flatten_article_text


# --- トークナイザ（MeCab不要の簡易版、文字N-gram + 単語分割） ---
_PARTICLE_RE = re.compile(
    r"[はがのをにへでとやもからまでよりばかりなどしかさえだけほどくらい]"
)
_SPLIT_RE = re.compile(r"[、。,.\s（）()「」『』\[\]]+")


def tokenize(text: str) -> list[str]:
    """日本語テキストをbi-gram + キーワード分割でトークン化する。
    MeCabなしでも法規テキストで十分な精度を出す2008年式アプローチ。
    """
    tokens: list[str] = []

    # 1) 句読点・括弧で分割した後のチャンクをbi-gramに
    chunks = _SPLIT_RE.split(text)
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 2:
            continue
        # 文字bi-gram
        for i in range(len(chunk) - 1):
            bigram = chunk[i : i + 2]
            # 助詞のみのbi-gramは除外
            if not _PARTICLE_RE.fullmatch(bigram):
                tokens.append(bigram)

    # 2) 法規特有のキーワードを明示的に追加
    legal_keywords = [
        "自転車", "歩道", "車道", "通行", "運転", "違反", "禁止",
        "義務", "罰則", "反則金", "信号", "一時停止", "徐行",
        "酒気帯び", "携帯電話", "ヘルメット", "制動装置", "ブレーキ",
        "高齢者", "児童", "幼児", "歩行者", "横断歩道",
        "普通自転車", "軽車両", "原動機", "踏切", "駐車",
        "並進", "安全運転", "左折", "右折", "追越し",
        "政令で定める", "にかかわらず", "を除く",
    ]
    for kw in legal_keywords:
        if kw in text:
            tokens.append(kw)

    return tokens


@dataclass
class VSMMatch:
    article: ArticleNode
    score: float
    rank: int


class VSMEngine:
    """2008年式 TF-IDF cos類似度検索エンジン"""

    def __init__(self, ast: LawAST, article_filter: list[ArticleNode] | None = None):
        self._articles = article_filter if article_filter is not None else ast.articles
        self._doc_tokens: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._doc_tfidf: list[dict[str, float]] = []
        self._build_index()

    def _build_index(self) -> None:
        n = len(self._articles)
        if n == 0:
            return

        # 各条文をトークン化
        self._doc_tokens = [
            tokenize(flatten_article_text(a)) for a in self._articles
        ]

        # DF (document frequency) を計算
        df: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            unique = set(tokens)
            for t in unique:
                df[t] += 1

        # IDF = log(N / df) + 1  (smoothed)
        self._idf = {t: math.log(n / freq) + 1.0 for t, freq in df.items()}

        # 各文書のTF-IDFベクトルを事前計算
        self._doc_tfidf = []
        for tokens in self._doc_tokens:
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1
            tfidf = {
                t: (count / total) * self._idf.get(t, 1.0) for t, count in tf.items()
            }
            self._doc_tfidf.append(tfidf)

    def search(self, query: str, top_k: int = 5) -> list[VSMMatch]:
        """クエリに最も類似する条文をcos類似度で検索する"""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        q_tf = Counter(q_tokens)
        q_total = len(q_tokens)
        q_tfidf = {
            t: (count / q_total) * self._idf.get(t, 1.0) for t, count in q_tf.items()
        }

        scores: list[tuple[int, float]] = []
        for i, doc_vec in enumerate(self._doc_tfidf):
            score = self._cosine_similarity(q_tfidf, doc_vec)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(scores[:top_k], 1):
            results.append(VSMMatch(
                article=self._articles[idx],
                score=round(score, 6),
                rank=rank,
            ))
        return results

    @staticmethod
    def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
        """2つのTF-IDFベクトル間のcos類似度を計算"""
        # 共通キーのみでドット積
        common = set(a) & set(b)
        if not common:
            return 0.0

        dot = sum(a[k] * b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


if __name__ == "__main__":
    from src.parser.legal_compiler import parse_egov_xml, extract_bicycle_articles
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    xml_path = data_dir / "road_traffic_act_full.xml"

    print("[2008-Thesis-Logic]: 法規ASTを構築中...")
    ast = parse_egov_xml(xml_path)
    bicycle_articles = extract_bicycle_articles(ast)

    print(f"[2008-Thesis-Logic]: VSMインデックス構築中... ({len(bicycle_articles)}条文)")
    engine = VSMEngine(ast, article_filter=bicycle_articles)

    test_queries = [
        "75歳の高齢者が歩道を自転車で走行した",
        "自転車でスマホを操作しながら運転した",
        "自転車で飲酒運転をした",
        "自転車で一時停止の標識を無視した",
        "自転車で二人乗りをした",
    ]

    for query in test_queries:
        results = engine.search(query, top_k=3)
        print(f"\n[2008-Thesis-Logic]: クエリ「{query}」")
        for m in results:
            print(f"  #{m.rank} cos={m.score:.4f} → {m.article.title} {m.article.caption}")

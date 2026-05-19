"""
Task 3: 2008年式 Vector Space Model (VSM) コサイン類似度エンジン

卒論ロジックの実装:
  入力事例をTF-IDFベクトル化し、全条文の中でcos類似度が最も高い条文を特定する。
  外部ライブラリ不使用 — 純粋なPythonとmathのみで2008年の精神を再現。

オプションのハイブリッドモード:
  TF-IDF (語彙一致) と Vertex AI Embedding (意味距離) を min-max 正規化後に
  加重和してスコアを出す。alpha=0.5 がデフォルトで、TF-IDF と embedding を
  半々で混ぜる。alpha=1.0 で純 TF-IDF (2008 年式) に戻る。
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
    """2008年式 TF-IDF cos類似度検索エンジン（オプションで embedding ハイブリッド）"""

    def __init__(
        self,
        ast: LawAST,
        article_filter: list[ArticleNode] | None = None,
        embedding_engine=None,
        alpha: float = 0.3,
    ):
        """
        Args:
            ast: パース済み法令 AST
            article_filter: 検索対象に絞り込む条文リスト（None で全条文）
            embedding_engine: 意味埋め込みエンジン（None で TF-IDF 単体）
            alpha: TF-IDF の重み (0.0=embedding のみ, 1.0=TF-IDF のみ)。
                デフォルト 0.3 は embedding 主・TF-IDF 補助。これは TC-003
                のような口語クエリ（「スマホ」⇄「無線通信のために用いられる
                装置」のような語彙ギャップ）で TF-IDF だけでは取りこぼした
                条文（第71条）を救うためのチューニング値。
        """
        self._articles = article_filter if article_filter is not None else ast.articles
        self._doc_tokens: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._doc_tfidf: list[dict[str, float]] = []
        self._embedding_engine = embedding_engine
        self._alpha = alpha
        self._build_index()

    @property
    def is_hybrid(self) -> bool:
        return self._embedding_engine is not None and self._alpha < 1.0

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
        """クエリに最も類似する条文を cos 類似度で検索する。

        embedding_engine が渡されており alpha < 1.0 なら、TF-IDF と
        embedding の cos を min-max 正規化してから alpha 加重和でスコアを出す。
        """
        tfidf_scores = self._tfidf_scores(query)
        if not tfidf_scores:
            return []

        if self.is_hybrid:
            emb_scores = self._embedding_engine.cosine_similarities(query)
            scores = _hybrid_combine(tfidf_scores, emb_scores, self._alpha)
        else:
            scores = tfidf_scores

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(ranked[:top_k], 1):
            results.append(VSMMatch(
                article=self._articles[idx],
                score=round(score, 6),
                rank=rank,
            ))
        return results

    def _tfidf_scores(self, query: str) -> list[float]:
        """全条文に対する TF-IDF cos 類似度を返す。条文と同じ順序。"""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        q_tf = Counter(q_tokens)
        q_total = len(q_tokens)
        q_tfidf = {
            t: (count / q_total) * self._idf.get(t, 1.0) for t, count in q_tf.items()
        }

        return [
            self._cosine_similarity(q_tfidf, doc_vec) for doc_vec in self._doc_tfidf
        ]

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


def _minmax_normalize(values: list[float]) -> list[float]:
    """値を [0, 1] に正規化する。すべて同値なら 0 を返す。"""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        return [0.0] * len(values)
    return [(v - lo) / span for v in values]


def _hybrid_combine(
    tfidf_scores: list[float],
    emb_scores: list[float],
    alpha: float,
) -> list[float]:
    """TF-IDF と embedding の cos を min-max 正規化してから alpha 加重和する。

    TF-IDF は 0.1〜0.4、embedding は 0.6〜0.8 のように分布が異なるため、
    生スコアを直接加重和すると常に embedding 側が支配する。条文内での
    相対順序を保ったまま [0, 1] に揃えてから混ぜる。
    """
    if len(tfidf_scores) != len(emb_scores):
        # embedding 側が空など想定外。TF-IDF にフォールバック。
        return tfidf_scores
    t_norm = _minmax_normalize(tfidf_scores)
    e_norm = _minmax_normalize(emb_scores)
    return [alpha * t + (1.0 - alpha) * e for t, e in zip(t_norm, e_norm)]


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

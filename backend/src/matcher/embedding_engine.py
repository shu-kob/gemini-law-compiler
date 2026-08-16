"""Vertex AI Embedding (text-multilingual-embedding-002) を使った条文ベクトル化。

VSMEngine がハイブリッドスコアを計算する際の補助モジュール。Layer 1 の
TF-IDF（2008 年式・語彙一致）に意味埋め込み（2026 年式・意味距離）を
合成することで、口語クエリと法令文体の語彙ギャップを救う。

- 初期化時に対象条文をまとめて embed し、メモリにキャッシュ
- search 時はクエリだけ embed し、cos 類似度を全条文に対して返す
- VSMEngine 側で TF-IDF cos と embedding cos を加重和して最終スコアにする
"""

from __future__ import annotations

import math

from src.config import GEMINI_PROVIDER
from src.parser.legal_compiler import ArticleNode, LawAST, flatten_article_text

DEFAULT_AI_STUDIO_EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_VERTEX_EMBEDDING_MODEL = "text-multilingual-embedding-002"

# 1 リクエストあたりの最大テキスト数。
# 小さめに切って遅延と失敗耐性のバランスを取る。
_BATCH_SIZE = 25

# 1 テキストあたりの最大文字数。条文は最大でも数千字なので 8000 で十分。
_MAX_CHARS_PER_TEXT = 8000


class EmbeddingEngine:
    """Gemini Embedding (AI Studio / Vertex AI) で条文をベクトル化し、cos 類似度を返すエンジン。"""

    def __init__(
        self,
        ast: LawAST,
        article_filter: list[ArticleNode] | None = None,
        model: str | None = None,
    ) -> None:
        self._articles = article_filter if article_filter is not None else ast.articles
        if model is None:
            model = (
                DEFAULT_VERTEX_EMBEDDING_MODEL
                if GEMINI_PROVIDER == "vertex"
                else DEFAULT_AI_STUDIO_EMBEDDING_MODEL
            )
        self._model = model
        self._client = None
        self._doc_embeddings: list[list[float]] = []
        self._build_index()


    @property
    def articles(self) -> list[ArticleNode]:
        return self._articles

    @property
    def doc_embeddings(self) -> list[list[float]]:
        return self._doc_embeddings

    def _get_client(self):
        if self._client is None:
            from src.config import get_genai_client
            self._client = get_genai_client()
        return self._client

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """テキストリストを embedding する。バッチ分割は内部処理。"""
        if not texts:
            return []
        client = self._get_client()
        all_embs: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = [t[:_MAX_CHARS_PER_TEXT] for t in texts[i : i + _BATCH_SIZE]]
            response = client.models.embed_content(
                model=self._model,
                contents=batch,
            )
            for emb in response.embeddings:
                all_embs.append(list(emb.values))
        return all_embs

    def _build_index(self) -> None:
        if not self._articles:
            return
        texts = [flatten_article_text(a) for a in self._articles]
        self._doc_embeddings = self._embed(texts)

    def cosine_similarities(self, query: str) -> list[float]:
        """クエリと全条文の cos 類似度を、条文と同じ順序で返す。"""
        if not self._doc_embeddings:
            return []
        q_emb = self._embed([query])[0]
        return [_cosine(q_emb, d_emb) for d_emb in self._doc_embeddings]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    from src.parser.legal_compiler import parse_egov_xml, extract_bicycle_articles
    from src.config import XML_PATH

    print("[Embedding-Engine]: 法規ASTを構築中...")
    ast = parse_egov_xml(XML_PATH)
    bicycle_articles = extract_bicycle_articles(ast)

    print(f"[Embedding-Engine]: Vertex AI Embedding でインデックス構築中... "
          f"({len(bicycle_articles)}条文)")
    engine = EmbeddingEngine(ast, article_filter=bicycle_articles)
    print(f"[Embedding-Engine]: 完了。ベクトル次元数: "
          f"{len(engine.doc_embeddings[0]) if engine.doc_embeddings else 0}")

    test_queries = [
        "自転車でスマホを操作しながら運転した",
        "イヤホンを両耳につけて自転車に乗った",
        "75歳の高齢者が歩道を自転車で走行した",
    ]

    for query in test_queries:
        sims = engine.cosine_similarities(query)
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n[Embedding-Engine]: クエリ「{query}」")
        for idx, score in ranked:
            a = engine.articles[idx]
            print(f"  cos={score:.4f} → {a.title} {a.caption}")

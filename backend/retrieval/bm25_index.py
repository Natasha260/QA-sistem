# backend/retrieval/bm25_index.py
import logging
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

Doc = Tuple[str, str]  # (doc_id, text)


class BM25Index:
    """In-memory BM25-индекс по корпусу текстов (чанки, вопросы)."""

    def __init__(self):
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._bm25: Optional[BM25Okapi] = None

    def initialize(self, docs: List[Doc]) -> None:
        self._ids = [doc_id for doc_id, _ in docs]
        self._texts = [text for _, text in docs]
        corpus = [t.lower().split() for t in self._texts]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @property
    def size(self) -> int:
        return len(self._ids)

    def search(self, query: str, top_n: int = 10) -> List[Tuple[str, str, float]]:
        """Возвращает [(doc_id, text, score)] по убыванию релевантности."""
        if self._bm25 is None or not query:
            return []

        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(
            zip(self._ids, self._texts, scores),
            key=lambda x: x[2],
            reverse=True,
        )
        hits = [(i, t, float(s)) for i, t, s in ranked if s > 0]
        return hits[:top_n]

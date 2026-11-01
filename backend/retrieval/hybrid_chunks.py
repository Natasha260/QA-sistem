# backend/retrieval/hybrid_chunks.py
"""Гибридный поиск по чанкам: BM25 + векторный поиск + RRF."""
import asyncio
import logging
from typing import Dict, List, Tuple

from backend.core.models import ChunkHit
from backend.data.documents import documents_manager
from backend.retrieval.chunk_vector import get_chunk_vector_index

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    bm25_results: List[ChunkHit],
    vector_results: List[ChunkHit],
    k: int = 60,
) -> List[ChunkHit]:
    """
    Reciprocal Rank Fusion по ключу чанка (document_id, chunk_index).

    Оценка каждого источника: 1 / (k + rank + 1). Итоговый score —
    сумма вкладов; текст и id берутся из любого из источников.
    """
    scores: Dict[Tuple[str, int], float] = {}
    samples: Dict[Tuple[str, int], ChunkHit] = {}

    for rank, hit in enumerate(bm25_results):
        key = (hit.document_id or "", hit.chunk_index or 0)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        samples.setdefault(key, hit)

    for rank, hit in enumerate(vector_results):
        key = (hit.document_id or "", hit.chunk_index or 0)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        samples.setdefault(key, hit)

    sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results: List[ChunkHit] = []
    for key, score in sorted_keys:
        sample = samples[key]
        results.append(ChunkHit(
            id=sample.id,
            document_id=key[0] or None,
            chunk_index=key[1],
            title=sample.title,
            text=sample.text,
            score=score,
        ))
    return results


async def hybrid_chunk_search(
    query: str,
    top_n: int = 5,
) -> List[ChunkHit]:
    """BM25 по чанкам + векторный поиск + RRF. Fallback на BM25."""
    bm25_results = await documents_manager.search_bm25(
        query, top_n=top_n * 2
    )

    try:
        vector_results = await asyncio.to_thread(
            get_chunk_vector_index().search, query, top_n * 2
        )
    except Exception as e:
        logger.warning("Vector chunk search failed, using BM25 only: %s", e)
        vector_results = []

    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    return fused[:top_n]

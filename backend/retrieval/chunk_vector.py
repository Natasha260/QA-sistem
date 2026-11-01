# backend/retrieval/chunk_vector.py
"""Векторный индекс чанков учебников (Qdrant + sentence-transformers)."""
import asyncio
import logging
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from backend.core.config import get_settings
from backend.core.models import ChunkHit
from backend.data.embeddings import embedding_model

logger = logging.getLogger(__name__)

_POINT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
_DELETE_RETRIES = 3


class ChunkVectorIndex:
    """Косинусный поиск по чанкам документов через Qdrant."""

    def __init__(self, host=None, port=None, collection_name=None):
        settings = get_settings()
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_name = (
            collection_name or settings.QDRANT_COLLECTION_NAME
        )
        self.api_key = settings.QDRANT_API_KEY
        self.client = None

    def _ensure_connection_and_collection(self, retries=5, delay=2):
        if self.client is not None:
            return
        url = f"http://{self.host}:{self.port}"
        client_kwargs = {"url": url, "timeout": 10.0, "check_compatibility": False}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        for attempt in range(1, retries + 1):
            try:
                self.client = QdrantClient(**client_kwargs)
                self.client.get_collections()
                self._ensure_collection()
                return
            except Exception as exc:
                logger.warning(
                    "Qdrant connection failed (attempt %d/%d): %s",
                    attempt, retries, exc,
                )
                self.client = None
                if attempt < retries:
                    import time
                    time.sleep(delay)
        logger.error("Could not connect to Qdrant after %d attempts", retries)

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_model.dimension,
                    distance=Distance.COSINE,
                ),
            )

    def recreate_collection(self) -> None:
        self._ensure_connection_and_collection()
        if self.client is None:
            return
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            logger.warning("Delete collection failed: %s", e)
        self._ensure_collection()

    @staticmethod
    def _point_id(text: str, document_id: str, chunk_index: int) -> str:
        key = f"{document_id}|{chunk_index}|{text}"
        return str(uuid.uuid5(_POINT_NAMESPACE, key))

    def add_chunks(self, chunks: List[dict]) -> None:
        """chunks: [{id, document_id, chunk_index, text}]"""
        if not chunks:
            return
        self._ensure_connection_and_collection()
        if self.client is None:
            logger.warning("Qdrant unavailable; skipping vector addition")
            return
        try:
            vectors = embedding_model.encode(
                [c["text"] for c in chunks], normalize=True
            )
            points = [
                PointStruct(
                    id=self._point_id(
                        c["text"], c["document_id"], c.get("chunk_index", 0)
                    ),
                    vector=vector,
                    payload={
                        "id": c.get("id"),
                        "document_id": c["document_id"],
                        "chunk_index": c.get("chunk_index", 0),
                        "text": c["text"],
                    },
                )
                for c, vector in zip(chunks, vectors)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as e:
            logger.error("Failed to add chunk vectors: %s", e)

    def delete_chunks(self, document_id: str) -> None:
        self._ensure_connection_and_collection()
        if self.client is None:
            return
        points_filter = Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=document_id)),
        ])
        for attempt in range(1, _DELETE_RETRIES + 1):
            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=points_filter,
                )
                return
            except Exception as e:
                logger.warning("Qdrant delete failed (attempt %d): %s", attempt, e)
        logger.error("Failed to delete chunk vectors for doc %s", document_id)

    def search(self, query: str, top_n: int = 5) -> List[ChunkHit]:
        self._ensure_connection_and_collection()
        if self.client is None:
            logger.warning("Qdrant unavailable; returning empty results")
            return []
        try:
            query_vector = embedding_model.encode(query)[0]
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_n,
                with_payload=True,
            )
            hits = []
            for hit in result.points:
                payload = hit.payload or {}
                hits.append(ChunkHit(
                    id=payload.get("id"),
                    document_id=payload.get("document_id"),
                    chunk_index=payload.get("chunk_index"),
                    text=payload.get("text", ""),
                    score=hit.score,
                ))
            return hits
        except Exception as e:
            logger.error("Chunk vector search failed: %s", e)
            return []


_chunk_vector_index = None


def get_chunk_vector_index() -> ChunkVectorIndex:
    global _chunk_vector_index
    if _chunk_vector_index is None:
        _chunk_vector_index = ChunkVectorIndex()
    return _chunk_vector_index

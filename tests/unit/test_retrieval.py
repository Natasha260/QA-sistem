# tests/unit/test_retrieval.py
"""Юнит-тесты чанкинга, BM25-индекса, RRF и уверенности."""
import pytest

from backend.core.models import ChunkHit
from backend.data.documents import chunk_text
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.hybrid_chunks import reciprocal_rank_fusion
from backend.eval.runner import Evaluator, exact_match, token_f1
from backend.services.pipeline import compute_confidence


class TestChunking:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Привет, мир.")
        assert len(chunks) == 1
        assert chunks[0] == "Привет, мир."

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   \n  ") == []

    def test_long_text_splits_with_overlap(self):
        paragraph = "Абзац про механику. " * 50  # ~1100 символов
        text = paragraph + "\n\n" + paragraph
        chunks = chunk_text(text, chunk_size=600, overlap=100)
        assert len(chunks) >= 3

    def test_headings_split(self):
        text = "Глава 1. Механика\n\nСодержание главы.\n\nГлава 2. Оптика\n\nСвет распространяется."
        chunks = chunk_text(text)
        assert any("Механика" in c for c in chunks)
        assert any("Оптика" in c for c in chunks)


class TestBM25Index:
    def _index(self):
        idx = BM25Index()
        idx.initialize([
            ("c1", "сила является мерой взаимодействия тел"),
            ("c2", "ускорение это быстрота изменения скорости"),
            ("c3", "энергия сохраняется в замкнутой системе"),
        ])
        return idx

    def test_search_finds_relevant(self):
        hits = self._index().search("взаимодействие тел", top_n=2)
        assert hits and hits[0][0] == "c1"

    def test_search_scores_positive(self):
        hits = self._index().search("скорость", top_n=3)
        assert all(s > 0 for _, _, s in hits)

    def test_empty_query(self):
        assert self._index().search("") == []

    def test_uninitialized(self):
        assert BM25Index().search("всё") == []


def _hit(doc_id, chunk_index, text):
    return ChunkHit(
        id=f"{doc_id}-{chunk_index}", document_id=doc_id,
        chunk_index=chunk_index, text=text,
    )


class TestRRF:
    def test_single_source(self):
        bm25 = [_hit("d", 0, "a"), _hit("d", 1, "b")]
        fused = reciprocal_rank_fusion(bm25, [])
        assert [f.chunk_index for f in fused] == [0, 1]

    def test_intersection_first(self):
        bm25 = [_hit("d", 0, "a"), _hit("d", 1, "b"), _hit("d", 2, "c")]
        vector = [_hit("d", 1, "b"), _hit("d", 3, "e")]
        fused = reciprocal_rank_fusion(bm25, vector)
        assert fused[0].chunk_index == 1
        assert fused[0].score > fused[1].score

    def test_both_empty(self):
        assert reciprocal_rank_fusion([], []) == []


class TestConfidence:
    def test_full_scores(self):
        c = compute_confidence(critic_score=10.0, validation_passed=True,
                               max_retrieval_score=1.0)
        assert c == 1.0

    def test_no_critic(self):
        c = compute_confidence(None, True, 0.8)
        assert 0 <= c <= 1

    def test_all_none(self):
        assert compute_confidence(None, None, None) is None

    def test_fail_lowers_confidence(self):
        high = compute_confidence(10.0, True, 1.0)
        low = compute_confidence(0.0, False, 0.0)
        assert low < high


class TestQAMetrics:
    def test_exact_match(self):
        assert exact_match("Сила 2 Н", "сила 2 н") == 1.0
        assert exact_match("Сила 2 Н", "Сила 5 Н") == 0.0

    def test_f1_perfect(self):
        assert token_f1("мера взаимодействия тел", "мера взаимодействия тел") == 1.0

    def test_f1_partial(self):
        f1 = token_f1("мера взаимодействия", "мера взаимодействия тел")
        assert 0 < f1 < 1

    def test_evaluator(self):
        ev = Evaluator()
        res = ev.evaluate(["мера взаимодействия тел"], ["мера взаимодействия тел"])
        assert res["exact_match"] == 100.0
        assert res["f1"] == 100.0
        assert res["n"] == 1

    def test_evaluator_length_mismatch(self):
        with pytest.raises(ValueError):
            Evaluator().evaluate(["a"], [])

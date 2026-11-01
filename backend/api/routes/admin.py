# backend/api/routes/admin.py
import csv
import io
import logging
from typing import Any, Dict, List, Optional, Type

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import require_admin
from backend.core.exceptions import ImportFormatError, RUNGException
from backend.core.models import DocumentCreate, DocumentInfo, TerminologyEntry
from backend.data.cache import cache_manager
from backend.data.documents import documents_manager
from backend.data.glossary import glossary_manager
from backend.data.verified_qa import verified_qa_manager
from backend.db.models import TranslationCache
from backend.db.session import get_async_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

_TXT_EXTENSIONS = (".txt", ".md", ".csv")


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def add_document(
    doc: DocumentCreate,
    _=Depends(require_admin),
):
    try:
        result = await documents_manager.add_document(
            title=doc.title,
            text=doc.text,
            filename=doc.filename,
            description=doc.description,
        )
        return {"status": "ok", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RUNGException:
        raise
    except Exception as e:
        logger.error("Failed to add document: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/import", status_code=status.HTTP_201_CREATED)
async def import_document_file(
    file: UploadFile = File(..., description="Text file (.txt/.md/.csv)"),
    title: Optional[str] = None,
    description: Optional[str] = None,
    _=Depends(require_admin),
):
    try:
        filename = file.filename or ""
        if not filename.lower().endswith(_TXT_EXTENSIONS):
            raise ImportFormatError(
                f"File must be one of {_TXT_EXTENSIONS}"
            )
        content = await file.read()
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ImportFormatError("File must be UTF-8 encoded")

        result = await documents_manager.add_document(
            title=title or filename,
            text=text,
            filename=filename,
            description=description,
        )
        return {"status": "ok", **result}
    except ImportFormatError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RUNGException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document import failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Import failed: {e}")


@router.get("/documents")
async def list_documents(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    _=Depends(require_admin),
):
    try:
        docs = await documents_manager.list_documents(limit=limit, offset=offset)
        total = await documents_manager.count_documents()
        return {"documents": docs, "total": total, "limit": limit, "offset": offset}
    except RUNGException:
        raise
    except Exception as e:
        logger.error("Failed to list documents: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/chunks")
async def list_document_chunks(
    document_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    _=Depends(require_admin),
):
    try:
        chunks = await documents_manager.list_chunks(
            document_id, limit=limit, offset=offset
        )
        return {"chunks": chunks, "limit": limit, "offset": offset}
    except RUNGException:
        raise
    except Exception as e:
        logger.error("Failed to list chunks: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, _=Depends(require_admin)):
    try:
        deleted = await documents_manager.delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "ok", "message": "Deleted"}
    except RUNGException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Terminology
# ---------------------------------------------------------------------------

def _terminology_entry_to_dict(entry: TerminologyEntry) -> Dict[str, str]:
    return {
        "subject": entry.subject or "general",
        "term": entry.term,
        "definition": entry.definition,
    }


@router.post("/terminology/add", status_code=status.HTTP_201_CREATED)
async def add_terminology_entry(
    entry: TerminologyEntry,
    _=Depends(require_admin),
):
    try:
        entry_id = await glossary_manager.add_entry(
            _terminology_entry_to_dict(entry)
        )
        if entry_id is None:
            return {"status": "skipped", "message": "Entry already exists"}
        return {"status": "ok", "id": entry_id, "message": f"Added: {entry.term}"}
    except RUNGException:
        raise
    except Exception as e:
        logger.error("Failed to add terminology entry: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/terminology/import-csv", status_code=status.HTTP_201_CREATED)
async def import_terminology_csv(
    file: UploadFile = File(..., description=(
        "CSV with columns: term,definition[,subject]"
    )),
    _=Depends(require_admin),
):
    try:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise ImportFormatError("File must be CSV")
        content = await file.read()
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ImportFormatError("File must be UTF-8 encoded")

        reader = csv.DictReader(io.StringIO(text))
        fieldnames = set(reader.fieldnames or [])
        if not {"term", "definition"}.issubset(fieldnames):
            raise ImportFormatError(
                "CSV must contain columns: term, definition"
            )

        added = 0
        skipped = 0
        errors: List[str] = []
        for row_num, row in enumerate(reader, start=2):
            term = (row.get("term") or "").strip()
            definition = (row.get("definition") or "").strip()
            subject = (row.get("subject") or "general").strip()
            if not term or not definition:
                errors.append(f"Row {row_num}: empty term or definition")
                skipped += 1
                continue
            try:
                entry_id = await glossary_manager.add_entry({
                    "subject": subject, "term": term, "definition": definition,
                })
                if entry_id:
                    added += 1
                else:
                    skipped += 1
            except Exception as e:
                errors.append(f"Row {row_num}: {e}")
                skipped += 1

        return {"status": "ok", "added": added, "skipped": skipped, "errors": errors[:20]}
    except ImportFormatError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RUNGException:
        raise
    except Exception as e:
        logger.error("Terminology CSV import failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Import failed: {e}")


@router.get("/terminology/list")
async def list_terminology(
    subject: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    _=Depends(require_admin),
):
    try:
        entries = await glossary_manager.list_all(
            subject, limit=limit, offset=offset,
        )
        total = await glossary_manager.count()
        return {"entries": entries, "total": total, "limit": limit, "offset": offset}
    except RUNGException:
        raise
    except Exception as e:
        logger.error("Failed to list terminology: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/terminology/delete/{entry_id}")
async def delete_terminology_entry(entry_id: str, _=Depends(require_admin)):
    try:
        deleted = await glossary_manager.delete_entry(entry_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Entry not found")
        return {"status": "ok", "message": "Deleted"}
    except RUNGException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete terminology entry: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Verified QA pairs
# ---------------------------------------------------------------------------

@router.post("/qa-pairs/add", status_code=status.HTTP_201_CREATED)
async def add_qa_pair(
    question: str = Query(..., min_length=1),
    answer: str = Query(..., min_length=1),
    _=Depends(require_admin),
):
    try:
        pair = await verified_qa_manager.add_pair(question, answer)
        if pair is None:
            return {"status": "skipped", "message": "Entry already exists"}
        return {"status": "ok", "id": pair.id, "message": "Added"}
    except RUNGException:
        raise
    except Exception as e:
        logger.error("Failed to add QA pair: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/qa-pairs/list")
async def list_qa_pairs(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    _=Depends(require_admin),
):
    try:
        entries = await verified_qa_manager.list_all(limit=limit, offset=offset)
        total = await verified_qa_manager.count()
        return {"entries": entries, "total": total, "limit": limit, "offset": offset}
    except RUNGException:
        raise
    except Exception as e:
        logger.error("Failed to list QA pairs: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/qa-pairs/delete/{pair_id}")
async def delete_qa_pair(pair_id: str, _=Depends(require_admin)):
    try:
        deleted = await verified_qa_manager.delete_pair(pair_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Pair not found")
        return {"status": "ok", "message": "Deleted"}
    except RUNGException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete QA pair: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@router.delete("/cache/clear")
async def clear_cache(
    eval_mode: Optional[str] = None,
    _=Depends(require_admin),
):
    try:
        deleted = await cache_manager.clear_cache(eval_mode)
        return {"status": "ok", "deleted": deleted}
    except RUNGException:
        raise
    except Exception as e:
        logger.error("Failed to clear cache: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def cache_stats(
    session: AsyncSession = Depends(get_async_session),
    _=Depends(require_admin),
):
    try:
        total = await session.scalar(
            select(func.count()).select_from(TranslationCache)
        )
        total_hits = await session.scalar(
            select(func.sum(TranslationCache.hits))
        ) or 0
        return {"total_entries": total, "total_hits": total_hits}
    except RUNGException:
        raise
    except Exception as e:
        logger.error("Failed to get cache stats: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

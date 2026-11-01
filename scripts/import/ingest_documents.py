#!/usr/bin/env python3
"""
Загрузка учебников в QA-систему.

Читает текстовые файлы из каталога (по умолчанию data/textbooks/)
и загружает их через documents_manager с чанкингом и индексацией.

Usage:
    python scripts/import/ingest_documents.py                 # все .txt/.md из data/textbooks/
    python scripts/import/ingest_documents.py path/to/book.md --title "Физика"
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.database import init_db, close_db  # noqa: E402
from backend.data.documents import documents_manager  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ingest_documents")

TEXT_EXTENSIONS = (".txt", ".md")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", default=str(ROOT / "data" / "textbooks"),
        help="Файл или каталог с учебниками (.txt/.md)",
    )
    parser.add_argument("--title", help="Название документа (для одиночного файла)")
    parser.add_argument("--description", help="Описание документа")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            f for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in TEXT_EXTENSIONS
        )
    if not files:
        logger.error("No text files found in %s", path)
        return

    await init_db()
    try:
        for f in files:
            title = args.title if (args.title and f == files[0]) else f.stem
            text = f.read_text(encoding="utf-8-sig")
            result = await documents_manager.add_document(
                title=title, text=text,
                filename=f.name, description=args.description,
            )
            logger.info(
                "Ingested '%s': %d chunks", result["title"], result["chunks"],
            )
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())

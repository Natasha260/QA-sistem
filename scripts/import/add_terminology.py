#!/usr/bin/env python3
"""
Загрузка терминологии из CSV (колонки: term,definition,subject).

Usage:
    python scripts/import/add_terminology.py data/import/terminology.csv
"""
import argparse
import asyncio
import csv
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.database import init_db, close_db  # noqa: E402
from backend.data.glossary import glossary_manager  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("add_terminology")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="CSV с колонками term,definition,subject")
    args = parser.parse_args()

    await init_db()
    try:
        with open(args.csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            added = 0
            for row in reader:
                term = (row.get("term") or "").strip()
                definition = (row.get("definition") or "").strip()
                subject = (row.get("subject") or "general").strip()
                if not term or not definition:
                    continue
                entry_id = await glossary_manager.add_entry({
                    "subject": subject, "term": term, "definition": definition,
                })
                if entry_id:
                    added += 1
        logger.info("Inserted %d terminology entries", added)
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())

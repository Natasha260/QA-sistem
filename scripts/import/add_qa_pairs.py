#!/usr/bin/env python3
"""
Загрузка подтверждённых пар «вопрос → ответ» из JSON/CSV.

JSON-формат: {"pairs": [{"question": ..., "answer": ...}]}
CSV-формат: колонки question,answer.

Usage:
    python scripts/import/add_qa_pairs.py data/import/verified_qa.json
"""
import argparse
import asyncio
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.database import init_db, close_db  # noqa: E402
from backend.data.verified_qa import verified_qa_manager  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("add_qa_pairs")


def load_pairs(path: Path):
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data.get("pairs", data if isinstance(data, list) else [])
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="JSON- или CSV-файл с парами")
    args = parser.parse_args()

    pairs = load_pairs(Path(args.path))
    await init_db()
    try:
        added = await verified_qa_manager.add_many([
            {
                "question": (p.get("question") or "").strip(),
                "answer": (p.get("answer") or "").strip(),
            }
            for p in pairs
        ])
        logger.info("Inserted %d verified QA pairs", added)
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())

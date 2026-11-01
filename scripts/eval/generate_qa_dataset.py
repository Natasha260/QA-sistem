#!/usr/bin/env python3
"""
Генерация QA eval-датасетов (офлайн, без HF).

Пары «вопрос → эталонный ответ» берутся из data/import/verified_qa.json
и раскладываются на наборы размеров 10 / 20 / all в eval_datasets/qa/.

Usage:
    python scripts/eval/generate_qa_dataset.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "import" / "verified_qa.json"
OUT = ROOT / "eval_datasets" / "qa"
SIZES = (10, 20, None)


def main():
    data = json.loads(SRC.read_text(encoding="utf-8-sig"))
    pairs = [
        (p["question"], p["answer"])
        for p in data["pairs"]
        if p.get("question") and p.get("answer")
    ]
    random.seed(42)
    random.shuffle(pairs)

    OUT.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        name = f"qa_{size}.json" if size else "qa_full.json"
        sample = pairs[:size] if size else pairs
        payload = {
            "name": f"qa_{size or 'full'}",
            "questions": [q for q, _ in sample],
            "references": [a for _, a in sample],
            "metadata": {"seed": 42, "n": len(sample), "synthetic": True},
        }
        path = OUT / name
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {path} ({len(sample)} pairs)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prepare evaluation datasets in the format consumed by scripts/eval/run_eval.py.

Usage:
    export HF_TOKEN=hf_...
    python scripts/eval/prepare_eval_datasets.py

Optional env:
    HF_TOKEN              Hugging Face token (required for gated datasets)
    EVAL_OUTPUT_DIR       output directory (default: ./eval_datasets)
    EVAL_SEED             sampling seed (default: 42)
    EVAL_SIZES            comma-separated sizes (default: 10,20,50,100,200,500,1000)
    EVAL_PAIRS            comma-separated pair names (default: all)
    EVAL_MAX_SRC_LEN      max length of source text; 0 = no limit (default: 600)
    EVAL_MAX_TGT_LEN      max length of reference text; 0 = no limit (default: 0)
    EVAL_NEWS_MIN_KEEP    min number of "news" examples to keep even if long
                          (shortest ones are picked). 0 = disable. (default: 3)
    EVAL_NEWS_ABS_MAX_LEN hard cap for forced news picks; 0 = no cap
                          (default: 2000)

Output JSON schema:
    {
        "sources":     [...],
        "references":  [...],
        "domains":     [...],
        "source_lang": "ar",
        "target_lang": "ru",
        "metadata":    {...}
    }
"""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from datasets import load_dataset
except ImportError:
    print("Install datasets: pip install datasets", file=sys.stderr)
    sys.exit(1)


# ----------------------------- CONFIG --------------------------------------
def _env_int_list(name: str, default: List[int]) -> List[int]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _env_str_list(name: str, default: Optional[List[str]]) -> Optional[List[str]]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


HF_TOKEN = os.getenv("HF_TOKEN") or None
OUTPUT_DIR = Path(os.getenv("EVAL_OUTPUT_DIR", "./eval_datasets"))
RANDOM_SEED = _env_int("EVAL_SEED", 42)
SAMPLE_SIZES = _env_int_list("EVAL_SIZES", [10, 20, 50, 100, 200, 500, 1000])
PAIRS_FILTER = _env_str_list("EVAL_PAIRS", None)
MAX_SRC_LEN = _env_int("EVAL_MAX_SRC_LEN", 600)  # 0 = no limit
MAX_TGT_LEN = _env_int("EVAL_MAX_TGT_LEN", 0)  # 0 = no limit
NEWS_MIN_KEEP = _env_int("EVAL_NEWS_MIN_KEEP", 3)  # 0 = disable
NEWS_ABS_MAX_LEN = _env_int("EVAL_NEWS_ABS_MAX_LEN", 2000)

CONFIGS: List[Dict] = [
    {
        "name": "ar_ru",
        "hf_dataset": "ArabicNLPWorld/arabic-russian-parallel-corpus",
        "split": "train",
        "source_lang": "ar",
        "target_lang": "ru",
        "source_col": "arabic",
        "target_col": "russian",
        "domain_col": "source",
        "domain_map": {
            "quran": "scripture",
            "bible": "scripture",
            "hadith": "hadith",
            "news": "news",
            "wiki": "encyclopedic",
            "books": "prose",
            "poetry": "poetry",
        },
        "default_domain": "prose",
    },
    {
        "name": "ru_tt",
        "hf_dataset": "TatarNLPWorld/tatar-english-russian-corpus",
        "split": "full",
        "source_lang": "ru",
        "target_lang": "tt",
        "source_col": "russian_text",
        "target_col": "tatar_text",
        "domain_col": None,
        "domain_map": {},
        "default_domain": "prose",
    },
    {
        "name": "tg_fa",
        "hf_dataset": "TajikNLPWorld/TajPersParallelCorpusFull",
        "split": "train",
        "source_lang": "tg",
        "target_lang": "fa",
        "source_col": "tajik",
        "target_col": "farsi",
        "domain_col": "category",
        "domain_map": {},
        "default_domain": "prose",
    },
    {
        "name": "ru_ba",
        "hf_dataset": "BorisTM/bak-rus-parallel",
        "split": "train",
        "source_lang": "ru",
        "target_lang": "ba",
        "source_col": "rus",
        "target_col": "bak",
        "domain_col": None,
        "domain_map": {},
        "default_domain": "prose",
    },
]


# --------------------------- DOMAIN HELPERS --------------------------------
_DOMAIN_ALIASES = {
    "quran": "scripture",
    "koran": "scripture",
    "bible": "scripture",
    "gospel": "scripture",
    "torah": "scripture",
    "hadith": "hadith",
    "sunnah": "hadith",
    "poem": "poetry",
    "poetry": "poetry",
    "verse": "poetry",
    "news": "news",
    "article": "news",
    "wiki": "encyclopedic",
    "wikipedia": "encyclopedic",
    "legal": "legal",
    "law": "legal",
    "medical": "medical",
    "tech": "technical",
    "it": "technical",
}


def normalize_domain(raw: Optional[str], default: str) -> str:
    if not raw:
        return default
    key = str(raw).strip().lower()
    if key in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[key]
    for alias, canonical in _DOMAIN_ALIASES.items():
        if alias in key:
            return canonical
    return default


def detect_column(ds, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in ds.column_names:
            return col
    return None


# ------------------------------- LOADING -----------------------------------
def load_split(cfg: Dict):
    kwargs = {"split": cfg["split"]}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    ds = load_dataset(cfg["hf_dataset"], **kwargs)
    src_col = detect_column(ds, [cfg["source_col"], "source", "src"])
    tgt_col = detect_column(ds, [cfg["target_col"], "target", "tgt", "ref"])
    if src_col is None or tgt_col is None:
        raise RuntimeError(
            f"[{cfg['name']}] cannot detect columns; "
            f"available: {ds.column_names}"
        )
    domain_col = None
    if cfg.get("domain_col"):
        domain_col = detect_column(ds, [cfg["domain_col"]])
    return ds, src_col, tgt_col, domain_col


def _passes_length_filter(l_src: int, l_tgt: int) -> bool:
    if MAX_SRC_LEN and l_src > MAX_SRC_LEN:
        return False
    if MAX_TGT_LEN and l_tgt > MAX_TGT_LEN:
        return False
    return True


def extract_pairs(
    ds,
    src_col: str,
    tgt_col: str,
    domain_col: Optional[str],
    cfg: Dict,
) -> Dict:
    default_domain = cfg.get("default_domain", "prose")
    domain_map = cfg.get("domain_map", {}) or {}
    n_raw = len(ds)
    n_empty = 0
    all_rows: List[Tuple[str, str, str, int, int]] = []

    for row in ds:
        src = (row.get(src_col) or "").strip()
        tgt = (row.get(tgt_col) or "").strip()
        if not src or not tgt:
            n_empty += 1
            continue
        if domain_col is not None:
            raw_domain = row.get(domain_col)
            if raw_domain and raw_domain in domain_map:
                domain = domain_map[raw_domain]
            else:
                domain = normalize_domain(raw_domain, default_domain)
        else:
            domain = default_domain
        all_rows.append((src, tgt, domain, len(src), len(tgt)))

    news_rows = [r for r in all_rows if r[2] == "news"]
    other_rows = [r for r in all_rows if r[2] != "news"]
    other_kept = [r for r in other_rows if _passes_length_filter(r[3], r[4])]
    news_kept = [r for r in news_rows if _passes_length_filter(r[3], r[4])]

    n_filtered_other = len(other_rows) - len(other_kept)
    n_filtered_news = len(news_rows) - len(news_kept)

    news_forced = 0
    news_forced_skipped = 0
    if NEWS_MIN_KEEP > 0 and news_rows and len(news_kept) < NEWS_MIN_KEEP:
        candidates = sorted(news_rows, key=lambda r: min(r[3], r[4]))
        already = set(id(r) for r in news_kept)
        for r in candidates:
            if len(news_kept) >= NEWS_MIN_KEEP:
                break
            if id(r) in already:
                continue
            if NEWS_ABS_MAX_LEN and (
                r[3] > NEWS_ABS_MAX_LEN or r[4] > NEWS_ABS_MAX_LEN
            ):
                news_forced_skipped += 1
                continue
            news_kept.append(r)
            news_forced += 1

    kept = other_kept + news_kept
    sources = [r[0] for r in kept]
    references = [r[1] for r in kept]
    domains = [r[2] for r in kept]

    print(
        f"    kept {len(sources):,} / {n_raw:,} "
        f"(empty={n_empty:,}, filtered_by_len={n_filtered_other + n_filtered_news:,})"
    )
    if MAX_SRC_LEN or MAX_TGT_LEN:
        print(
            f"    length filter: src<={MAX_SRC_LEN or 'inf'}, "
            f"tgt<={MAX_TGT_LEN or 'inf'}"
        )
    if news_rows:
        print(
            f"    news: {len(news_rows):,} raw -> {len(news_kept):,} kept "
            f"(forced={news_forced}, skipped_by_cap={news_forced_skipped}, "
            f"min_keep={NEWS_MIN_KEEP}, abs_max={NEWS_ABS_MAX_LEN or 'inf'})"
        )
    print(f"    domains: {dict(Counter(domains).most_common())}")

    stats = {
        "n_raw": n_raw,
        "n_empty": n_empty,
        "n_filtered_by_len": n_filtered_other + n_filtered_news,
        "length_filter": {
            "max_src_len": MAX_SRC_LEN or None,
            "max_tgt_len": MAX_TGT_LEN or None,
        },
        "news": {
            "raw": len(news_rows),
            "kept_after_filter": len(news_kept) - news_forced,
            "forced_shortest": news_forced,
            "skipped_by_cap": news_forced_skipped,
            "min_keep": NEWS_MIN_KEEP,
            "abs_max_len": NEWS_ABS_MAX_LEN or None,
        },
    }

    return {
        "sources": sources,
        "references": references,
        "domains": domains,
        "stats": stats,
    }


# ---------------------------- SAMPLING -------------------------------------
def make_sample(data: Dict, size: int, seed: int) -> Dict:
    total = len(data["sources"])
    if size >= total:
        return {
            "sources": list(data["sources"]),
            "references": list(data["references"]),
            "domains": list(data["domains"]),
        }

    rng = random.Random(seed)
    by_domain: Dict[str, List[int]] = {}
    for idx, dom in enumerate(data["domains"]):
        by_domain.setdefault(dom, []).append(idx)

    for idxs in by_domain.values():
        rng.shuffle(idxs)

    domains_sorted = sorted(by_domain.keys(), key=lambda d: -len(by_domain[d]))
    n_domains = len(domains_sorted)

    quota = {d: 0 for d in domains_sorted}
    remaining = size
    for d in domains_sorted:
        take = min(len(by_domain[d]), max(1, size // n_domains))
        quota[d] = take
        remaining -= take

    while remaining > 0:
        progressed = False
        for d in domains_sorted:
            if remaining == 0:
                break
            if quota[d] < len(by_domain[d]):
                quota[d] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    selected: List[int] = []
    for d in domains_sorted:
        selected.extend(by_domain[d][: quota[d]])
    rng.shuffle(selected)

    return {
        "sources": [data["sources"][i] for i in selected],
        "references": [data["references"][i] for i in selected],
        "domains": [data["domains"][i] for i in selected],
    }


# ----------------------------- WRITING -------------------------------------
def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def process_config(cfg: Dict, out_dir: Path) -> Dict:
    name = cfg["name"]
    print("=" * 70)
    print(f"[{name}] loading {cfg['hf_dataset']} (split={cfg['split']})")
    print("=" * 70)

    ds, src_col, tgt_col, domain_col = load_split(cfg)

    DATASET_LIMIT = 10000
    if len(ds) > DATASET_LIMIT:
        print(f"  Limiting dataset to {DATASET_LIMIT:,} rows (from {len(ds):,})")
        ds = ds.select(range(DATASET_LIMIT))
    
    print(f"  total rows: {len(ds):,}")
    print(f"  src_col={src_col!r}, tgt_col={tgt_col!r}, domain_col={domain_col!r}")

    data = extract_pairs(ds, src_col, tgt_col, domain_col, cfg)
    stats = data["stats"]

    pair_dir = out_dir / name
    pair_dir.mkdir(parents=True, exist_ok=True)

    full_payload = {
        "sources": data["sources"],
        "references": data["references"],
        "domains": data["domains"],
        "source_lang": cfg["source_lang"],
        "target_lang": cfg["target_lang"],
        "metadata": {
            "name": name,
            "hf_dataset": cfg["hf_dataset"],
            "split": cfg["split"],
            "total_pairs": len(data["sources"]),
            "domain_counts": dict(Counter(data["domains"])),
            "seed": RANDOM_SEED,
            "sample_size": "full",
            "stats": stats,
        },
    }
    full_path = pair_dir / f"{name}_full.json"
    save_json(full_path, full_payload)
    print(f"  wrote {full_path} ({len(data['sources']):,} pairs)")

    sample_meta = {}
    for size in SAMPLE_SIZES:
        sample = make_sample(data, size, RANDOM_SEED)
        payload = {
            "sources": sample["sources"],
            "references": sample["references"],
            "domains": sample["domains"],
            "source_lang": cfg["source_lang"],
            "target_lang": cfg["target_lang"],
            "metadata": {
                "name": name,
                "hf_dataset": cfg["hf_dataset"],
                "split": cfg["split"],
                "total_pairs": len(data["sources"]),
                "domain_counts": dict(Counter(sample["domains"])),
                "seed": RANDOM_SEED,
                "sample_size": size,
                "stats": stats,
            },
        }
        sample_path = pair_dir / f"{name}_{size}.json"
        save_json(sample_path, payload)
        sample_meta[str(size)] = {
            "file": str(sample_path.relative_to(out_dir)),
            "num_examples": len(sample["sources"]),
            "domain_counts": dict(Counter(sample["domains"])),
        }
        print(
            f"  wrote {sample_path.name}: n={len(sample['sources'])}, "
            f"domains={dict(Counter(sample['domains']))}"
        )

    return {
        "name": name,
        "source_lang": cfg["source_lang"],
        "target_lang": cfg["target_lang"],
        "hf_dataset": cfg["hf_dataset"],
        "split": cfg["split"],
        "total_pairs": len(data["sources"]),
        "domain_counts": dict(Counter(data["domains"])),
        "full_file": str(full_path.relative_to(out_dir)),
        "samples": sample_meta,
        "stats": stats,
    }


# -------------------------------- MAIN -------------------------------------
def main() -> None:
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN is not set. Gated datasets will fail.")

    print(f"Length filter: src<={MAX_SRC_LEN or 'inf'}, " f"tgt<={MAX_TGT_LEN or 'inf'}")
    print(f"News min keep: {NEWS_MIN_KEEP} " f"(abs max len: {NEWS_ABS_MAX_LEN or 'inf'})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    configs = CONFIGS
    if PAIRS_FILTER:
        configs = [c for c in CONFIGS if c["name"] in PAIRS_FILTER]
        print(f"Filtering pairs: {[c['name'] for c in configs]}")

    for cfg in configs:
        try:
            info = process_config(cfg, OUTPUT_DIR)
            summary.append(info)
        except Exception as e:
            print(f"[{cfg['name']}] FAILED: {e}")
            summary.append({"name": cfg["name"], "error": str(e)})

    manifest = {
        "sample_sizes": SAMPLE_SIZES,
        "seed": RANDOM_SEED,
        "length_filter": {
            "max_src_len": MAX_SRC_LEN or None,
            "max_tgt_len": MAX_TGT_LEN or None,
        },
        "news_policy": {
            "min_keep": NEWS_MIN_KEEP,
            "abs_max_len": NEWS_ABS_MAX_LEN or None,
        },
        "pairs": summary,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    save_json(manifest_path, manifest)

    print("=" * 70)
    print(f"wrote {manifest_path}")
    print("=" * 70)
    for info in summary:
        if "error" in info:
            print(f"  {info['name']}: FAILED - {info['error']}")
        else:
            print(
                f"  {info['name']}: {info['total_pairs']:,} pairs, "
                f"{len(info['samples'])} samples, "
                f"domains={info['domain_counts']}"
            )


if __name__ == "__main__":
    main()
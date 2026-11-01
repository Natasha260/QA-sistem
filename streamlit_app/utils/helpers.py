# streamlit_app/utils/helpers.py


def entry_id(entry: dict) -> str:
    if entry.get("id"):
        return entry["id"]
    return (
        f"{entry['source_lang']}|{entry['target_lang']}|"
        f"{entry['en']}|{entry['tgt']}"
    )


def filter_value(label: str):
    return None if label == "All" else label


def short(text: str, n: int = 60) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + "..."


def format_score(score) -> str:
    return "N/A" if score is None else f"{score:.2f}"

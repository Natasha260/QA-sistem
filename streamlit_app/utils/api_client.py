# streamlit_app/utils/api_client.py
import requests
import streamlit as st
from typing import Optional, Dict, Any, List
from urllib.parse import quote

try:
    API_URL = st.secrets.get("API_URL", "http://localhost:8000")
except Exception:
    API_URL = "http://localhost:8000"

_EPHEMERAL_SUFFIXES = ("_result", "_data", "_key")


def get_headers() -> Dict[str, str]:
    token = st.session_state.get("access_token")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _extract_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "")[:500]

    detail = payload.get("detail") if isinstance(payload, dict) else None

    if isinstance(detail, str):
        return detail

    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(x) for x in item.get("loc", []))
                msg = item.get("msg", "")
                parts.append(f"{loc}: {msg}" if loc else msg)
            else:
                parts.append(str(item))
        return "; ".join(p for p in parts if p)

    return ""


def api_request(method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
    url = f"{API_URL}{endpoint}"
    headers = get_headers()
    if "headers" in kwargs:
        headers.update(kwargs.pop("headers"))
    if "files" not in kwargs:
        headers.setdefault("Content-Type", "application/json")

    try:
        response = requests.request(method, url, headers=headers, **kwargs)

        if endpoint == "/health" and response.status_code == 503:
            try:
                return response.json()
            except ValueError:
                return {}

        response.raise_for_status()
        return response.json() if response.content else {}

    except requests.exceptions.RequestException as e:
        status = e.response.status_code if e.response is not None else "??"
        detail = _extract_detail(e.response) if e.response is not None else ""
        message = detail or str(e)
        st.error(f"API Error ({status}): {message}")

        if e.response is not None and e.response.status_code == 401:
            _clear_session_state()

        return None


def _matches_suffix(key: str) -> bool:
    return any(key.endswith(sfx) for sfx in _EPHEMERAL_SUFFIXES)


def _clear_session_state() -> None:
    st.session_state.access_token = None
    st.session_state.user = None
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and _matches_suffix(key):
            del st.session_state[key]


def login(email: str, password: str) -> bool:
    result = api_request(
        "POST", "/auth/login",
        json={"email": email, "password": password},
    )
    if result and "access_token" in result:
        st.session_state.access_token = result["access_token"]
        st.session_state.user = result
        return True
    return False


def register(email: str, username: str, password: str, full_name: str = "") -> bool:
    result = api_request("POST", "/auth/register", json={
        "email": email,
        "username": username,
        "password": password,
        "full_name": full_name,
    })
    if result and "access_token" in result:
        st.session_state.access_token = result["access_token"]
        st.session_state.user = result
        return True
    return False


def logout():
    _clear_session_state()


def get_me() -> Optional[Dict]:
    return api_request("GET", "/auth/me")


def query(
    question: str,
    top_k: int = 5,
    engine: Optional[str] = None,
    subject: str = "general",
    eval_mode: Optional[str] = None,
) -> Optional[Dict]:
    return api_request("POST", "/query", json={
        "question": question,
        "top_k": top_k,
        "engine": engine,
        "subject": subject,
        "eval_mode": eval_mode,
    })


def query_batch(
    questions: List[str],
    top_k: int = 5,
    subject: str = "general",
    eval_mode: Optional[str] = None,
) -> Optional[Dict]:
    return api_request("POST", "/query/batch", json={
        "questions": questions,
        "top_k": top_k,
        "subject": subject,
        "eval_mode": eval_mode,
    })


def query_ensemble(
    question: str,
    top_k: int = 5,
    models: Optional[List[str]] = None,
    subject: str = "general",
) -> Optional[Dict]:
    return api_request("POST", "/query/ensemble", json={
        "question": question,
        "top_k": top_k,
        "models": models,
        "subject": subject,
    })


def add_document(title: str, text: str, filename: str = None, description: str = None) -> Optional[Dict]:
    return api_request("POST", "/admin/documents", json={
        "title": title, "text": text,
        "filename": filename, "description": description,
    })


def import_document_file(file, title: str = None, description: str = None) -> Optional[Dict]:
    data = {}
    if title:
        data["title"] = title
    if description:
        data["description"] = description
    return api_request(
        "POST", "/admin/documents/import",
        files={"file": file}, data=data,
    )


def list_documents(limit: int = 100, offset: int = 0) -> Optional[Dict]:
    return api_request("GET", "/admin/documents", params={"limit": limit, "offset": offset})


def list_document_chunks(document_id: str, limit: int = 100, offset: int = 0) -> Optional[Dict]:
    return api_request(
        "GET", f"/admin/documents/{quote(document_id, safe='')}/chunks",
        params={"limit": limit, "offset": offset},
    )


def delete_document(document_id: str) -> Optional[Dict]:
    return api_request("DELETE", f"/admin/documents/{quote(document_id, safe='')}")


def add_terminology(term: str, definition: str, subject: str = "general") -> Optional[Dict]:
    return api_request("POST", "/admin/terminology/add", json={
        "term": term, "definition": definition, "subject": subject,
    })


def list_terminology(subject: str = None, limit: int = 100, offset: int = 0) -> Optional[Dict]:
    params = {"limit": limit, "offset": offset}
    if subject:
        params["subject"] = subject
    return api_request("GET", "/admin/terminology/list", params=params)


def import_terminology_csv(file) -> Optional[Dict]:
    return api_request("POST", "/admin/terminology/import-csv", files={"file": file})


def delete_terminology(entry_id: str) -> Optional[Dict]:
    return api_request("DELETE", f"/admin/terminology/delete/{quote(entry_id, safe='')}")


def add_qa_pair(question: str, answer: str) -> Optional[Dict]:
    return api_request("POST", "/admin/qa-pairs/add",
                       params={"question": question, "answer": answer})


def list_qa_pairs(limit: int = 100, offset: int = 0) -> Optional[Dict]:
    return api_request("GET", "/admin/qa-pairs/list", params={"limit": limit, "offset": offset})


def delete_qa_pair(pair_id: str) -> Optional[Dict]:
    return api_request("DELETE", f"/admin/qa-pairs/delete/{quote(pair_id, safe='')}")


def clear_cache(eval_mode: str = None) -> Optional[Dict]:
    params = {}
    if eval_mode:
        params["eval_mode"] = eval_mode
    return api_request("DELETE", "/admin/cache/clear", params=params)


def get_cache_stats() -> Optional[Dict]:
    return api_request("GET", "/admin/cache/stats")


def save_correction(question: str, corrected_answer: str) -> Optional[Dict]:
    return api_request("POST", "/hitl/save-correction", json={
        "question": question,
        "corrected_answer": corrected_answer,
    })


def get_uncertain(limit: int = 10, offset: int = 0) -> Optional[Dict]:
    params = {"limit": limit, "offset": offset}
    return api_request("GET", "/hitl/uncertain", params=params)


def confirm_uncertain(example_id: str, corrected_answer: str) -> Optional[Dict]:
    return api_request(
        "POST", f"/hitl/uncertain/{quote(example_id, safe='')}/confirm",
        json={"corrected_answer": corrected_answer},
    )


def reject_uncertain(example_id: str) -> Optional[Dict]:
    return api_request(
        "POST", f"/hitl/uncertain/{quote(example_id, safe='')}/reject",
    )


def get_uncertain_stats() -> Optional[Dict]:
    return api_request("GET", "/hitl/uncertain/stats")


def get_health() -> Optional[Dict]:
    return api_request("GET", "/health")


def get_healthz() -> Optional[Dict]:
    return api_request("GET", "/healthz")

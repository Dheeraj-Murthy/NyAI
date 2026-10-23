import os
from typing import Any, Dict, List, Optional

import requests

DEFAULT_API_URL = "http://localhost:8000"
DOCUMENT_TIMEOUT = 300  # OCR + 7B generation on an uploaded document can be slow
DEFAULT_TIMEOUT = 60
# Probes every candidate model live (10 at a time, 10s cap each) — bounded
# but can legitimately take longer than a normal request on a big catalog.
# Shared across all three "test API key" flows (Gemini, Anthropic, OpenAI).
MODEL_TEST_TIMEOUT = 120


class LegalRAGAPIError(Exception):
    """Raised for any failure talking to the backend (connection, timeout, non-2xx, or a
    {"success": false} payload)."""


def get_api_base_url() -> str:
    return os.environ.get("RAG_API_URL", DEFAULT_API_URL).rstrip("/")


BASE_URL = get_api_base_url()


def _url(path: str) -> str:
    return f"{BASE_URL}{path}"


def _handle_response(resp: requests.Response) -> Dict[str, Any]:
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        detail = None
        try:
            detail = resp.json().get("detail")
        except Exception:
            pass
        raise LegalRAGAPIError(detail or str(e)) from e

    payload = resp.json()
    if payload.get("success") is False:
        raise LegalRAGAPIError(payload.get("message", "Request failed"))
    return payload


def health() -> Dict[str, Any]:
    try:
        resp = requests.get(_url("/health"), timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e


def status() -> Dict[str, Any]:
    try:
        resp = requests.get(_url("/status"), timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def query(
    query_text: str,
    top_k: int = 8,
    threshold: float = 0.3,
    include_debug: bool = False,
    model: Optional[str] = None,
    external_ok: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    data = {
        "query": query_text,
        "top_k": top_k,
        "threshold": threshold,
        "include_debug": include_debug,
        "external_ok": external_ok,
    }
    if model:
        data["model"] = model
    if api_key:
        data["api_key"] = api_key
    try:
        resp = requests.post(_url("/query"), data=data, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def chat(
    message: str,
    include_debug: bool = False,
    model: Optional[str] = None,
    external_ok: bool = False,
    api_key: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """session_id: an existing conversation's ID, "new" to start one (the
    response's session_id is its ID), or None for the shared default."""
    data = {"message": message, "include_debug": include_debug, "external_ok": external_ok}
    if model:
        data["model"] = model
    if api_key:
        data["api_key"] = api_key
    if session_id:
        data["session_id"] = session_id
    try:
        resp = requests.post(_url("/chat"), data=data, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def gemini_models(api_key: str) -> List[str]:
    try:
        resp = requests.post(
            _url("/models/gemini"),
            data={"api_key": api_key},
            timeout=MODEL_TEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]["models"]


def anthropic_models(api_key: str) -> List[str]:
    try:
        resp = requests.post(
            _url("/models/anthropic"),
            data={"api_key": api_key},
            timeout=MODEL_TEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]["models"]


def openai_models(api_key: str) -> List[str]:
    try:
        resp = requests.post(
            _url("/models/openai"),
            data={"api_key": api_key},
            timeout=MODEL_TEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]["models"]


def chat_clear(session_id: Optional[str] = None) -> None:
    data = {"session_id": session_id} if session_id else None
    try:
        resp = requests.post(_url("/chat/clear"), data=data, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    _handle_response(resp)


def chat_history(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    params = {"session_id": session_id} if session_id else None
    try:
        resp = requests.get(_url("/chat/history"), params=params, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def upload_document(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    query_text: Optional[str] = None,
    include_retrieval: bool = True,
    include_debug: bool = False,
    model: Optional[str] = None,
    external_ok: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    data = {
        "query": query_text or "",
        "include_retrieval": include_retrieval,
        "include_debug": include_debug,
        "external_ok": external_ok,
    }
    if model:
        data["model"] = model
    if api_key:
        data["api_key"] = api_key
    try:
        resp = requests.post(
            _url("/document"),
            files={"file": (filename, file_bytes, content_type)},
            data=data,
            timeout=DOCUMENT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def retrieval_test(query_text: str) -> Dict[str, Any]:
    try:
        resp = requests.post(
            _url("/retrieval-test"),
            data={"query": query_text},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def graph_judgment(judgment_id: int, depth: int = 2, max_nodes: int = 100) -> Dict[str, Any]:
    try:
        resp = requests.get(
            _url(f"/graph/judgment/{judgment_id}"),
            params={"depth": depth, "max_nodes": max_nodes},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def graph_landmark_cases(limit: int = 10, rank_by: str = "pagerank") -> List[Dict[str, Any]]:
    try:
        resp = requests.get(
            _url("/graph/landmark-cases"),
            params={"limit": limit, "rank_by": rank_by},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def graph_path(source_id: int, target_id: int) -> Optional[List[int]]:
    try:
        resp = requests.get(
            _url("/graph/path"),
            params={"source_id": source_id, "target_id": target_id},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e

    if not resp.ok:
        _handle_response(resp)
    payload = resp.json()
    if not payload.get("success"):
        return None
    return payload["data"]["path"]


def graph_search(q: str, limit: int = 10) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(
            _url("/graph/search"), params={"q": q, "limit": limit}, timeout=DEFAULT_TIMEOUT
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def graph_neighbors(
    judgment_id: int,
    direction: str = "cites",
    limit: int = 25,
    offset: int = 0,
    relationship: Optional[str] = None,
    q: Optional[str] = None,
    court: Optional[str] = None,
) -> Dict[str, Any]:
    """direction: "cites" (cases this judgment cites) or "cited-by". q filters
    on case name or citation text, court on the court name."""
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if relationship:
        params["relationship"] = relationship
    if q:
        params["q"] = q
    if court:
        params["court"] = court
    try:
        resp = requests.get(
            _url(f"/graph/judgment/{judgment_id}/{direction}"),
            params=params,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def graph_mind_map(judgment_id: int, per_branch: int = 12, child_per_branch: int = 5) -> Dict[str, Any]:
    """A judgment's citations as a tree for the mind-map view."""
    try:
        resp = requests.get(
            _url(f"/graph/judgment/{judgment_id}/mind-map"),
            params={"per_branch": per_branch, "child_per_branch": child_per_branch},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def document(doc_type: str, doc_id: int) -> Dict[str, Any]:
    """doc_type "judgment" (doc_id = judgment ID) or "statute" (doc_id = statute section ID)."""
    path = f"/documents/judgment/{doc_id}" if doc_type == "judgment" else f"/documents/statute-section/{doc_id}"
    try:
        resp = requests.get(_url(path), timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def document_pdf(doc_type: str, doc_id: int) -> bytes:
    """Original PDF bytes: doc_id = judgment ID, or statute ID for statutes."""
    path = f"/documents/judgment/{doc_id}/pdf" if doc_type == "judgment" else f"/documents/statute/{doc_id}/pdf"
    try:
        resp = requests.get(_url(path), timeout=DOCUMENT_TIMEOUT)
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    if resp.status_code != 200:
        _handle_response(resp)
    return resp.content


def chat_export(session_id: str, fmt: str = "json") -> Any:
    """fmt "json": the portable export dict (re-importable); "markdown": transcript text."""
    try:
        resp = requests.get(
            _url("/chat/export"), params={"session_id": session_id, "format": fmt}, timeout=DEFAULT_TIMEOUT
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    if fmt == "markdown":
        if resp.status_code != 200:
            _handle_response(resp)
        return resp.text
    return _handle_response(resp)["data"]


def chat_import(conversation: Dict[str, Any]) -> str:
    """Start a new conversation from an exported one; returns its session ID."""
    try:
        resp = requests.post(_url("/chat/import"), json=conversation, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]["session_id"]


def chat_sessions(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """Chats, most recent first: {"total", "items": [{session_id, title, last_active, ...}]}."""
    try:
        resp = requests.get(
            _url("/chat/sessions"), params={"limit": limit, "offset": offset}, timeout=DEFAULT_TIMEOUT
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def chat_rename(session_id: str, title: str) -> None:
    try:
        resp = requests.patch(
            _url(f"/chat/sessions/{session_id}"), json={"title": title}, timeout=DEFAULT_TIMEOUT
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    _handle_response(resp)


def chat_delete(session_id: str) -> None:
    try:
        resp = requests.delete(_url(f"/chat/sessions/{session_id}"), timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    _handle_response(resp)

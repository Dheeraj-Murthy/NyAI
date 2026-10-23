from fastapi import Body, FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal
import tempfile
import os

from rag_pipeline import LegalRAGPipeline
from chat_store import is_valid_session_id
from retrieval.citation_graph import CitationGraphManager
from documents import DocumentStore

app = FastAPI(
    title="Legal RAG API",
    description="Retrieval-Augmented Generation for legal documents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[LegalRAGPipeline] = None
graph_manager: Optional[CitationGraphManager] = None
document_store: Optional[DocumentStore] = None


@app.on_event("startup")
async def startup_event():
    global pipeline, graph_manager, document_store
    pipeline = LegalRAGPipeline(load_llm=True)
    # Share the pipeline's manager rather than building a second one.
    graph_manager = getattr(pipeline, "graph_manager", None) or CitationGraphManager()
    graph_manager.load_graph_from_db()
    # Same database settings as the graph manager.
    document_store = DocumentStore(
        {
            "host": graph_manager.db_host,
            "port": graph_manager.db_port,
            "dbname": graph_manager.db_name,
            "user": graph_manager.db_user,
            "password": graph_manager.db_password,
        }
    )


@app.get("/")
async def root():
    return {"status": "ok", "message": "Legal RAG API is running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipeline.backends.get("default") is not None if pipeline else False,
    }


def _requires_external_ok(model: Optional[str]) -> bool:
    return bool(model) and model.startswith(("claude-", "gpt-", "o1", "o3", "gemini-"))


@app.post("/query")
async def query_legal(
    query: str = Form(...),
    top_k: int = Form(8),
    threshold: float = Form(0.3),
    include_debug: bool = Form(False),
    model: Optional[str] = Form(None),
    external_ok: bool = Form(False),
    api_key: Optional[str] = Form(None),
):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    if _requires_external_ok(model) and not external_ok:
        raise HTTPException(status_code=400, detail="External model requires external_ok=true")

    try:
        result = pipeline.query(
            user_query=query, include_debug_info=include_debug,
            model=model, external_ok=external_ok, api_key=api_key,
        )

        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_legal(
    message: str = Form(...),
    include_debug: bool = Form(False),
    model: Optional[str] = Form(None),
    external_ok: bool = Form(False),
    api_key: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """session_id: an ID from an earlier response to continue that
    conversation, "new" to start one (the response carries its ID), or
    omitted for the shared default conversation."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    if _requires_external_ok(model) and not external_ok:
        raise HTTPException(status_code=400, detail="External model requires external_ok=true")

    if session_id == "new":
        session_id = pipeline.new_chat_session()
    else:
        session_id = _checked_session_id(session_id)

    try:
        result = pipeline.chat(
            user_message=message, include_debug_info=include_debug,
            model=model, external_ok=external_ok, api_key=api_key,
            session_id=session_id,
        )

        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/gemini")
async def list_gemini_models(api_key: str = Form(...)):
    from llm.gemini_backend import list_available_models

    try:
        models = list_available_models(api_key)
        return {"success": True, "data": {"models": models}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/models/anthropic")
async def list_anthropic_models(api_key: str = Form(...)):
    from llm.anthropic_backend import list_available_models

    try:
        models = list_available_models(api_key)
        return {"success": True, "data": {"models": models}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/models/openai")
async def list_openai_models(api_key: str = Form(...)):
    from llm.openai_backend import list_available_models

    try:
        models = list_available_models(api_key)
        return {"success": True, "data": {"models": models}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _checked_session_id(session_id: Optional[str]) -> Optional[str]:
    if session_id is not None and not is_valid_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    return session_id


@app.post("/chat/clear")
async def clear_chat(session_id: Optional[str] = Form(None)):
    """Delete one conversation (or the shared default one if no session_id)."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    pipeline.clear_chat_history(_checked_session_id(session_id))
    return {"success": True, "message": "Chat history cleared"}


@app.get("/chat/history")
async def get_chat_history(session_id: Optional[str] = None):
    """A conversation's messages, oldest first (the shared default one if no session_id)."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    history = pipeline.get_chat_history(_checked_session_id(session_id))
    return {"success": True, "data": history}


@app.get("/chat/sessions")
async def list_chat_sessions(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    """All chats with at least one message, most recently active first.
    No authorisation yet: every chat is listed."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    return {"success": True, "data": pipeline.list_chat_sessions(limit=limit, offset=offset)}


@app.post("/chat/sessions")
async def create_chat_session():
    """Start an empty chat; send its session_id with /chat to use it."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    return {"success": True, "data": {"session_id": pipeline.new_chat_session()}}


@app.patch("/chat/sessions/{session_id}")
async def rename_chat_session(session_id: str, body: Dict[str, Any] = Body(...)):
    """Rename a chat: body {"title": "..."}."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    _checked_session_id(session_id)
    title = body.get("title")
    if not isinstance(title, str) or not title.strip():
        raise HTTPException(status_code=422, detail="title must be a non-empty string")

    if not pipeline.rename_chat_session(session_id, title):
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"success": True}


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat and all its messages."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    pipeline.clear_chat_history(_checked_session_id(session_id))
    return {"success": True}


@app.get("/chat/export")
async def export_chat(
    session_id: Optional[str] = None,
    format: str = "json",
):
    """A conversation as JSON (messages, cited documents and running
    summary — import it to continue anywhere) or a readable Markdown
    transcript."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    if format not in ("json", "markdown"):
        raise HTTPException(status_code=422, detail="format must be json or markdown")

    exported = pipeline.export_chat(_checked_session_id(session_id), fmt=format)
    if format == "markdown":
        return PlainTextResponse(exported, media_type="text/markdown")
    return {"success": True, "data": exported}


@app.post("/chat/import")
async def import_chat(conversation: Dict[str, Any] = Body(...)):
    """Start a new conversation from a JSON export; returns its session_id."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        session_id = pipeline.import_chat(conversation)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"success": True, "data": {"session_id": session_id}}


@app.post("/document")
async def query_document(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    include_retrieval: bool = Form(True),
    include_debug: bool = Form(False),
    model: Optional[str] = Form(None),
    external_ok: bool = Form(False),
    api_key: Optional[str] = Form(None),
):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    if _requires_external_ok(model) and not external_ok:
        raise HTTPException(status_code=400, detail="External model requires external_ok=true")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.query_with_document(
            document_path=tmp_path,
            user_query=query,
            include_retrieval=include_retrieval,
            include_debug_info=include_debug,
            model=model,
            external_ok=external_ok,
            api_key=api_key,
        )

        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/retrieval-test")
async def test_retrieval(query: str = Form(...)):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        result = pipeline.test_retrieval_only(query)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    return {"success": True, "data": pipeline.get_system_status()}


@app.get("/graph/judgment/{judgment_id}")
async def get_citation_graph(
    judgment_id: int,
    depth: int = Query(2, ge=1, le=3),
    max_nodes: int = Query(100, ge=1, le=500),
):
    if not graph_manager:
        raise HTTPException(status_code=500, detail="Graph manager not initialized")

    try:
        subgraph = graph_manager.get_subgraph(judgment_id, depth=depth, max_nodes=max_nodes)
        return {"success": True, "data": subgraph}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/landmark-cases")
async def get_landmark_cases(
    limit: int = 10,
    rank_by: Literal["pagerank", "citations_per_year", "in_degree"] = "pagerank",
):
    if not graph_manager:
        raise HTTPException(status_code=500, detail="Graph manager not initialized")

    try:
        landmarks = graph_manager.get_landmark_cases(limit=limit, rank_by=rank_by)
        return {"success": True, "data": landmarks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/path")
async def get_precedent_path(source_id: int, target_id: int):
    if not graph_manager:
        raise HTTPException(status_code=500, detail="Graph manager not initialized")

    try:
        path = graph_manager.get_shortest_path(source_id=source_id, target_id=target_id)
        if path is None:
            return {"success": False, "message": "No citation path found between cases"}
        return {"success": True, "data": {"path": path}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/search")
async def search_cases(
    q: str = Query(..., min_length=1, max_length=300),
    limit: int = Query(10, ge=1, le=50),
):
    """Find cases by judgment ID, reporter citation or case name."""
    if not graph_manager:
        raise HTTPException(status_code=500, detail="Graph manager not initialized")

    try:
        return {"success": True, "data": graph_manager.search(q, limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _neighbors_response(judgment_id: int, direction: str, relationship, limit: int, offset: int, q, court):
    if not graph_manager:
        raise HTTPException(status_code=500, detail="Graph manager not initialized")

    try:
        page = graph_manager.get_neighbors(
            judgment_id, direction=direction, relationship=relationship, limit=limit, offset=offset, q=q, court=court
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if page is None:
        raise HTTPException(status_code=404, detail=f"Judgment {judgment_id} not found")
    return {"success": True, "data": page}


@app.get("/graph/judgment/{judgment_id}/cites")
async def get_cases_cited(
    judgment_id: int,
    relationship: Optional[str] = None,
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
    q: Optional[str] = Query(None, max_length=300, description="Case name or citation text contains"),
    court: Optional[str] = Query(None, max_length=200, description="Court name contains"),
):
    """Paged list of the cases this judgment cites, most important first."""
    return _neighbors_response(judgment_id, "cites", relationship, limit, offset, q, court)


@app.get("/graph/judgment/{judgment_id}/cited-by")
async def get_citing_cases(
    judgment_id: int,
    relationship: Optional[str] = None,
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
    q: Optional[str] = Query(None, max_length=300, description="Case name or citation text contains"),
    court: Optional[str] = Query(None, max_length=200, description="Court name contains"),
):
    """Paged list of the cases that cite this judgment, most important first."""
    return _neighbors_response(judgment_id, "cited_by", relationship, limit, offset, q, court)


@app.get("/graph/judgment/{judgment_id}/mind-map")
async def get_mind_map(
    judgment_id: int,
    per_branch: int = Query(12, ge=1, le=50),
    child_per_branch: int = Query(5, ge=0, le=20),
):
    """A judgment's citations as a tree (cites / cited by, grouped by
    relationship, two levels deep) for the mind-map view."""
    if not graph_manager:
        raise HTTPException(status_code=500, detail="Graph manager not initialized")

    try:
        tree = graph_manager.get_mind_map(judgment_id, per_branch=per_branch, child_per_branch=child_per_branch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if tree is None:
        raise HTTPException(status_code=404, detail=f"Judgment {judgment_id} not found")
    return {"success": True, "data": tree}


class ResolveRequest(BaseModel):
    citations: List[str]


MAX_RESOLVE_BATCH = 500


@app.post("/graph/resolve")
async def resolve_citations(body: ResolveRequest):
    """Resolve citation strings to judgment IDs, the same way edge building does."""
    if not graph_manager:
        raise HTTPException(status_code=500, detail="Graph manager not initialized")
    if not 1 <= len(body.citations) <= MAX_RESOLVE_BATCH:
        raise HTTPException(
            status_code=422, detail=f"Send between 1 and {MAX_RESOLVE_BATCH} citations"
        )

    try:
        return {"success": True, "data": graph_manager.resolve_citations(body.citations)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _documents() -> DocumentStore:
    if not document_store:
        raise HTTPException(status_code=500, detail="Document store not initialized")
    return document_store


def _pdf_response(path, missing: str):
    if path is None:
        raise HTTPException(status_code=404, detail=missing)
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@app.get("/documents/judgment/{judgment_id}")
async def get_judgment_document(judgment_id: int):
    """A judgment's metadata and full text; has_pdf says whether the
    original PDF is available."""
    try:
        doc = _documents().judgment(judgment_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Judgment {judgment_id} not found")
    return {"success": True, "data": doc}


@app.get("/documents/judgment/{judgment_id}/pdf")
async def get_judgment_pdf(judgment_id: int):
    try:
        path = _documents().judgment_pdf(judgment_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _pdf_response(path, f"No original PDF available for judgment {judgment_id}")


@app.get("/documents/statute-section/{section_id}")
async def get_statute_section_document(section_id: int):
    """A statute section's text; has_pdf says whether the statute's PDF is
    available (GET /documents/statute/{statute_id}/pdf)."""
    try:
        doc = _documents().statute_section(section_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Statute section {section_id} not found")
    return {"success": True, "data": doc}


@app.get("/documents/statute/{statute_id}/pdf")
async def get_statute_pdf(statute_id: int):
    try:
        path = _documents().statute_pdf(statute_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _pdf_response(path, f"No original PDF available for statute {statute_id}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

import streamlit as st

from lib import api_client
from lib.case_graph import render_answer_graph
from lib.ui_helpers import (
    confidence_badge,
    linked_label,
    render_api_error,
    render_metrics,
    render_model_selector,
    render_precedent_chains,
    render_sectioned_sources_and_citations,
)

st.set_page_config(page_title="Document upload — Legal RAG", page_icon=":material/upload_file:", layout="wide")

st.title("Document upload", icon=":material/upload_file:")
st.caption("Upload a PDF, DOCX, or TXT document. OCR runs server-side for scanned PDFs and can take a while.")

uploaded_file = st.file_uploader("Document", type=["pdf", "docx", "txt"])
mode = st.segmented_control(
    "Mode", ["Ask a question", "Summarize judgment"], default="Ask a question"
) or "Ask a question"

if mode == "Summarize judgment":
    st.caption(
        "Summarizes the document and resolves every case it cites to a judgment "
        "already in the corpus, where possible."
    )
    include_debug = st.toggle("Include debug info", value=False, key="summarize_debug")
else:
    query_text = st.text_input("Optional question about this document", placeholder="What did the court hold?")
    with st.container(horizontal=True):
        include_retrieval = st.toggle("Cross-reference case law", value=True)
        include_debug = st.toggle("Include debug info", value=False, key="ask_debug")

model, external_ok, api_key, is_external = render_model_selector("document")

action_label = "Summarize" if mode == "Summarize judgment" else "Analyze document"
if st.button(action_label, icon=":material/play_arrow:", type="primary", disabled=uploaded_file is None):
    if is_external and (not external_ok or not model):
        st.warning(
            "Tick the external-model confirmation checkbox and pick a model first."
            if not external_ok
            else "Pick a model first (test your API key if you haven't).",
            icon=":material/lock:",
        )
    elif mode == "Summarize judgment":
        with st.spinner("Uploading and summarizing (OCR + generation can take several minutes)..."):
            try:
                data = api_client.summarize_document(
                    file_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                    content_type=uploaded_file.type or "application/octet-stream",
                    include_debug=include_debug,
                    model=model,
                    external_ok=external_ok,
                    api_key=api_key,
                )
                st.session_state.last_summary_result = data
                st.session_state.last_document_result = None
            except Exception as e:
                render_api_error(e)
                st.session_state.last_summary_result = None
    else:
        with st.spinner("Uploading and processing (OCR + generation can take several minutes)..."):
            try:
                data = api_client.upload_document(
                    file_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                    content_type=uploaded_file.type or "application/octet-stream",
                    query_text=query_text or None,
                    include_retrieval=include_retrieval,
                    include_debug=include_debug,
                    model=model,
                    external_ok=external_ok,
                    api_key=api_key,
                )
                st.session_state.last_document_result = data
                st.session_state.last_summary_result = None
            except Exception as e:
                render_api_error(e)
                st.session_state.last_document_result = None

summary_result = st.session_state.get("last_summary_result")
if mode == "Summarize judgment" and summary_result:
    if summary_result.get("error"):
        st.error(summary_result["error"], icon=":material/error:")
    else:
        doc = summary_result.get("document", {})
        if doc:
            st.badge(
                f"{doc.get('source', 'document')} — {doc.get('pages', '?')} pages · {doc.get('type', '?')}",
                icon=":material/description:",
                color="blue",
            )
            st.space("small")

        with st.container(border=True):
            st.markdown(summary_result.get("summary") or "No summary produced.")

        st.space("medium")

        cited = summary_result.get("cited_documents", [])
        unresolved = summary_result.get("unresolved_citations", [])
        st.subheader(f"Cited documents ({len(cited)})", icon=":material/gavel:")
        if cited:
            for c in cited:
                st.markdown(
                    f"- “{c['citation_text']}” → "
                    f"{linked_label(c['label'], {'judgment_id': c['judgment_id']})} "
                    f"`{c['resolution_method']}`"
                )
        else:
            st.caption("No citations in this document resolved to a judgment already in the corpus.")
        if unresolved:
            with st.expander(f"Unresolved citations ({len(unresolved)})", icon=":material/help_outline:"):
                for u in unresolved:
                    st.markdown(f"- “{u['citation_text']}” — {u['status']}")

        render_precedent_chains(summary_result.get("precedent_chains", []))
        render_answer_graph(summary_result, key="doc_summary_graph")

        st.space("medium")
        render_metrics(summary_result.get("metrics", {}))

        if summary_result.get("debug"):
            with st.expander("Debug info", icon=":material/bug_report:"):
                st.json(summary_result["debug"])

result = st.session_state.get("last_document_result")
if mode == "Ask a question" and result:
    if result.get("error"):
        st.error(result["error"], icon=":material/error:")
    else:
        doc = result.get("document", {})
        if doc:
            st.badge(
                f"{doc.get('source', 'document')} — {doc.get('pages', '?')} pages · {doc.get('type', '?')}",
                icon=":material/description:",
                color="blue",
            )
            st.space("small")

        with st.container(border=True):
            if not result.get("answer_found", True):
                st.warning(result.get("answer", "No answer produced."), icon=":material/search_off:")
            else:
                st.markdown(result.get("answer", ""))

            confidence = result.get("confidence")
            if confidence is not None:
                confidence_badge(confidence)

            render_sectioned_sources_and_citations(result)

        st.space("medium")
        render_metrics(result.get("metrics", {}))

        if result.get("debug"):
            with st.expander("Debug info", icon=":material/bug_report:"):
                st.json(result["debug"])

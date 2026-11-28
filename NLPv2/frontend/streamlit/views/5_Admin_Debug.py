import streamlit as st

from lib import api_client
from lib.theme import page_header
from lib.ui_helpers import render_api_error

st.set_page_config(page_title="Admin / debug — NyAI", page_icon=":material/tune:", layout="wide")

page_header("Admin / debug")

tab_status, tab_retrieval = st.tabs(["Pipeline status", "Retrieval test"])

with tab_status:
    st.caption("Model, retriever, and memory configuration reported by the running backend.")

    @st.cache_data(ttl=30)
    def _status():
        return api_client.status()

    with st.container(horizontal=True):
        st.button("Refresh", icon=":material/refresh:", on_click=_status.clear)

    try:
        with st.container(border=True):
            st.json(_status())
    except Exception as e:
        render_api_error(e)

with tab_retrieval:
    st.caption(
        "CPU-safe: runs the hybrid retriever only, skips generation. "
        "Useful for checking retrieval quality directly."
    )

    query_text = st.text_input("Query", key="retrieval_test_query", placeholder="Section 302 IPC punishment")
    if st.button(
        "Run retrieval test", icon=":material/play_arrow:", type="primary", disabled=not query_text.strip()
    ):
        with st.spinner("Retrieving..."):
            try:
                data = api_client.retrieval_test(query_text)
            except Exception as e:
                render_api_error(e)
                data = None

        if data is not None:
            with st.container(horizontal=True):
                st.metric("Chunks found", data.get("chunks_found", 0), border=True)
                st.metric("Retrieval time", f"{data.get('retrieval_time')}s", border=True)

            st.subheader("Top chunks", icon=":material/library_books:")
            for chunk in data.get("retrieved_chunks", []):
                title = f"{chunk.get('case', 'Unknown')} — {chunk.get('section', chunk.get('doc_type', ''))}"
                with st.expander(title, icon=":material/description:"):
                    st.write(chunk.get("text", ""))
                    st.json({k: v for k, v in chunk.items() if k != "text"})

            with st.expander("Retrieval stats", icon=":material/query_stats:"):
                st.json(data.get("retrieval_stats", {}))

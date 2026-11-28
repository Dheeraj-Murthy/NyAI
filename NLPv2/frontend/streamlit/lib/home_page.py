import streamlit as st

from lib import api_client
from lib.theme import CHAT_PAGE, backend_health, hero, page_header, start_new_research

SUGGESTIONS = [
    "What are the principles of natural justice?",
    "What is the punishment for murder under Indian law?",
    "Explain the right to life under Article 21",
]

# (key, page, icon, colour, title, description, link label)
CARDS = [
    ("ask", CHAT_PAGE, "chat_bubble", "navy", "Ask a legal question",
     "Get a cited answer you can follow up on in conversation.", "Start asking"),
    ("analyse", "views/3_Document_Upload.py", "upload_file", "amber", "Analyse a judgment",
     "Upload a PDF, DOCX or TXT to summarize it or ask about it.", "Upload file"),
    ("explore", "views/4_Citation_Graph.py", "hub", "green", "Explore precedents",
     "See which cases a judgment relies on, follows or overrules.", "Open graph"),
]


def _ask(question: str) -> None:
    """Hand a question to the chat page, which asks it in a new conversation."""
    st.session_state.pending_input = question
    start_new_research()


def render() -> None:
    page_header("Home")
    with st.container(key="nyai-home"):
        hero(
            "What would you like to research today?",
            "Retrieval-augmented answers from Indian Supreme Court judgments and statutes.",
        )
        # Inside a container the chat input sits inline instead of at the bottom.
        with st.container():
            question = st.chat_input(
                "e.g. What are the principles of natural justice in departmental inquiries?",
                key="home_question",
            )
        selected = st.pills("Try asking:", SUGGESTIONS, label_visibility="collapsed", key="home_suggestion")
        if question or selected:
            _ask(question or selected)

        st.space("medium")
        for col, (key, page, icon, colour, title, desc, link) in zip(st.columns(len(CARDS)), CARDS):
            with col, st.container(height="stretch", key=f"nyai-card-{key}"):
                st.html(
                    f'<div class="nyai-card"><span class="nyai-icon {colour}" aria-hidden="true">{icon}</span>'
                    f"<h3>{title}</h3><p>{desc}</p></div>"
                )
                st.page_link(page, label=f"{link} →")

        st.html(
            '<div class="nyai-note">Research tool, not legal advice. '
            "Always verify citations against the original judgment.</div>"
        )

        health = backend_health()
        if "error" in health:
            st.error(
                f"Cannot reach backend at `{api_client.BASE_URL}`. "
                f"Check the `RAG_API_URL` environment variable. ({health['error']})",
                icon=":material/error:",
            )
        else:
            try:
                status = api_client.status()
                with st.expander("System status", icon=":material/monitoring:"):
                    st.json(status)
            except Exception:
                st.caption("System status unavailable")

import json

import streamlit as st

from lib import api_client
from lib.case_graph import render_answer_graph
from lib.home_page import SUGGESTIONS
from lib.theme import hero, page_header
from lib.ui_helpers import (
    confidence_badge,
    init_session_state,
    render_api_error,
    render_model_selector,
    render_sectioned_sources_and_citations,
)

def render_context_note(context) -> None:
    """One line on what the model had to work with for this answer."""
    if not context:
        return
    def plural(n: int, word: str) -> str:
        return f"{n} {word}{'' if n == 1 else 's'}"

    parts = [f"saw {plural(context.get('history_messages', 0), 'earlier message')}"]
    if context.get("summarized_messages"):
        parts.append(f"{plural(context['summarized_messages'], 'older message')} summarised to fit")
    if context.get("carried_documents"):
        parts.append(f"{plural(context['carried_documents'], 'document')} from earlier answers")
    st.caption(":material/history: Context: " + " · ".join(parts))


st.set_page_config(page_title="Ask a question — NyAI", page_icon=":material/chat:", layout="wide")
init_session_state()

# Filled once the chat list is loaded, which is where the title comes from.
header_slot = st.empty()

# The conversation's ID lives in the URL (?chat=...), so a refresh or a
# shared link reopens it. No ID yet means a fresh conversation; the server
# assigns one with the first answer.
chat_id = st.query_params.get("chat")
if st.session_state.get("chat_loaded") != chat_id:
    st.session_state.chat_history = []
    if chat_id:
        try:
            for msg in api_client.chat_history(chat_id):
                turn = {k: v for k, v in msg.items() if k != "timestamp"}
                st.session_state.chat_history.append(turn)
        except Exception as e:
            render_api_error(e)
    st.session_state.chat_loaded = chat_id


def _start_new_chat() -> None:
    st.query_params.pop("chat", None)
    st.session_state.chat_history = []
    st.session_state.chat_loaded = None


CHATS_PAGE = 20
st.session_state.setdefault("chats_shown", CHATS_PAGE)


def _open_chat(session_id: str) -> None:
    st.query_params["chat"] = session_id
    st.session_state.pop("confirm_delete", None)


def _ask_delete(session_id: str) -> None:
    st.session_state.confirm_delete = session_id


def _delete_chat(session_id: str) -> None:
    st.session_state.pop("confirm_delete", None)
    try:
        api_client.chat_delete(session_id)
    except Exception as e:
        st.session_state.delete_error = e
        return
    if session_id == st.query_params.get("chat"):
        _start_new_chat()


try:
    chats = api_client.chat_sessions(limit=st.session_state.chats_shown)
except Exception as e:
    chats = None
    chats_error = e
current_title = next(
    (c["title"] for c in (chats or {}).get("items", []) if c["session_id"] == chat_id), None
)

with st.sidebar:
    model, external_ok, api_key, is_external = render_model_selector("chat")
    include_debug = st.toggle("Include debug info", value=False)

    st.html('<div class="nyai-label">Recent</div>')
    if chats is None:
        render_api_error(chats_error)
    elif not chats["items"]:
        st.caption("No chats yet — ask a question to start one.")
    else:
        if "delete_error" in st.session_state:
            render_api_error(st.session_state.pop("delete_error"))
        for c in chats["items"]:
            sid = c["session_id"]
            if st.session_state.get("confirm_delete") == sid:
                st.caption(f"Delete “{c['title']}” and all its messages?")
                yes, no = st.columns(2)
                yes.button("Delete", key=f"delete_yes_{sid}", on_click=_delete_chat, args=(sid,),
                           type="primary", icon=":material/delete:", width="stretch")
                no.button("Cancel", key=f"delete_no_{sid}", on_click=_ask_delete, args=(None,),
                          width="stretch")
                continue
            open_col, delete_col = st.columns([6, 1], gap="small", vertical_alignment="center")
            open_col.button(
                c["title"],
                key=f"open_chat_{sid}",
                on_click=_open_chat,
                args=(sid,),
                width="stretch",
                type="secondary" if sid == chat_id else "tertiary",
                icon=":material/chat_bubble:" if sid == chat_id else None,
            )
            delete_col.button(
                "",
                key=f"delete_chat_{sid}",
                on_click=_ask_delete,
                args=(sid,),
                icon=":material/delete:",
                type="tertiary",
                help="Delete this chat",
            )
        if chats["total"] > len(chats["items"]):
            if st.button(f"Show more ({chats['total'] - len(chats['items'])})", width="stretch"):
                st.session_state.chats_shown += CHATS_PAGE
                st.rerun()

    if chat_id:
        with st.expander("This chat", icon=":material/tune:"):
            new_title = st.text_input("Title", value=current_title or "", key=f"title_{chat_id}")
            if st.button("Rename", width="stretch", disabled=not new_title.strip() or new_title == current_title):
                try:
                    api_client.chat_rename(chat_id, new_title)
                    st.rerun()
                except Exception as e:
                    render_api_error(e)
            st.button("Delete this chat", icon=":material/delete_sweep:", width="stretch",
                      on_click=_ask_delete, args=(chat_id,))

    with st.expander("Export / import", icon=":material/import_export:"):
        if chat_id:
            st.caption("JSON keeps everything needed to continue the chat later or elsewhere.")
            if st.button("Prepare export", width="stretch"):
                try:
                    st.session_state.chat_export = {
                        "id": chat_id,
                        "json": json.dumps(api_client.chat_export(chat_id), indent=2),
                        "markdown": api_client.chat_export(chat_id, fmt="markdown"),
                    }
                except Exception as e:
                    render_api_error(e)
            prepared = st.session_state.get("chat_export")
            if prepared and prepared["id"] == chat_id:
                st.download_button("Download JSON", prepared["json"], file_name=f"chat-{chat_id[:8]}.json",
                                   mime="application/json", width="stretch")
                st.download_button("Download Markdown", prepared["markdown"], file_name=f"chat-{chat_id[:8]}.md",
                                   mime="text/markdown", width="stretch")
        uploaded = st.file_uploader("Continue an exported chat", type=["json"], key="chat_import_file")
        if uploaded is not None and st.button("Import", width="stretch"):
            try:
                new_id = api_client.chat_import(json.loads(uploaded.getvalue()))
                st.query_params["chat"] = new_id
                st.rerun()
            except json.JSONDecodeError:
                st.error("That file isn't valid JSON.", icon=":material/error:")
            except Exception as e:
                render_api_error(e)

with header_slot.container():
    page_header(current_title or "Ask a question")

user_input = st.chat_input("Ask a legal question...") or st.session_state.pop("pending_input", None)

if not st.session_state.chat_history and not user_input:
    with st.container(key="nyai-empty"):
        hero(
            "What would you like to research today?",
            "Each conversation keeps its own history, and you can switch models at any point.",
        )
        selected = st.pills("Try asking:", SUGGESTIONS, label_visibility="collapsed")
        st.html(
            '<div class="nyai-note">Very long chats are summarised to fit the model. There are no accounts '
            "yet, so everyone using this app sees the same chat list.</div>"
        )
    if selected:
        st.session_state.pending_input = selected
        st.rerun()

for i, turn in enumerate(st.session_state.chat_history):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            if turn.get("model_id"):
                st.caption(
                    f":material/smart_toy: {turn['model_id']}"
                    + (f" ({turn['model_version']})" if turn.get("model_version") else "")
                )
            if turn.get("confidence") is not None:
                confidence_badge(turn["confidence"])
            render_context_note(turn.get("context"))
            render_sectioned_sources_and_citations(turn)
            render_answer_graph(turn, key=f"chat_graph_{chat_id}_{i}")
            if turn.get("debug"):
                with st.expander("Debug info", icon=":material/bug_report:"):
                    st.json(turn["debug"])

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if is_external and (not external_ok or not model):
            st.warning(
                "Tick the external-model confirmation checkbox and pick a model "
                "in the sidebar first."
                if not external_ok
                else "Pick a model in the sidebar first (test your API key if you "
                "haven't).",
                icon=":material/lock:",
            )
            data = None
        else:
            started_new_chat = False
            with st.status(":shimmer[Retrieving and generating]", type="compact") as status:
                try:
                    data = api_client.chat(
                        user_input, include_debug=include_debug,
                        model=model, external_ok=external_ok, api_key=api_key,
                        session_id=chat_id or "new",
                    )
                    new_id = data.get("session_id")
                    started_new_chat = bool(not chat_id and new_id and new_id != "default")
                    if started_new_chat:
                        st.query_params["chat"] = new_id
                        st.session_state.chat_loaded = new_id
                    status.update(
                        label="Failed" if data.get("error") else "Done",
                        state="error" if data.get("error") else "complete",
                    )
                except Exception as e:
                    status.update(label="Failed", state="error")
                    render_api_error(e)
                    data = None

        if data is not None and data.get("error"):
            st.error(data["error"], icon=":material/error:")
        elif data is not None:
            model_id = data.get("metrics", {}).get("model_id")
            model_version = data.get("metrics", {}).get("model_version")
            if model_id:
                st.caption(
                    f":material/smart_toy: {model_id}"
                    + (f" ({model_version})" if model_version else "")
                )

            if not data.get("answer_found", True):
                st.warning(data["answer"], icon=":material/search_off:")
            else:
                st.markdown(data["answer"])

            confidence = data.get("confidence")
            if confidence is not None:
                confidence_badge(confidence)
            render_context_note(data.get("context"))
            render_sectioned_sources_and_citations(data)
            # Same key it gets in the history loop once appended below.
            render_answer_graph(data, key=f"chat_graph_{chat_id}_{len(st.session_state.chat_history)}")
            if data.get("debug"):
                with st.expander("Debug info", icon=":material/bug_report:"):
                    st.json(data["debug"])

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": data["answer"],
                    "citations": data.get("citations", []),
                    "sources": data.get("sources", []),
                    "citations_by_type": data.get("citations_by_type"),
                    "sources_by_type": data.get("sources_by_type"),
                    "citation_refs": data.get("citation_refs"),
                    "source_refs": data.get("source_refs"),
                    "citation_refs_by_type": data.get("citation_refs_by_type"),
                    "source_refs_by_type": data.get("source_refs_by_type"),
                    "confidence": confidence,
                    "context": data.get("context"),
                    "debug": data.get("debug"),
                    "model_id": model_id,
                    "model_version": model_version,
                }
            )
            # A chat that just started isn't in the sidebar list drawn above yet.
            if started_new_chat:
                st.rerun()

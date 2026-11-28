"""NyAI look and feel: stylesheet, sidebar shell and page headers.

Colours, radii and fonts that Streamlit can theme natively live in
.streamlit/config.toml; the CSS here covers what it can't. The citation graph
views draw in their own iframes and are not affected by it.
"""

from html import escape
from typing import Any, Dict, Optional, Sequence

import streamlit as st

from lib import api_client

CHAT_PAGE = "views/1_Chat.py"
# How long the sidebar status waits for the backend before calling it
# unreachable. Short, because every page draws the sidebar first.
STATUS_TIMEOUT = 4

_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap");

:root {
    --nyai-ink: #141B2D; --nyai-ink2: #3B4458; --nyai-muted: #6B7385;
    --nyai-navy: #1F3A8A; --nyai-navy-soft: #E8EDFB;
    --nyai-border: #E3E6EC; --nyai-subtle: #F3F4F7;
    --nyai-serif: "Source Serif 4", Georgia, serif;
    --nyai-shadow: 0 1px 2px rgba(20,26,46,.06), 0 4px 16px rgba(20,26,46,.06);
}

/* ---- page frame and top bar ---- */
[data-testid="stMainBlockContainer"] { padding: 0 2.5rem 6rem; }
[data-testid="stHeader"] { background: transparent; }
.st-key-nyai-topbar {
    margin: 0 -2.5rem; padding: 0 12rem 0 2.5rem; width: calc(100% + 5rem) !important; max-width: none !important;
    min-height: 60px; justify-content: center;
    background: #FFFFFF; border-bottom: 1px solid var(--nyai-border);
}
.st-key-nyai-topbar h1 {
    margin: 0; padding: 0; font-size: 17px; font-weight: 600; line-height: 1.4; color: var(--nyai-ink);
    overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
}

/* ---- sidebar ---- */
[data-testid="stSidebarUserContent"] { padding-top: 0; }
[data-testid="stSidebarHeader"] { height: 2rem; padding-bottom: 0; margin-bottom: 0; }
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { gap: .55rem; }
.nyai-brand { display: flex; align-items: center; gap: 10px; }
.nyai-logo {
    display: grid; place-items: center; width: 32px; height: 32px; border-radius: 8px;
    background: var(--nyai-navy); color: #FFFFFF;
    font-family: "Material Symbols Rounded"; font-size: 20px; line-height: 1;
}
.nyai-brand b { display: block; font-size: 16px; line-height: 1.2; color: var(--nyai-ink); }
.nyai-brand small { font-size: 11.5px; color: var(--nyai-muted); }
.nyai-label {
    font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    color: var(--nyai-muted); padding: 6px 10px 0;
}
[class*="st-key-nyai-nav-"] { margin-block: -.2rem; }
[class*="st-key-nyai-nav-"] a { padding: 7px 10px; border-radius: 8px; text-decoration: none; }
[class*="st-key-nyai-nav-"] a:hover { background: var(--nyai-subtle); }
.st-key-nyai-nav-current a, .st-key-nyai-nav-current a:hover { background: var(--nyai-navy-soft); }
.st-key-nyai-nav-current a * { color: var(--nyai-navy) !important; }
/* Pin the connection status to the bottom of the sidebar. */
[data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] { flex: 1; display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] > div { flex: 1; display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] { flex: 1; }
[data-testid="stSidebarUserContent"] div:has(> .st-key-nyai-status) { order: 99; margin-top: auto; }
.st-key-nyai-status { border-top: 1px solid var(--nyai-border); padding-top: 10px; }
.nyai-status { display: flex; gap: 8px; align-items: baseline; font-size: 12px; color: var(--nyai-muted); }
.nyai-status i { width: 7px; height: 7px; border-radius: 50%; flex: none; background: #1E7B4F; }
.nyai-status.off i { background: #B42318; }

/* ---- buttons, pills, links ---- */
[data-testid^="stBaseButton-primary"], [data-testid^="stBaseButton-secondary"] { font-weight: 600; }
[data-testid^="stBaseButton-secondary"] { background: #FFFFFF; }
button[data-variant="pills"] { border-radius: 999px; background: #FFFFFF; padding-inline: 14px; }

/* ---- cards: bordered containers, expanders, forms, metrics ---- */
[data-testid="stMain"] [data-testid="stExpander"] details,
[data-testid="stMain"] [data-testid="stForm"],
[data-testid="stMain"] [data-testid="stMetric"] {
    background: #FFFFFF; border-radius: 14px; box-shadow: var(--nyai-shadow);
}
.st-key-nyai-document { background: #FFFFFF; border-radius: 14px; box-shadow: var(--nyai-shadow); }
.st-key-nyai-document [data-testid="stText"] * { font-family: var(--nyai-serif); font-size: 16px; line-height: 1.7; }
[data-testid="stFileUploaderDropzone"] { background: #FFFFFF; border: 1px dashed #C3C9D6; border-radius: 14px; }

/* ---- conversation ---- */
[data-testid="stChatMessage"] {
    background: #FFFFFF; border: 1px solid var(--nyai-border); border-radius: 14px;
    padding: 16px 20px; box-shadow: var(--nyai-shadow);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--nyai-navy); border-color: var(--nyai-navy); border-bottom-right-radius: 4px;
    margin-left: auto; width: fit-content; max-width: 75%; box-shadow: none; padding: 12px 16px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) * { color: #FFFFFF; }
[data-testid="stChatMessageAvatarUser"] { display: none; }
[data-testid="stChatMessageAvatarAssistant"] { background: var(--nyai-navy-soft); color: var(--nyai-navy); }
/* Answers read like a judgment: serif body text. */
[data-testid="stChatMessageContent"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]
    > [data-testid="stMarkdown"] > div > [data-testid="stMarkdownContainer"]:not(:has(span[class*="stMarkdownBadge"])) p,
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] li,
.st-key-nyai-answer [data-testid="stMarkdownContainer"] p,
.st-key-nyai-answer [data-testid="stMarkdownContainer"] li {
    font-family: var(--nyai-serif); font-size: 16px; line-height: 1.7;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] :is(p, li) {
    font-family: Inter, sans-serif !important; font-size: 15px !important; line-height: 1.5 !important;
}
[data-testid="stChatInput"] > div { background: #FFFFFF; border-radius: 14px; box-shadow: var(--nyai-shadow); }
[data-testid="stBottomBlockContainer"] { background: #FAF8F4; }

/* ---- empty-state hero and home cards ---- */
.nyai-hero { text-align: center; padding: 56px 0 8px; }
.nyai-hero h2 {
    font-family: var(--nyai-serif); font-size: 40px; font-weight: 600; line-height: 1.2;
    margin: 0 0 10px; padding: 0; color: var(--nyai-ink);
}
.nyai-hero p { color: var(--nyai-muted); font-size: 15.5px; margin: 0; }
.nyai-note { text-align: center; color: var(--nyai-muted); font-size: 12.5px; padding-top: 8px; }
.st-key-nyai-home { max-width: 880px; margin-inline: auto; }
.st-key-home_suggestion { align-self: center; width: fit-content !important; }
.st-key-nyai-home [data-testid="stButtonGroup"] > div,
.st-key-nyai-empty [data-testid="stButtonGroup"] > div { justify-content: center; }
[class*="st-key-nyai-card-"] {
    position: relative; background: #FFFFFF; border: 1px solid var(--nyai-border); border-radius: 14px;
    padding: 20px; box-shadow: var(--nyai-shadow); transition: border-color .15s;
}
[class*="st-key-nyai-card-"]:hover, [class*="st-key-nyai-card-"]:focus-within { border-color: var(--nyai-navy); }
[class*="st-key-nyai-card-"] [data-testid="stElementContainer"] { position: static; }
[class*="st-key-nyai-card-"] [data-testid="stPageLink"] { margin-top: auto; }
[class*="st-key-nyai-card-"] [data-testid="stPageLink"] a { padding-left: 0; text-decoration: none; }
[class*="st-key-nyai-card-"] [data-testid="stPageLink"] a * { color: var(--nyai-navy); font-weight: 600; }
[class*="st-key-nyai-card-"] [data-testid="stPageLink"] a::after { content: ""; position: absolute; inset: 0; z-index: 1; }
.nyai-card h3 { font-size: 15px; font-weight: 600; margin: 14px 0 6px; padding: 0; }
.nyai-card p { font-size: 13px; color: var(--nyai-muted); margin: 0; line-height: 1.55; }
.nyai-icon {
    display: grid; place-items: center; width: 40px; height: 40px; border-radius: 10px;
    font-family: "Material Symbols Rounded"; font-size: 20px; line-height: 1;
}
.nyai-icon.navy { background: #E8EDFB; color: #1F3A8A; }
.nyai-icon.amber { background: #FBF2E2; color: #B7791F; }
.nyai-icon.green { background: #E3F4EA; color: #1E7B4F; }

@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 0 1rem 6rem; }
    .st-key-nyai-topbar { margin: 0 -1rem; padding: 0 8rem 0 4rem; width: calc(100% + 2rem) !important; max-width: none !important; }
    .nyai-hero { padding-top: 28px; }
    .nyai-hero h2 { font-size: 28px; }
}
</style>
"""


def apply_theme() -> None:
    st.html(_CSS)


@st.cache_data(ttl=30, show_spinner=False)
def backend_health() -> Dict[str, Any]:
    """The backend's /health reply, or {"error": ...} if it can't be reached.
    Failures are cached too, so a down backend doesn't stall every rerun."""
    try:
        return api_client.health(timeout=STATUS_TIMEOUT)
    except Exception as e:
        return {"error": str(e)}


def start_new_research() -> None:
    """Open the chat page on a fresh conversation."""
    st.session_state.chat_history = []
    st.session_state.chat_loaded = None
    st.session_state.pop("confirm_delete", None)
    st.switch_page(CHAT_PAGE)


def render_sidebar(nav: Dict[str, Sequence[Any]], current: Any) -> None:
    """Brand, "New research", navigation and connection status. `nav` maps a
    group label to its st.Page objects; `current` is the page being shown."""
    with st.sidebar:
        st.html(
            '<div class="nyai-brand"><span class="nyai-logo" aria-hidden="true">balance</span>'
            "<div><b>NyAI</b><small>Indian case-law research</small></div></div>"
        )
        if st.button("New research", icon=":material/add:", type="primary", width="stretch"):
            start_new_research()

        for group, pages in nav.items():
            st.html(f'<div class="nyai-label">{escape(group)}</div>')
            for page in pages:
                is_current = page.url_path == current.url_path
                state = "current" if is_current else "other"
                with st.container(key=f"nyai-nav-{state}" if is_current else f"nyai-nav-{state}-{page.url_path or 'home'}"):
                    st.page_link(page, width="stretch")

        with st.container(key="nyai-status"):
            health = backend_health()
            if health.get("status") == "healthy":
                model = "model loaded" if health.get("model_loaded") else "model not loaded"
                text, state = f"Connected · {model}", "on"
            elif "error" in health:
                text, state = "Backend unreachable", "off"
            else:
                text, state = f"Backend status: {health.get('status')}", "off"
            st.html(
                f'<div class="nyai-status {state}"><i></i><span>{escape(text)}<br>'
                f"{escape(api_client.BASE_URL)}</span></div>"
            )


def page_header(title: str, caption: Optional[str] = None) -> None:
    """The white bar across the top of a page, with an optional line under it."""
    with st.container(key="nyai-topbar"):
        st.html(f"<h1>{escape(title)}</h1>")
    if caption:
        st.caption(caption)


def hero(title: str, subtitle: str) -> None:
    st.html(f'<div class="nyai-hero"><h2>{escape(title)}</h2><p>{escape(subtitle)}</p></div>')

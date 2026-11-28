"""Entry point: page registry, shared theme and sidebar."""

import streamlit as st

from lib import home_page
from lib.theme import apply_theme, render_sidebar
from lib.ui_helpers import init_session_state

st.set_page_config(page_title="NyAI", page_icon=":material/balance:", layout="wide")
init_session_state()

NAV = {
    "Workspace": [
        st.Page(home_page.render, title="Home", icon=":material/home:", default=True),
        st.Page("views/1_Chat.py", title="Ask a question", icon=":material/chat_bubble:"),
        st.Page("views/3_Document_Upload.py", title="Analyse a judgment", icon=":material/upload_file:"),
        st.Page("views/4_Citation_Graph.py", title="Explore precedents", icon=":material/hub:"),
    ],
    "Tools": [
        st.Page("views/2_Single_Query.py", title="Single query", icon=":material/search:"),
        st.Page("views/6_Document.py", title="Document", icon=":material/description:"),
        st.Page("views/5_Admin_Debug.py", title="Admin / debug", icon=":material/tune:"),
    ],
}

# Navigation is drawn by render_sidebar, so Streamlit's own menu stays hidden.
page = st.navigation(NAV, position="hidden")
apply_theme()
render_sidebar(NAV, page)
page.run()

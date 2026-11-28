import streamlit as st

from lib import api_client
from lib.theme import page_header
from lib.ui_helpers import render_api_error

st.set_page_config(page_title="Document — NyAI", page_icon=":material/description:", layout="wide")

# Opened from citation links: Document?type=judgment&id=123 or
# Document?type=statute&id=<statute section ID>.
params = st.query_params
doc_type = params.get("type", "judgment")
if doc_type not in ("judgment", "statute"):
    doc_type = "judgment"
try:
    doc_id = int(params.get("id", ""))
except ValueError:
    doc_id = None

page_header("Document")

if doc_id is None:
    st.caption("Open a cited judgment or statute from an answer, or look up a judgment by ID.")
    with st.form("open_document", border=False):
        typed_id = st.number_input("Judgment ID", min_value=1, step=1)
        if st.form_submit_button("Open", icon=":material/open_in_new:", type="primary"):
            st.query_params.update({"type": "judgment", "id": str(int(typed_id))})
            st.rerun()
    st.stop()


@st.cache_data(ttl=300, show_spinner=False)
def _document(doc_type: str, doc_id: int):
    return api_client.document(doc_type, doc_id)


@st.cache_data(ttl=300, show_spinner=False)
def _pdf(doc_type: str, pdf_id: int) -> bytes:
    return api_client.document_pdf(doc_type, pdf_id)


try:
    doc = _document(doc_type, doc_id)
except Exception as e:
    render_api_error(e)
    st.stop()

st.subheader(doc["title"])

if doc_type == "judgment":
    details = [doc.get("court"), doc.get("date")]
    st.caption(" · ".join(str(d) for d in details if d) + f" · Judgment #{doc['judgment_id']}")
    if doc.get("bench"):
        st.caption("Bench: " + ", ".join(doc["bench"]))
    if doc.get("reporter_citations"):
        st.caption("Reported as: " + "; ".join(doc["reporter_citations"]))
    pdf_id, pdf_name = doc["judgment_id"], f"judgment_{doc['judgment_id']}.pdf"
else:
    if doc.get("heading"):
        st.caption(doc["heading"])
    pdf_id, pdf_name = doc["statute_id"], f"{doc.get('short_title') or 'statute'}.pdf"

if doc.get("has_pdf"):
    if st.toggle("Load original PDF", key="load_pdf"):
        try:
            st.download_button(
                "Download original PDF",
                data=_pdf(doc_type, pdf_id),
                file_name=pdf_name,
                mime="application/pdf",
                icon=":material/picture_as_pdf:",
            )
        except Exception as e:
            render_api_error(e)
else:
    st.caption("The original PDF isn't available for this document yet; showing the stored text.")

with st.container(height=640, border=True, key="nyai-document"):
    st.text(doc.get("text") or "(No text stored for this document.)")

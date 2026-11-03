"""Streamlit front end for Ask Your PDF."""
import os

import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer

from rag import EMBED_MODEL, PDFIndex, answer

st.set_page_config(page_title="Ask Your PDF", page_icon="📄", layout="wide")


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


def show_sources(sources, cited=()):
    with st.expander(f"Sources ({len(sources)} passages retrieved)"):
        for s in sources:
            mark = "✅ cited" if int(s["tag"][1:]) in cited else "not cited"
            st.markdown(f"**[{s['tag']}] {s['source']}, page {s['page']}** ({mark})")
            st.caption(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))


st.title("📄 Ask Your PDF")
st.caption("Upload PDFs, ask questions in plain English, get answers with page-level citations.")

api_key = get_api_key()
if not api_key:
    st.error("Missing GEMINI_API_KEY. Add it to .streamlit/secrets.toml (see README).")
    st.stop()
client = genai.Client(api_key=api_key)

with st.sidebar:
    st.header("1. Upload your PDFs")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    top_k = st.slider("Passages to retrieve", 3, 10, 5)
    if st.button("Build index", disabled=not files, use_container_width=True):
        with st.spinner("Reading and indexing..."):
            try:
                idx = PDFIndex(load_embedder())
                n = idx.build([(f.name, f) for f in files])
                st.session_state.index = idx
                st.session_state.messages = []
                st.success(f"Indexed {n} chunks from {len(files)} file(s).")
            except ValueError as e:
                st.error(str(e))
    st.header("2. Ask questions")
    st.caption("Use the chat box at the bottom.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"], m.get("cited", ()))

question = st.chat_input("Ask a question about your PDF")
if question:
    if "index" not in st.session_state:
        st.warning("Upload a PDF and click Build index first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF..."):
            chunks = st.session_state.index.search(question, k=top_k)
            try:
                result = answer(client, question, chunks)
            except Exception as e:
                st.error(f"Gemini API error: {e}")
                st.stop()
        st.markdown(result["answer"])
        if not result["grounded"]:
            st.warning("This answer has missing or invalid citations. Check it against the sources below.")
        sources = [{"tag": f"S{n}", "source": c.source, "page": c.page, "text": c.text}
                   for n, c in enumerate(chunks, start=1)]
        show_sources(sources, result["cited"])

    st.session_state.messages.append({
        "role": "assistant", "content": result["answer"],
        "sources": sources, "cited": result["cited"],
    })

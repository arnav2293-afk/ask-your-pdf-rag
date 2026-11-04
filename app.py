"""Streamlit front end for Ask Your PDF (v2)."""
import hashlib
import html
import io
import os
import re

import streamlit as st
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

import rag
from rag import EMBED_MODEL, PDFIndex, check_citations, stream_answer, suggest_questions

st.set_page_config(page_title="Ask Your PDF", page_icon="📄", layout="centered")

# Look: reading-room serif for answers, clean sans for controls,
# and page references marked like a highlighter on a printed page.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Literata:opsz,wght@7..72,400;7..72,600&display=swap');
html, body, .stMarkdown, button, input, textarea { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
h1 { font-family: 'Literata', Georgia, serif !important; font-weight: 600 !important; letter-spacing: -0.01em; }
[data-testid="stChatMessage"] .stMarkdown p,
[data-testid="stChatMessage"] .stMarkdown li { font-family: 'Literata', Georgia, serif; font-size: 1.02rem; line-height: 1.7; }
.cite { font-family: 'IBM Plex Sans', sans-serif; font-size: .78em; font-weight: 600; color: #1F2A44;
        background: linear-gradient(transparent 45%, #FFE066 45%); padding: 0 .25em; margin-left: .15em;
        white-space: nowrap; cursor: help; }
.tagline { color: #5B6478; margin-top: -0.6rem; font-size: 1.05rem; }
.docline { color: #5B6478; font-size: .92rem; margin: .2rem 0 1rem; }
.step { display: flex; gap: .9rem; align-items: baseline; margin: .5rem 0; }
.step b { color: #2F5BEA; font-family: 'Literata', serif; font-size: 1.15rem; min-width: 1rem; }
</style>
""", unsafe_allow_html=True)


# ---------- helpers ----------

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, "")


def render_answer(text, sources):
    """Turn [S2] tags into highlighted 'p. 4' marks. Hover shows the file name."""
    safe = html.escape(text, quote=False)

    def mark(m):
        out = []
        for n in (int(x) for x in re.findall(r"S(\d+)", m.group(1))):
            if 1 <= n <= len(sources):
                s = sources[n - 1]
                out.append(f'<span class="cite" title="{html.escape(s["source"])}">p.&nbsp;{s["page"]}</span>')
        return "".join(out)

    return re.sub(r"\[(S[^\]]*)\]", mark, safe)


def show_sources(sources, cited):
    used = [s for i, s in enumerate(sources, start=1) if i in set(cited)]
    rest = [s for i, s in enumerate(sources, start=1) if i not in set(cited)]
    label = f"See the {len(used)} passage{'s' if len(used) != 1 else ''} this answer used" if used else "See the passages that were searched"
    with st.expander(label):
        for s in used or rest:
            st.markdown(f"**Page {s['page']}**, {s['source']}")
            st.caption(s["text"][:700] + ("..." if len(s["text"]) > 700 else ""))
        if used and rest:
            pages = ", ".join(f"p. {s['page']}" for s in rest)
            st.caption(f"Also searched but not used: {pages}")


def grounding_note(msg):
    if msg.get("engine"):
        st.caption(f"Answered by {msg['engine']}")
    if msg.get("not_found"):
        return
    if msg["grounded"]:
        st.caption("✓ Every sentence points to a page in your PDF.")
    else:
        st.warning("Some sentences have no page reference. Check them against the passages below.")


def friendly_error(e):
    t = str(e)
    if "429" in t or "RESOURCE_EXHAUSTED" in t:
        return "Gemini's free limit was reached. Wait about a minute, then ask again."
    if "503" in t or "UNAVAILABLE" in t or "took too long" in t or "timed out" in t.lower():
        return "The AI service is very busy right now and did not answer after several tries. Wait a minute, then ask again."
    if "API key" in t or "401" in t or "403" in t:
        return "Gemini rejected the API key. Check GEMINI_API_KEY in your secrets."
    return f"Gemini could not answer this time: {t}"


def chat_markdown(messages):
    lines = ["# Ask Your PDF: chat export", ""]
    for m in messages:
        if m["role"] == "user":
            lines += [f"**Q:** {m['content']}", ""]
        else:
            refs = ", ".join(f"{s['source']} p. {s['page']}" for i, s in enumerate(m["sources"], 1) if i in m["cited"])
            lines += [f"**A:** {m['content']}", "", f"_Sources: {refs or 'none'}_", ""]
    return "\n".join(lines)


# ---------- setup ----------

gemini_key = get_secret("GEMINI_API_KEY")
groq_key = get_secret("GROQ_API_KEY")
if not gemini_key and not groq_key:
    st.error("Missing API keys. Add GEMINI_API_KEY and/or GROQ_API_KEY to .streamlit/secrets.toml (see README).")
    st.stop()

# Each request gives up after 20 s so one slow service can never freeze the app
client = genai.Client(api_key=gemini_key, http_options=types.HttpOptions(timeout=20000)) if gemini_key else None
rag.PRIMARY_PROVIDER = (get_secret("PRIMARY_PROVIDER") or "groq").lower()  # which AI to try first
groq_client = None
if groq_key:
    from groq import Groq
    groq_client = Groq(api_key=groq_key, timeout=20.0, max_retries=1)

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("index_key", None)
ss.setdefault("suggestions", [])

st.title("Ask Your PDF")
st.markdown('<p class="tagline">Answers from your document, with the page each one came from.</p>',
            unsafe_allow_html=True)

files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True,
                         label_visibility="collapsed",
                         help="Text-based PDFs work best. Scanned pages need OCR first.")

# Index automatically whenever the set of uploaded files changes
if files:
    key = hashlib.md5("|".join(f"{f.name}:{f.size}" for f in files).encode()).hexdigest()
    if key != ss.index_key:
        bar = st.progress(0.0, text="Reading your PDFs...")
        ss.index_key, ss.messages, ss.suggestions = key, [], []
        ss.pop("index", None)
        ss.pop("index_error", None)
        try:
            idx = PDFIndex(load_embedder())
            idx.build([(f.name, io.BytesIO(f.getvalue())) for f in files],
                      progress=lambda p, msg: bar.progress(p, text=msg))
            ss.index = idx
            ss.need_suggestions = True
        except ValueError as e:
            ss.index_error = str(e)
        bar.empty()
else:
    ss.index_key = None
    ss.pop("index", None)
    ss.pop("index_error", None)

with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("Passages to read per answer", 3, 10, 5,
                      help="More passages give more context but slower answers.")
    if ss.messages:
        st.download_button("Download this chat", data=chat_markdown(ss.messages),
                           file_name="ask-your-pdf-chat.md", use_container_width=True)
        if st.button("Start a new chat", use_container_width=True):
            ss.messages = []
            st.rerun()
    engines = [n for n, ok in (("Groq", groq_client), ("Gemini", client)) if ok]
    if rag.PRIMARY_PROVIDER == "gemini":
        engines.reverse()
    st.caption("AI order: " + " then ".join(engines))
    st.caption("Built by Arnav Shahi with FAISS, BM25, Gemini and Groq.")

# ---------- main area ----------

if ss.get("index_error"):
    st.error(ss.index_error)

if "index" not in ss:
    st.markdown("""
<div class="step"><b>1</b><span>Drop one or more PDFs in the box above.</span></div>
<div class="step"><b>2</b><span>Wait a few seconds while the app reads every page.</span></div>
<div class="step"><b>3</b><span>Ask anything. Each answer shows the page it came from, so you can check it.</span></div>
""", unsafe_allow_html=True)
else:
    idx = ss.index
    pages = sum(f["pages"] for f in idx.files)
    names = ", ".join(f["name"] for f in idx.files)
    st.markdown(f'<p class="docline">Reading {html.escape(names)}: {pages} pages, '
                f'{len(idx.chunks)} passages.</p>', unsafe_allow_html=True)

    if ss.pop("need_suggestions", False):
        with st.spinner("Thinking of questions you could ask..."):
            try:
                ss.suggestions = suggest_questions(client, idx, groq=groq_client)
            except Exception:
                ss.suggestions = []
    if not ss.messages and ss.suggestions:
        st.markdown("**Try asking**")
        for i, q in enumerate(ss.suggestions):
            if st.button(q, key=f"sugg{i}"):
                ss.pending = q
                st.rerun()

for m in ss.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            st.markdown(render_answer(m["content"], m["sources"]), unsafe_allow_html=True)
            grounding_note(m)
            show_sources(m["sources"], m["cited"])

typed = st.chat_input("Ask about your PDF" if "index" in ss else "Upload a PDF to start",
                      disabled="index" not in ss)
question = typed or ss.pop("pending", None)

if question and "index" in ss:
    # Earlier turns help with follow-ups like "explain that more simply"
    pairs = [(ss.messages[i]["content"], ss.messages[i + 1]["content"])
             for i in range(0, len(ss.messages) - 1, 2)][-2:]
    query = question
    if pairs and len(question.split()) < 8:
        query = f"{pairs[-1][0]} {question}"

    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    chunks = ss.index.search(query, k=top_k)
    sources = [{"source": c.source, "page": c.page, "text": c.text} for c in chunks]

    with st.chat_message("assistant"):
        box = st.empty()
        text = ""
        try:
            with st.spinner("Finding the right pages..."):
                engine = {}
                stream = stream_answer(client, question, chunks, pairs, groq=groq_client, info=engine)
                first = next(stream, "")
            text = first
            box.markdown(render_answer(text, sources) + " ▍", unsafe_allow_html=True)
            for piece in stream:
                text += piece
                box.markdown(render_answer(text, sources) + " ▍", unsafe_allow_html=True)
        except Exception as e:
            box.empty()
            st.error(friendly_error(e))
            ss.messages.pop()
            st.stop()

        text = text.strip()
        valid, invalid, grounded = check_citations(text, len(chunks))
        msg = {"role": "assistant", "content": text, "sources": sources, "cited": valid,
               "grounded": grounded, "not_found": "could not find" in text.lower(),
               "engine": engine.get("engine", "")}
        box.markdown(render_answer(text, sources), unsafe_allow_html=True)
        grounding_note(msg)
        show_sources(sources, valid)

    ss.messages.append(msg)

"""Streamlit front end for Ask Your PDF (v3: minimal single-column layout)."""
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

GITHUB_URL = "https://github.com/arnav2293-afk/ask-your-pdf-rag"
LINKEDIN_URL = "https://www.linkedin.com/in/arnav-shahi-b7a787369"

st.set_page_config(page_title="Ask Your PDF", page_icon="📄", layout="centered",
                   initial_sidebar_state="collapsed")

# Look: monochrome page, thin grey borders, one blue accent used only for focus and page marks.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {
  --ink: #111827;
  --muted: #6B7280;
  --line: #E5E7EB;
  --panel: #FAFAFA;
  --accent: #2563EB;
  --accent-soft: #EFF6FF;
}

html, body, .stApp, .stApp p, .stApp div, .stApp label, .stApp button, .stApp input, .stApp textarea, .stApp a,
.stApp span:not([data-testid="stIconMaterial"]) { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
.stApp { background: #FFFFFF; color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 1.25rem !important; }

/* top bar with a hairline border instead of a sidebar */
.st-key-topbar { border-bottom: 1px solid var(--line); padding-bottom: .75rem; margin-bottom: 3rem; }
.crumb { margin: 0; font-size: .95rem; color: var(--muted); }
.crumb a { color: var(--muted); text-decoration: none; }
.crumb a:hover { color: var(--accent); }
.crumb strong { color: var(--ink); font-weight: 600; }
.st-key-topbar [data-testid="stColumn"]:not(:first-child) [data-testid="stVerticalBlock"] { align-items: flex-end; }

/* title: large, thin, centred */
.title { font-weight: 300; font-size: 3.4rem; letter-spacing: -0.035em; line-height: 1.05; color: var(--ink); text-align: center; margin: 0; }
.subtitle { text-align: center; color: var(--muted); font-size: 1.08rem; margin: .8rem 0 2.4rem; }

/* full-width upload panel */
.st-key-upload [data-testid="stFileUploaderDropzone"] { background: var(--panel) !important; border: 1px dashed #D1D5DB !important; border-radius: 10px !important; }
.st-key-upload [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }
.status { color: var(--muted); font-size: .9rem; margin: .55rem 0 0; }

/* command bar: a clean search field, not a chat bubble */
.st-key-command { margin: 1.4rem 0 .4rem; }
.st-key-command [data-testid="stChatInput"] { background: #FFFFFF !important; border: 1px solid var(--line) !important; border-radius: 10px !important; box-shadow: 0 1px 2px rgba(17, 24, 39, .04); }
.st-key-command [data-testid="stChatInput"] > div { background: transparent !important; border: none !important; }
.st-key-command [data-testid="stChatInput"]:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-soft); }
.st-key-command textarea { font-size: 1rem !important; }

/* results */
.q { font-size: 1.12rem; font-weight: 500; color: var(--ink); margin: 2rem 0 .6rem; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { line-height: 1.7; }
.cite { font-size: .74em; font-weight: 600; color: var(--accent); background: var(--accent-soft); border: 1px solid #BFDBFE;
        border-radius: 4px; padding: 0 .35em; margin-left: .2em; white-space: nowrap; cursor: help; }
[data-testid="stExpander"] details { border: 1px solid var(--line) !important; border-radius: 8px; }
.stButton > button { border-radius: 8px !important; }

@media (max-width: 640px) { .title { font-size: 2.4rem; } .st-key-topbar { margin-bottom: 2rem; } }
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
    """Messages do not name one provider, because Groq or Gemini may have answered."""
    t = str(e)
    if "429" in t or "RESOURCE_EXHAUSTED" in t or "rate limit" in t.lower():
        return "The AI service hit its free usage limit. Wait about a minute, then ask again."
    if "503" in t or "UNAVAILABLE" in t or "took too long" in t or "timed out" in t.lower():
        return "The AI service is very busy right now and did not answer after several tries. Wait a minute, then ask again."
    if "API key" in t or "401" in t or "403" in t:
        return "An API key was rejected. Check GEMINI_API_KEY and GROQ_API_KEY in your secrets."
    return f"The AI service could not answer this time: {t}"


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

# ---------- top bar ----------
with st.container(key="topbar"):
    crumb, gh, settings = st.columns([7, 1.3, 1.5], vertical_alignment="center")
    crumb.markdown(f'<p class="crumb"><a href="{LINKEDIN_URL}" target="_blank">arnav-shahi</a> / '
                   '<strong>ask-your-pdf</strong></p>', unsafe_allow_html=True)
    gh.link_button("GitHub", GITHUB_URL)
    with settings.popover("Settings"):
        top_k = st.slider("Passages to read per answer", 3, 10, 5,
                          help="More passages give more context but slower answers.")
        if ss.messages:
            st.download_button("Download this chat", data=chat_markdown(ss.messages),
                               file_name="ask-your-pdf-chat.md")
            if st.button("Start a new chat"):
                ss.messages = []
                st.rerun()
        engines = [n for n, ok in (("Groq", groq_client), ("Gemini", client)) if ok]
        if rag.PRIMARY_PROVIDER == "gemini":
            engines.reverse()
        st.caption("AI order: " + " then ".join(engines))

st.markdown('<div class="title" role="heading" aria-level="1">Ask Your PDF</div>'
            '<p class="subtitle">Answers from your documents, indexed by page.</p>', unsafe_allow_html=True)

# ---------- upload panel ----------
with st.container(key="upload"):
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

    if ss.get("index_error"):
        st.error(ss.index_error)
    elif "index" in ss:
        idx = ss.index
        pages = sum(f["pages"] for f in idx.files)
        names = ", ".join(f["name"] for f in idx.files)
        st.markdown(f'<p class="status">Reading {html.escape(names)}: {pages} pages, '
                    f'{len(idx.chunks)} passages.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status">No document uploaded yet. Drag a PDF into the box above to begin.</p>',
                    unsafe_allow_html=True)

# ---------- command bar ----------
with st.container(key="command"):
    typed = st.chat_input("Ask a question about this document..." if "index" in ss
                          else "Upload a PDF above to start asking.",
                          disabled="index" not in ss)
question = typed or ss.pop("pending", None)

if "index" in ss and not question:
    if ss.pop("need_suggestions", False):
        with st.spinner("Thinking of questions you could ask..."):
            try:
                ss.suggestions = suggest_questions(client, ss.index, groq=groq_client)
            except Exception:
                ss.suggestions = []
    if not ss.messages and ss.suggestions:
        st.caption("Try asking")
        for i, q in enumerate(ss.suggestions):
            if st.button(q, key=f"sugg{i}"):
                ss.pending = q
                st.rerun()


# ---------- results: newest answer sits right under the command bar ----------
def render_turn(q, m):
    st.markdown(f'<p class="q">{html.escape(q)}</p>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(render_answer(m["content"], m["sources"]), unsafe_allow_html=True)
        grounding_note(m)
        show_sources(m["sources"], m["cited"])


live = st.container()
history = st.container()

with history:
    msgs = ss.messages
    for i in range(len(msgs) - 2, -1, -2):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            render_turn(msgs[i]["content"], msgs[i + 1])

if question and "index" in ss:
    # Earlier turns help with follow-ups like "explain that more simply"
    pairs = [(ss.messages[i]["content"], ss.messages[i + 1]["content"])
             for i in range(0, len(ss.messages) - 1, 2)][-2:]
    query = question
    if pairs and len(question.split()) < 8:
        query = f"{pairs[-1][0]} {question}"

    chunks = ss.index.search(query, k=top_k)
    sources = [{"source": c.source, "page": c.page, "text": c.text} for c in chunks]

    with live:
        st.markdown(f'<p class="q">{html.escape(question)}</p>', unsafe_allow_html=True)
        with st.container(border=True):
            box = st.empty()
            text = ""
            engine = {}
            try:
                with st.spinner("Finding the right pages..."):
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
                st.stop()

            text = text.strip()
            valid, invalid, grounded = check_citations(text, len(chunks))
            msg = {"role": "assistant", "content": text, "sources": sources, "cited": valid,
                   "grounded": grounded, "not_found": rag.is_not_found(text),
                   "engine": engine.get("engine", "")}
            box.markdown(render_answer(text, sources), unsafe_allow_html=True)
            grounding_note(msg)
            show_sources(sources, valid)
    ss.messages += [{"role": "user", "content": question}, msg]

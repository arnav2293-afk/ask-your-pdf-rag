"""Core RAG pipeline for Ask Your PDF: parse, chunk, index, retrieve, answer."""
import re
import time
from dataclasses import dataclass

import faiss
import numpy as np
from google.genai import types
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

EMBED_MODEL = "all-MiniLM-L6-v2"  # free, runs locally, 384-dim vectors
LLM_MODEL = "gemini-3.8-flash"  # change here if Google renames the model
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]  # tried if the main one is busy
GROQ_PREFERRED = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]
PROVIDER_COOLDOWN = 300  # after a provider fails, skip it for 5 minutes and try the other one first
BUSY_ERRORS = ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL")

CHUNK_SIZE = 900  # characters per chunk
CHUNK_OVERLAP = 150  # characters shared between neighbor chunks
NOT_FOUND = "I could not find that in the uploaded PDF."

SYSTEM_RULES = f"""You answer questions using ONLY the numbered sources you are given.
Rules:
1. End every factual sentence with the tag of the source it came from, like [S2].
2. If the sources do not contain the answer, reply exactly: {NOT_FOUND}
3. Never use outside knowledge. Keep the answer short, clear and in plain English."""


@dataclass
class Chunk:
    chunk_id: int
    source: str  # file name
    page: int  # 1-based page number
    text: str


def extract_pages(file):
    """Return ([(page_number, text), ...], total_pages) for one PDF."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages, len(reader.pages)


def chunk_page(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split one page into overlapping chunks, ending on a sentence where possible."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class PDFIndex:
    """Hybrid index: FAISS (meaning) + BM25 (exact keywords), merged with RRF."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks = []
        self.files = []  # [{"name": ..., "pages": ...}]
        self.faiss_index = None
        self.bm25 = None

    def build(self, files, progress=None):
        """files: list of (file_name, file_like_object). Returns chunk count.
        progress: optional function(fraction, message) for a progress bar."""
        self.chunks, self.files = [], []
        for n, (name, f) in enumerate(files, start=1):
            if progress:
                progress(0.6 * (n - 1) / len(files), f"Reading {name}...")
            pages, total = extract_pages(f)
            self.files.append({"name": name, "pages": total})
            for page_num, text in pages:
                for piece in chunk_page(text):
                    self.chunks.append(Chunk(len(self.chunks), name, page_num, piece))
        if progress:
            progress(0.65, "Understanding the text (embeddings)...")
        if not self.chunks:
            raise ValueError("No readable text found. The PDF may be a scanned image (needs OCR).")

        texts = [c.text for c in self.chunks]
        vecs = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype="float32")
        self.faiss_index = faiss.IndexFlatIP(vecs.shape[1])  # cosine, since vectors are normalized
        self.faiss_index.add(vecs)
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        if progress:
            progress(1.0, "Ready")
        return len(self.chunks)

    def search(self, query, k=5, pool=20, rrf_k=60, mode="hybrid"):
        """mode: "hybrid" (both), "dense" (FAISS only) or "bm25" (keywords only)."""
        pool = min(pool, len(self.chunks))
        q = self.embedder.encode([query], normalize_embeddings=True)
        _, idx = self.faiss_index.search(np.asarray(q, dtype="float32"), pool)
        dense_rank = [int(i) for i in idx[0] if i != -1]

        scores = self.bm25.get_scores(tokenize(query))
        bm25_rank = [int(i) for i in np.argsort(scores)[::-1][:pool] if scores[i] > 0]

        # Reciprocal Rank Fusion: a chunk ranked high by either method rises to the top
        fused = {}
        rankings = {"hybrid": (dense_rank, bm25_rank), "dense": (dense_rank,), "bm25": (bm25_rank,)}[mode]
        for ranking in rankings:
            for r, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + r + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.chunks[i] for i in top]


def build_prompt(question, chunks, history=None):
    """history: optional list of (question, answer) pairs from earlier turns."""
    blocks = [f"[S{n}] (file: {c.source}, page {c.page})\n{c.text}"
              for n, c in enumerate(chunks, start=1)]
    prompt = "SOURCES:\n\n" + "\n\n".join(blocks)
    if history:
        cite_tag = re.compile(r"\[S[^\]]*\]")
        turns = [f"Q: {q}\nA: {cite_tag.sub('', a).strip()}" for q, a in history]
        prompt += ("\n\nCONVERSATION SO FAR (only to understand follow-up questions; "
                   "facts must still come from SOURCES):\n" + "\n\n".join(turns))
    return prompt + f"\n\nQUESTION: {question}"


def is_not_found(text):
    """True when the model gave the fixed 'not in the PDF' reply."""
    return NOT_FOUND.lower()[:25] in text.lower()


def check_citations(text, n_sources):
    """Find [S#] tags (also handles [S1, S3]), flag tags that point nowhere,
    and flag any sentence that carries no tag at all.
    Returns (valid, invalid, grounded); grounded means every sentence is cited."""
    tags = re.findall(r"\[(S[^\]]*)\]", text)
    cited = sorted({int(n) for t in tags for n in re.findall(r"S(\d+)", t)})
    valid = [n for n in cited if 1 <= n <= n_sources]
    invalid = [n for n in cited if n not in valid]

    # Split into sentences: after . ! ? (unless a tag follows right away) or after a closing tag.
    # Very short pieces (3 words or fewer) are ignored, so stray fragments do not raise warnings.
    parts = re.split(r"(?<=[.!?])\s+(?!\[S)|(?<=\])\s+", text.strip())
    uncited = [s for s in parts if len(s.split()) > 3 and not re.search(r"\[S\d", s)]

    grounded = is_not_found(text) or (bool(valid) and not invalid and not uncited)
    return valid, invalid, grounded


def _models():
    return [LLM_MODEL] + [m for m in FALLBACK_MODELS if m != LLM_MODEL]


def _with_retry(call, budget=45):
    """Run call(model_name). If Gemini is busy, wait and retry, then move to the
    next model. Stops after `budget` seconds so the user is never left waiting long.
    Messages are printed so they show up in the Streamlit Cloud logs."""
    last_error, start = None, time.time()
    for model in _models():
        for attempt in range(2):
            if time.time() - start > budget:
                print(f"[gemini] gave up after {budget}s")
                raise last_error or TimeoutError("Gemini took too long")
            try:
                t0 = time.time()
                result = call(model)
                print(f"[gemini] {model} answered in {time.time() - t0:.1f}s")
                return result
            except Exception as e:
                last_error = e
                msg = str(e)
                print(f"[gemini] {model} attempt {attempt + 1} failed: {msg[:120]}")
                if "404" in msg or "NOT_FOUND" in msg:
                    break  # model not available to us, try the next one
                if any(code in msg for code in BUSY_ERRORS) or "timed out" in msg.lower():
                    time.sleep(2 * (attempt + 1))  # short back off: 2s, then 4s
                    continue
                raise  # e.g. bad API key: no point retrying
    raise last_error


def _config(temperature=0.2):
    return types.GenerateContentConfig(system_instruction=SYSTEM_RULES, temperature=temperature)


# ---------- circuit breaker: works for both providers ----------
_down_until = {"gemini": 0.0, "groq": 0.0}  # time until which each provider is skipped
_groq_model_cache = None


def _mark_down(provider, error):
    _down_until[provider] = time.time() + PROVIDER_COOLDOWN
    print(f"[{provider}] failed, trying it last for the next {PROVIDER_COOLDOWN // 60} min: {str(error)[:100]}")


# ---------- second provider: Groq ----------
def _groq_models(groq):
    """Ask Groq which models exist right now, so a retired model name never breaks the app."""
    global _groq_model_cache
    if _groq_model_cache is None:
        try:
            ids = [m.id for m in groq.models.list().data]
        except Exception as e:
            print(f"[groq] could not list models: {str(e)[:100]}")
            ids = []
        skip = ("whisper", "guard", "tts", "embed", "vision", "audio", "speech")
        chat = [i for i in ids if not any(w in i.lower() for w in skip)]
        ordered = [m for m in GROQ_PREFERRED if m in chat] + [m for m in chat if m not in GROQ_PREFERRED]
        _groq_model_cache = ordered or GROQ_PREFERRED
    return _groq_model_cache[:3]


def _groq_create(groq, system, prompt, temperature, stream, info=None):
    last_error = None
    for model in _groq_models(groq):
        try:
            resp = groq.chat.completions.create(
                model=model, temperature=temperature, stream=stream,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
            print(f"[groq] {model} answered")
            if info is not None:
                info["engine"] = f"Groq ({model})"
            return resp
        except Exception as e:
            last_error = e
            print(f"[groq] {model} failed: {str(e)[:120]}")
    raise last_error


# ---------- answering: try providers in order until one works ----------
PRIMARY_PROVIDER = "groq"  # "groq" or "gemini"; the other one is the backup


def _provider_order(client, groq):
    """Primary provider first, unless it failed recently. A provider that is
    cooling down is not dropped, it just moves to the end as a last resort."""
    order = [p for p, ok in (("groq", groq is not None), ("gemini", client is not None)) if ok]
    if PRIMARY_PROVIDER == "gemini":
        order.reverse()
    now = time.time()
    healthy = [p for p in order if now >= _down_until[p]]
    resting = [p for p in order if now < _down_until[p]]
    return healthy + resting


def _complete(client, groq, system, prompt, temperature, stream=False, info=None):
    """Return text (or a generator of text pieces when stream=True) from the
    first provider that works. Failures are logged and the next one is tried."""
    last_error = None
    for provider in _provider_order(client, groq):
        try:
            if provider == "groq":
                resp = _groq_create(groq, system, prompt, temperature, stream, info)
                if stream:
                    result = (c.choices[0].delta.content for c in resp
                              if c.choices and c.choices[0].delta.content)
                else:
                    result = resp.choices[0].message.content or ""
            else:
                cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
                budget = 15 if groq is not None else 45
                if stream:
                    def start(model):
                        it = iter(client.models.generate_content_stream(model=model, contents=prompt, config=cfg))
                        return next(it, None), it  # busy errors show up on the first piece

                    first, rest = _with_retry(start, budget=budget)

                    def pieces():
                        if first is not None and first.text:
                            yield first.text
                        for part in rest:
                            if part.text:
                                yield part.text

                    result = pieces()
                else:
                    resp = _with_retry(lambda m: client.models.generate_content(
                        model=m, contents=prompt, config=cfg), budget=budget)
                    result = resp.text or ""
                if info is not None:
                    info["engine"] = "Gemini"
            _down_until[provider] = 0.0  # it answered, so it is healthy again
            return result
        except Exception as e:
            last_error = e
            _mark_down(provider, e)
    raise last_error or RuntimeError("No AI provider is configured. Add an API key.")


def answer(client, question, chunks, history=None, groq=None, info=None):
    text = _complete(client, groq, SYSTEM_RULES, build_prompt(question, chunks, history), 0.2, info=info).strip()
    valid, invalid, grounded = check_citations(text, len(chunks))
    return {"answer": text, "cited": valid, "invalid": invalid, "grounded": grounded}


def stream_answer(client, question, chunks, history=None, groq=None, info=None):
    """Yield the answer piece by piece so the user sees it being written."""
    yield from _complete(client, groq, SYSTEM_RULES, build_prompt(question, chunks, history),
                         0.2, stream=True, info=info)


def suggest_questions(client, index, n=3, groq=None):
    """Ask the LLM for a few starter questions based on the first pages."""
    sample = " ".join(c.text for c in index.chunks[:8])[:5000]
    prompt = (f"Here is the start of a document:\n\n{sample}\n\n"
              f"Write {n} short questions (under 12 words each) that this text clearly answers. "
              "One per line, no numbering, no extra text.")
    text = _complete(client, groq, "You write short questions about documents.", prompt, 0.4)
    lines = [re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip() for line in text.splitlines()]
    return [line for line in lines if line.endswith("?")][:n]

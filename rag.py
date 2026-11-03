"""Core RAG pipeline for Ask Your PDF: parse, chunk, index, retrieve, answer."""
import re
from dataclasses import dataclass

import faiss
import numpy as np
from google.genai import types
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

EMBED_MODEL = "all-MiniLM-L6-v2"   # free, runs locally, 384-dim vectors
LLM_MODEL = "gemini-3.8-flash"      # change here if Google renames the model
CHUNK_SIZE = 900                    # characters per chunk
CHUNK_OVERLAP = 150                 # characters shared between neighbor chunks

NOT_FOUND = "I could not find that in the uploaded PDF."

SYSTEM_RULES = f"""You answer questions using ONLY the numbered sources you are given.
Rules:
1. End every factual sentence with the tag of the source it came from, like [S2].
2. If the sources do not contain the answer, reply exactly: {NOT_FOUND}
3. Never use outside knowledge. Keep the answer short, clear and in plain English."""


@dataclass
class Chunk:
    chunk_id: int
    source: str   # file name
    page: int     # 1-based page number
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
        self.files = []   # [{"name": ..., "pages": ...}]
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


def check_citations(text, n_sources):
    """Find [S#] tags (also handles [S1, S3]) and flag any that point nowhere."""
    tags = re.findall(r"\[(S[^\]]*)\]", text)
    cited = sorted({int(n) for t in tags for n in re.findall(r"S(\d+)", t)})
    valid = [n for n in cited if 1 <= n <= n_sources]
    invalid = [n for n in cited if n not in valid]
    not_found = NOT_FOUND.lower()[:25] in text.lower()
    grounded = not_found or (bool(valid) and not invalid)
    return valid, invalid, grounded


def _config(temperature=0.2):
    return types.GenerateContentConfig(system_instruction=SYSTEM_RULES, temperature=temperature)


def answer(client, question, chunks, history=None):
    resp = client.models.generate_content(
        model=LLM_MODEL,
        contents=build_prompt(question, chunks, history),
        config=_config(),
    )
    text = (resp.text or "").strip()
    valid, invalid, grounded = check_citations(text, len(chunks))
    return {"answer": text, "cited": valid, "invalid": invalid, "grounded": grounded}


def stream_answer(client, question, chunks, history=None):
    """Yield the answer piece by piece so the user sees it being written."""
    for part in client.models.generate_content_stream(
        model=LLM_MODEL,
        contents=build_prompt(question, chunks, history),
        config=_config(),
    ):
        if part.text:
            yield part.text


def suggest_questions(client, index, n=3):
    """Ask Gemini for a few starter questions based on the first pages."""
    sample = " ".join(c.text for c in index.chunks[:8])[:5000]
    prompt = (f"Here is the start of a document:\n\n{sample}\n\n"
              f"Write {n} short questions (under 12 words each) that this text clearly answers. "
              "One per line, no numbering, no extra text.")
    resp = client.models.generate_content(
        model=LLM_MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=0.4),
    )
    lines = [re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip() for line in (resp.text or "").splitlines()]
    return [line for line in lines if line.endswith("?")][:n]

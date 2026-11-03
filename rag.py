"""Core RAG pipeline for Ask Your PDF: parse, chunk, index, retrieve, answer."""
import re
from dataclasses import dataclass

import faiss
import numpy as np
from google.genai import types
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

EMBED_MODEL = "all-MiniLM-L6-v2"   # free, runs locally, 384-dim vectors
LLM_MODEL = "gemini-2.5-flash"      # change here if Google renames the model
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
    """Return a list of (page_number, text) for every page that has text."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


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
        self.faiss_index = None
        self.bm25 = None

    def build(self, files):
        """files: list of (file_name, file_like_object). Returns chunk count."""
        self.chunks = []
        for name, f in files:
            for page_num, text in extract_pages(f):
                for piece in chunk_page(text):
                    self.chunks.append(Chunk(len(self.chunks), name, page_num, piece))
        if not self.chunks:
            raise ValueError("No readable text found. The PDF may be a scanned image (needs OCR).")

        texts = [c.text for c in self.chunks]
        vecs = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype="float32")
        self.faiss_index = faiss.IndexFlatIP(vecs.shape[1])  # cosine, since vectors are normalized
        self.faiss_index.add(vecs)
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
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


def build_prompt(question, chunks):
    blocks = [f"[S{n}] (file: {c.source}, page {c.page})\n{c.text}"
              for n, c in enumerate(chunks, start=1)]
    return "SOURCES:\n\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}"


def check_citations(text, n_sources):
    """Find [S#] tags (also handles [S1, S3]) and flag any that point nowhere."""
    tags = re.findall(r"\[(S[^\]]*)\]", text)
    cited = sorted({int(n) for t in tags for n in re.findall(r"S(\d+)", t)})
    valid = [n for n in cited if 1 <= n <= n_sources]
    invalid = [n for n in cited if n not in valid]
    not_found = NOT_FOUND.lower()[:25] in text.lower()
    grounded = not_found or (bool(valid) and not invalid)
    return valid, invalid, grounded


def answer(client, question, chunks):
    resp = client.models.generate_content(
        model=LLM_MODEL,
        contents=build_prompt(question, chunks),
        config=types.GenerateContentConfig(system_instruction=SYSTEM_RULES, temperature=0.2),
    )
    text = (resp.text or "").strip()
    valid, invalid, grounded = check_citations(text, len(chunks))
    return {"answer": text, "cited": valid, "invalid": invalid, "grounded": grounded}

# 📄 Ask Your PDF: RAG Document Chatbot

Upload one or more PDFs, ask questions in plain English, and get short answers that cite the exact file and page they came from. If the answer is not in the PDF, the bot says so instead of guessing.

**Live demo:** [add your Streamlit link here]

## How it works

1. **Parse:** `pypdf` reads each page and keeps the page number.
2. **Chunk:** each page is split into ~900 character chunks with 150 characters of overlap, ending on full sentences where possible.
3. **Index (hybrid):**
   - `sentence-transformers` (all-MiniLM-L6-v2) turns chunks into vectors stored in a **FAISS** index, which finds passages by meaning.
   - **BM25** finds passages by exact keywords (names, numbers, codes).
   - **Reciprocal Rank Fusion** merges both rankings.
4. **Answer:** the top passages go to **Gemini** with strict rules: use only these sources, tag every sentence with [S#], or reply "not found".
5. **Citation check:** the app verifies every [S#] tag points to a real retrieved passage and warns the user when an answer is uncited.

## Tech stack

Python, Streamlit, pypdf, sentence-transformers, FAISS, rank-bm25, Google Gemini API, pandas

## Run it locally

```bash
git clone https://github.com/arnav2293-afk/ask-your-pdf-rag.git
cd ask-your-pdf-rag
python -m venv venv
# Windows: venv\Scripts\activate    Mac: source venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your Gemini key
streamlit run app.py
```

Get a free Gemini API key at https://aistudio.google.com

## Evaluation

`eval.py` scores retrieval against a hand-written question set (`eval_questions.csv`, each question paired with the page that holds the answer).

```bash
python eval.py              # retrieval metrics only
python eval.py --with-llm   # also checks citation coverage
python eval.py --mode dense # FAISS only (compare against hybrid)
python eval.py --mode bm25  # keywords only
```

| Metric | Dense only | BM25 only | Hybrid |
|---|---|---|---|
| Test PDFs | [fill in] | | |
| Questions | [fill in] | | |
| Hit rate @ 5 | [fill in] | [fill in] | [fill in] |
| MRR | [fill in] | [fill in] | [fill in] |
| Fully cited answers | | | [fill in] |

## Limitations

- Scanned PDFs (images of text) need OCR first; the app shows a message when no text is found.
- Tables are read as plain text, so complex tables may lose structure.
- The index lives in memory and resets when the app restarts.

## Author

Arnav Shahi | [GitHub](https://github.com/arnav2293-afk) | [LinkedIn](https://www.linkedin.com/in/arnav-shahi-b7a787369) | B.Tech AIML, 2nd year

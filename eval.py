"""Measure retrieval quality: hit rate @ k and MRR for dense, BM25 and hybrid search.

Usage:
    python eval.py                  # all three modes, prints a table you can paste into the README
    python eval.py --mode hybrid    # one mode only: hybrid, dense or bm25
    python eval.py --with-llm       # also checks how many hybrid answers cite every sentence
                                    # (needs GROQ_API_KEY and/or GEMINI_API_KEY in your environment)
"""
import argparse
import glob
import os
import time

import pandas as pd
from sentence_transformers import SentenceTransformer

from rag import EMBED_MODEL, PDFIndex, answer

MODES = ["dense", "bm25", "hybrid"]
NAMES = {"dense": "Dense only", "bm25": "BM25 only", "hybrid": "Hybrid"}


def load_llm_clients():
    client, groq = None, None
    if os.getenv("GEMINI_API_KEY"):
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    if os.getenv("GROQ_API_KEY"):
        from groq import Groq
        groq = Groq(api_key=os.environ["GROQ_API_KEY"])
    if client is None and groq is None:
        raise SystemExit("--with-llm needs GROQ_API_KEY or GEMINI_API_KEY set in your environment.")
    return client, groq


def score(idx, qs, k, mode):
    """One row per question: did the expected page show up in the top k, and at what rank."""
    rows = []
    for _, r in qs.iterrows():
        hits = idx.search(r["question"], k=k, mode=mode)
        want_file = r.get("file")
        ranks = [i + 1 for i, c in enumerate(hits)
                 if c.page == int(r["expected_page"])
                 and (pd.isna(want_file) or c.source == want_file)]
        rows.append({"question": r["question"], "mode": mode, "hit": bool(ranks),
                     "reciprocal_rank": 1.0 / ranks[0] if ranks else 0.0})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdfs", default="sample_pdfs")
    p.add_argument("--questions", default="eval_questions.csv")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--mode", default="all", choices=["all"] + MODES)
    p.add_argument("--with-llm", action="store_true")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.pdfs, "*.pdf")))
    if not paths:
        raise SystemExit(f"No PDFs found in {args.pdfs}/")
    qs = pd.read_csv(args.questions)
    if qs.empty:
        raise SystemExit(f"No questions in {args.questions}")

    idx = PDFIndex(SentenceTransformer(EMBED_MODEL))
    handles = [(os.path.basename(pth), open(pth, "rb")) for pth in paths]
    n_chunks = idx.build(handles)
    for _, h in handles:
        h.close()

    modes = MODES if args.mode == "all" else [args.mode]
    results = pd.concat([score(idx, qs, args.k, m) for m in modes], ignore_index=True)
    results.to_csv("eval_results.csv", index=False)

    summary = results.groupby("mode", sort=False).agg(hit=("hit", "mean"), mrr=("reciprocal_rank", "mean"))
    print(f"PDFs: {len(paths)} | chunks: {n_chunks} | questions: {len(qs)} | k = {args.k}\n")
    print("| Metric | " + " | ".join(NAMES[m] for m in summary.index) + " |")
    print("|---" * (len(summary) + 1) + "|")
    print(f"| Hit rate @ {args.k} | " + " | ".join(f"{v:.1%}" for v in summary["hit"]) + " |")
    print("| MRR | " + " | ".join(f"{v:.3f}" for v in summary["mrr"]) + " |")

    if args.with_llm:
        client, groq = load_llm_clients()
        grounded = []
        for _, r in qs.iterrows():
            hits = idx.search(r["question"], k=args.k, mode="hybrid")
            grounded.append(answer(client, r["question"], hits, groq=groq)["grounded"])
            time.sleep(1)  # stay under free-tier rate limits
        print(f"\nFully cited answers (hybrid, k = {args.k}): {sum(grounded) / len(grounded):.1%}")


if __name__ == "__main__":
    main()

"""Measure retrieval quality so the resume bullet has real numbers.

Usage:
    python eval.py                 # retrieval only (free, no API calls)
    python eval.py --with-llm      # also checks how many answers are fully cited
    python eval.py --mode dense    # compare: FAISS only, BM25 only, or hybrid
"""
import argparse
import glob
import os

import pandas as pd
from sentence_transformers import SentenceTransformer

from rag import EMBED_MODEL, PDFIndex, answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdfs", default="sample_pdfs")
    p.add_argument("--questions", default="eval_questions.csv")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--with-llm", action="store_true")
    p.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.pdfs, "*.pdf")))
    if not paths:
        raise SystemExit(f"No PDFs found in {args.pdfs}/")

    idx = PDFIndex(SentenceTransformer(EMBED_MODEL))
    handles = [(os.path.basename(pth), open(pth, "rb")) for pth in paths]
    n_chunks = idx.build(handles)
    for _, h in handles:
        h.close()

    client = None
    if args.with_llm:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    qs = pd.read_csv(args.questions)
    rows = []
    for _, r in qs.iterrows():
        hits = idx.search(r["question"], k=args.k, mode=args.mode)
        want_file = r.get("file")
        ranks = [i + 1 for i, c in enumerate(hits)
                 if c.page == int(r["expected_page"])
                 and (pd.isna(want_file) or c.source == want_file)]
        row = {"question": r["question"],
               "hit": bool(ranks),
               "reciprocal_rank": 1.0 / ranks[0] if ranks else 0.0}
        if client:
            row["grounded"] = answer(client, r["question"], hits)["grounded"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("eval_results.csv", index=False)
    print(f"PDFs: {len(paths)} | pages chunked into {n_chunks} chunks | questions: {len(df)}")
    print(f"Mode: {args.mode}")
    print(f"Hit rate @ {args.k}: {df['hit'].mean():.1%}")
    print(f"MRR: {df['reciprocal_rank'].mean():.3f}")
    if client:
        print(f"Fully cited answers: {df['grounded'].mean():.1%}")


if __name__ == "__main__":
    main()

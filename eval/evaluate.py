"""
evaluate.py — Evaluation framework for the RAG pipeline.

Measures three things, which map directly onto the project's three
interview talking points:

  1. Retrieval quality  -> did chunking/embedding find the right section?
     (keyword hit rate + expected-section match in top-k retrieved chunks)

  2. Groundedness        -> hallucination check.
     Every sentence in the generated answer that isn't a hedge ("the
     filings do not specify...") should be traceable to a citation [n],
     We flag citation-free factual-looking sentences, but do not verify
     that a cited excerpt supports a claim.

  3. Citation validity   -> do citation numbers in the answer actually
     correspond to chunks that were retrieved (no fabricated indices)?

This is intentionally a lightweight, rule-based eval (not an LLM judge)
so scoring is fast and deterministic (live generation can vary and uses Groq) — a reasonable engineering
trade-off for a portfolio project, but worth naming as a limitation:
it can't catch subtle factual errors within a correctly-cited chunk.
"""

import sys
import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
HEDGE_PHRASES = [
    "does not specify", "do not specify", "not mentioned", "cannot find",
    "not enough information", "context does not", "excerpts do not",
]


def load_dataset():
    with open(DATASET_PATH) as f:
        return json.load(f)


def score_retrieval(item, retrieved_chunks):
    """Check whether retrieval surfaced a chunk matching expected keywords
    and (if specified) the expected filing section."""
    normalize = lambda value: value.lower().replace("’", "'")
    combined_text = " ".join(c["text"].lower() for c in retrieved_chunks)
    keyword_hit = any(kw.lower() in combined_text for kw in item["expected_keywords"])

    section_hit = True  # default true if no section expectation set
    if item.get("expected_section_contains"):
        section_hit = any(
            normalize(item["expected_section_contains"]) in normalize(c["section"])
            for c in retrieved_chunks
        )
    return {"keyword_hit": keyword_hit, "section_hit": section_hit}


def score_groundedness(answer_text, num_sources):
    """Flag sentences that look like factual claims but carry no citation,
    and flag any citation index that doesn't correspond to a real source."""
    sentences = re.split(r"(?<=[.!?])\s+", answer_text)
    uncited_claims = []
    for sent in sentences:
        if not sent.strip():
            continue
        is_hedge = any(h in sent.lower() for h in HEDGE_PHRASES)
        has_citation = bool(CITATION_PATTERN.search(sent))
        if not is_hedge and not has_citation and len(sent.split()) > 4:
            uncited_claims.append(sent.strip())

    cited_indices = {int(n) for n in CITATION_PATTERN.findall(answer_text)}
    invalid_citations = [i for i in cited_indices if i < 1 or i > num_sources]

    return {
        "uncited_claim_count": len(uncited_claims),
        "uncited_claims": uncited_claims,
        "invalid_citations": invalid_citations,
    }


def run_eval(k=5, verbose=True, retrieval_only=False):
    """Full eval loop. Requires a built index + GROQ_API_KEY (calls the
    live pipeline). Use retrieval_only=True to skip generation calls."""
    from vectorstore import VectorStore
    import generator

    store = VectorStore()
    store.load()
    dataset = load_dataset()

    results = []
    for item in dataset:
        retrieved = store.search(item["question"], k=k)
        retrieval_scores = score_retrieval(item, retrieved)

        if retrieval_only:
            groundedness_scores = {"uncited_claim_count": None, "uncited_claims": [], "invalid_citations": []}
        else:
            gen_result = generator.answer_question(item["question"], retrieved)
            groundedness_scores = score_groundedness(gen_result["answer"], len(retrieved))

        row = {
            "question": item["question"],
            **retrieval_scores,
            **groundedness_scores,
        }
        results.append(row)
        if verbose:
            print(f"\nQ: {item['question']}")
            print(f"  retrieval: keyword_hit={row['keyword_hit']} "
                  f"section_hit={row['section_hit']}")
            print(f"  groundedness: uncited_claims={row['uncited_claim_count']} "
                  f"invalid_citations={row['invalid_citations']}")

    n = len(results)
    summary = {
        "keyword_hit_rate": sum(r["keyword_hit"] for r in results) / n,
        "section_hit_rate": sum(r["section_hit"] for r in results) / n,
        "avg_uncited_claims_per_answer": None if retrieval_only else sum(r["uncited_claim_count"] for r in results) / n,
        "total_invalid_citations": None if retrieval_only else sum(len(r["invalid_citations"]) for r in results),
    }
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for k_, v in summary.items():
        print(f"  {k_}: {v:.2f}" if isinstance(v, float) else f"  {k_}: {v}")

    return {"per_question": results, "summary": summary}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    run_eval(retrieval_only=args.retrieval_only)

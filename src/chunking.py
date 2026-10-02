"""
chunking.py — Splits filing text into retrievable chunks.

Strategy (this is the interview talking point):
  - SEC filings are long (10-Ks run 100+ pages) and structured into numbered
    Items (e.g. "Item 7. Management's Discussion and Analysis"). We first
    try to split on these section headers so a chunk never straddles two
    unrelated topics (e.g. half "Risk Factors", half "Legal Proceedings").
  - Within a section, we do fixed-size sliding-window chunking with overlap.
    Overlap prevents a fact from being severed exactly at a chunk boundary
    (e.g. "revenue grew 122%" split from the sentence naming what quarter).
  - Chunk size is measured in words, not characters, since it maps more
    predictably to token count for the embedding model's context window.

Trade-off we accept: larger chunks retain more context per retrieval (better
for narrative sections like MD&A) but dilute the embedding's ability to
match narrow queries (worse for pulling out a single number). We settled on
~250 words with 50-word overlap as a middle ground, the eval framework (see eval/evaluate.py) can be used to assess this choice.
"""

import re

SECTION_PATTERN = re.compile(
    r"(Item\s+\d+[A-Za-z]?\.\s+[A-Z][^\n]{3,80})", re.IGNORECASE
)


def split_into_sections(text):
    """Split filing text on 'Item N.' headers. Falls back to one section
    if no headers are found (e.g. exhibit documents)."""
    parts = SECTION_PATTERN.split(text)
    if len(parts) <= 1:
        return [{"heading": "Full Document", "text": text}]

    sections = []
    # parts alternates: [preamble, heading1, body1, heading2, body2, ...]
    if parts[0].strip():
        sections.append({"heading": "Preamble", "text": parts[0]})
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append({"heading": heading, "text": body})
    return sections


def chunk_text(text, chunk_size=250, overlap=50):
    """Sliding-window word-based chunking with overlap."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap  # step forward, but re-include the tail
    return chunks


def chunk_document(text, form, report_date, chunk_size=250, overlap=50):
    """Full pipeline: section-split, then window-chunk each section.
    Returns a list of chunk dicts with metadata for citation later."""
    sections = split_into_sections(text)
    all_chunks = []
    for sec in sections:
        sub_chunks = chunk_text(sec["text"], chunk_size, overlap)
        for idx, ct in enumerate(sub_chunks):
            all_chunks.append({
                "text": ct,
                "section": sec["heading"],
                "form": form,
                "report_date": report_date,
                "chunk_index": idx,
            })
    return all_chunks


if __name__ == "__main__":
    sample = (
        "Item 7. Management's Discussion and Analysis\n"
        "Revenue for fiscal year 2025 increased 122% year-over-year, "
        "driven primarily by strong demand for our Data Center platform. "
        * 20 +
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk\n"
        "We are exposed to foreign currency risk. " * 20
    )
    chunks = chunk_document(sample, form="10-K", report_date="2025-01-26")
    print(f"Produced {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"  [{c['section'][:40]}] chunk {c['chunk_index']}: "
              f"{c['text'][:80]}...")

"""
generator.py — Calls Groq (openai/gpt-oss-120b) to answer a question using
retrieved chunks, with two deliberate hallucination-prevention measures:

  1. A strict system prompt that instructs the model to answer ONLY from
     the provided context and to explicitly say when the context doesn't
     contain the answer, rather than filling the gap from parametric
     knowledge.
  2. A structured output format that forces every answer to cite which
     chunk(s) it drew from ([1], [2], ...), so a claim with no citation
     is visibly ungrounded, and a human (or the eval framework) can check
     the cited chunk actually supports the claim.

This doesn't eliminate hallucination, but it makes it detectable — an
unsupported or wrongly-cited claim shows up as a mismatch during eval
instead of blending in invisibly.
"""

import os
from groq import Groq

SYSTEM_PROMPT = """You are a financial research assistant answering questions about NVIDIA using excerpts from its SEC filings (10-K / 10-Q).

Rules:
1. Answer ONLY using the numbered context excerpts provided. Do not use outside knowledge about NVIDIA, even if you're confident it's correct.
2. Every factual claim must end with a citation to the excerpt number it came from, like [1] or [2][3].
3. If the excerpts don't contain enough information to answer, say so explicitly instead of guessing. Do not fill gaps with plausible-sounding numbers.
4. Be precise with figures (percentages, dollar amounts, dates) — copy them exactly as they appear in the excerpts, don't round or approximate.
"""


def format_context(chunks):
    """Turn retrieved chunks into a numbered context block the model can
    cite by index, and return a lookup for displaying sources afterward."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] (Source: NVIDIA {c['form']}, {c['report_date']}, "
            f"section: {c['section']})\n{c['text']}\n"
        )
    return "\n".join(lines)


def answer_question(question, chunks, model=None):
    """Send the question + retrieved context to Groq, return the answer
    text plus the source list so the frontend can render citations."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=40.0, max_retries=0)
    model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    context_block = format_context(chunks)
    user_prompt = (
        f"Context excerpts:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the excerpts above, with citations."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # low temperature: favor faithfulness over creativity
    )

    answer = resp.choices[0].message.content
    sources = [
        {
            "index": i + 1,
            "form": c["form"],
            "report_date": c["report_date"],
            "section": c["section"],
            "score": c.get("score"),
            "url": c.get("source_url"),
            "excerpt": c["text"],
        }
        for i, c in enumerate(chunks)
    ]
    return {"answer": answer, "sources": sources}

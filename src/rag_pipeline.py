"""
rag_pipeline.py — Orchestrates the full pipeline:
  ingest filings -> chunk -> embed & index -> retrieve -> generate

Usage:
    python src/rag_pipeline.py build     # one-time: fetch filings + build index
    python src/rag_pipeline.py ask "What was NVIDIA's data center revenue?"
"""

import sys
import os
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

import ingest
import chunking
from vectorstore import VectorStore
import generator


def build_index():
    docs = ingest.run(forms=("10-K", "10-Q"), limit=6)

    all_chunks = []
    for doc in docs:
        with open(doc["source_file"], "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunking.chunk_document(
            text, form=doc["form"], report_date=doc["report_date"]
        )
        for chunk in chunks:
            chunk["source_url"] = doc.get("source_url")
        all_chunks.extend(chunks)
        print(f"  {doc['form']} {doc['report_date']}: {len(chunks)} chunks")

    print(f"Total chunks: {len(all_chunks)}")

    store = VectorStore()
    store.build(all_chunks)
    store.save()
    print("Index built and saved to data/processed/")


def ask(question, k=5):
    store = VectorStore()
    store.load()
    retrieved = store.search(question, k=k)
    result = generator.answer_question(question, retrieved)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    if command == "build":
        build_index()
    elif command == "ask":
        question = " ".join(sys.argv[2:])
        result = ask(question)
        print("\nANSWER:\n" + result["answer"])
        print("\nSOURCES:")
        for s in result["sources"]:
            print(f"  [{s['index']}] {s['form']} {s['report_date']} "
                  f"- {s['section']} (score: {s['score']:.3f})")
    else:
        print(f"Unknown command: {command}")
        print(__doc__)

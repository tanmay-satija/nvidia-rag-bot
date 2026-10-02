"""
main.py — FastAPI backend exposing the RAG pipeline as an HTTP API.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError, APIStatusError
from dotenv import load_dotenv

from vectorstore import VectorStore
import generator

load_dotenv()

app = FastAPI(title="NVIDIA Filings RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_store = None  # lazy-loaded singleton so the embedding model loads once


def get_store():
    global _store
    if _store is None:
        from vectorstore import INDEX_DIR
        if not all(os.path.isfile(os.path.join(INDEX_DIR, name)) for name in ("index.faiss", "chunks.pkl")):
            raise FileNotFoundError("Index not built yet")
        store = VectorStore()
        store.load()
        _store = store
    return _store


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=5, ge=1, le=10)


class Source(BaseModel):
    index: int
    form: str
    report_date: str
    section: str
    score: float | None = None
    url: str | None = None
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
def root():
    return {
        "name": "NVIDIA Filings RAG API",
        "health": "/health",
        "query": "POST /query",
        "docs": "/docs",
        "ui": "streamlit run frontend/app.py",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        store = get_store()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Index not built yet. Run: python src/rag_pipeline.py build",
        )

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=503, detail="Set GROQ_API_KEY in .env and restart the backend.")
    try:
        retrieved = store.search(req.question, k=req.k)
        return generator.answer_question(req.question, retrieved)
    except AuthenticationError:
        raise HTTPException(status_code=503, detail="Groq authentication failed. Check GROQ_API_KEY.")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Groq rate limit reached. Please try again later.")
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="Answer generation timed out. Please try again.")
    except APIConnectionError:
        raise HTTPException(status_code=503, detail="Cannot connect to Groq. Please try again later.")
    except APIStatusError:
        raise HTTPException(status_code=502, detail="Groq could not generate an answer. Please try again later.")

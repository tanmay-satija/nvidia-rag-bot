"""
vectorstore.py — Local embeddings + FAISS index.

Why local embeddings instead of an API: embedding is called once per chunk
at index-build time and once per query at search time — far more frequent
than generation. Running it locally with sentence-transformers keeps the
pipeline free to operate and removes an external dependency (and network
round-trip) from the hot path. Generation, which needs a much larger model,
is the one place we call out to Groq.

Model: BAAI/bge-small-en-v1.5 — a strong, small (~130MB) embedding model
that runs fast on CPU, which matters since this needs to run without a GPU.
"""

import os
import json
import pickle
import numpy as np

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


class VectorStore:
    def __init__(self, model_name=None):
        # Imported lazily so the rest of the codebase can be used/tested
        # without requiring torch/sentence-transformers to be installed.
        from sentence_transformers import SentenceTransformer
        import faiss

        self.model = SentenceTransformer(model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
        self.faiss = faiss
        self.index = None
        self.chunks = []  # parallel list: chunks[i] <-> vector i in index

    def build(self, chunks):
        """Embed a list of chunk dicts (from chunking.py) and build the
        FAISS index. Stores the chunk metadata alongside for citation."""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=True,
            normalize_embeddings=True,  # so inner product == cosine similarity
        )
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = self.faiss.IndexFlatIP(dim)  # exact search, cosine sim
        self.index.add(embeddings)

    def search(self, query, k=5):
        """Return the top-k chunks most similar to the query, with scores."""
        q_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def save(self, path=INDEX_DIR):
        os.makedirs(path, exist_ok=True)
        self.faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path=INDEX_DIR):
        if not all(os.path.isfile(os.path.join(path, name)) for name in ("index.faiss", "chunks.pkl")):
            raise FileNotFoundError("Index not built yet")
        self.index = self.faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)

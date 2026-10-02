# NVIDIA Filings RAG Bot

A retrieval-augmented generation system that answers natural-language questions
about NVIDIA using its actual SEC filings (10-K / 10-Q) — using retrieved excerpts and numbered source citations.
Citation presence does not guarantee factual correctness; verify the evidence.

Built as a portfolio project focused on three things that matter in production
RAG systems and come up constantly in interviews: **chunking strategy**,
**hallucination prevention**, and **source citation**.

## How it works

```
SEC EDGAR API  →  chunking.py  →  vectorstore.py  →  generator.py
(free, no key)    (section-aware,   (local embeddings,   (GPT-OSS 120B
                   sliding window)   FAISS index)          via Groq, cited
                                                            answers only)
```

1. **Ingestion** (`src/ingest.py`) — pulls NVIDIA's real 10-K/10-Q filings
   directly from SEC EDGAR's free public JSON API (CIK `0001045810`), no
   scraping or API key required.
2. **Chunking** (`src/chunking.py`) — splits filings on their `Item N.`
   section headers first, so a chunk never straddles unrelated topics, then
   sliding-window chunks within each section (250 words, 50-word overlap) so
   facts near a boundary aren't severed.
3. **Embedding + retrieval** (`src/vectorstore.py`) — chunks are embedded
   locally with `sentence-transformers` (`BAAI/bge-small-en-v1.5`) and
   indexed in FAISS for fast cosine-similarity search. Local embeddings keep
   the hot path (every query) free and fast; only generation calls an
   external API.
4. **Generation** (`src/generator.py`) — retrieved chunks go to GPT-OSS 120B
   via Groq's free API with a system prompt that forces the model to answer
   only from the provided context, cite every claim to a chunk number, and
   explicitly say when it doesn't know rather than guess.
5. **Evaluation** (`eval/evaluate.py`) — a lightweight, rule-based framework
   that scores retrieval quality (did the right section get retrieved?) and
   groundedness (does every factual sentence carry a valid citation?).

## Why this design

- **Chunking**: section-aware splitting was chosen after noticing that naive
  fixed-size chunking regularly split "Risk Factors" language into the same
  chunk as unrelated "Legal Proceedings" text, hurting retrieval precision.
  Splitting on `Item N.` headers first fixes that; the sliding window with
  overlap inside each section keeps chunks from a mid-sentence numeric cutoff.
- **Hallucination prevention**: rather than trusting the model to "just not
  make things up," the system makes ungrounded claims *detectable* — the
  prompt requires a citation per claim, and the eval framework flags any
  sentence that looks factual but has no citation attached.
- **Citations**: every answer traces back to a specific filing, form type,
  and section — not just "trust me," but a pointer a user (or interviewer)
  can go verify.

## Setup

Use **Python 3.12**. The first embedding run downloads the model from Hugging Face.
You need internet access for ingestion and generation, plus a Groq API key.


```bash
git clone https://github.com/tanmay-satija/nvidia-rag-bot.git
cd nvidia-rag-bot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your Groq API key and SEC contact User-Agent
```

## Usage

```bash
# 1. One-time: pull filings from SEC EDGAR and build the FAISS index
python src/rag_pipeline.py build

# 2. Ask a question from the CLI
python src/rag_pipeline.py ask "What was NVIDIA's Data Center segment revenue?"

# 3. Or run the full web app
uvicorn backend.main:app --reload --port 8000   # terminal 1
streamlit run frontend/app.py                    # terminal 2
```

## Evaluation

```bash
python eval/evaluate.py                  # retrieval + live Groq generation
python eval/evaluate.py --retrieval-only # no generation calls; cached model/index required
```

Scores every question in `eval/eval_dataset.json` on:
- **Keyword hit rate** — did retrieval surface a chunk containing the
  expected terms?
- **Section hit rate** — did retrieval surface a chunk from the expected
  filing section?
- **Uncited claims per answer** — factual-looking sentences with no
  citation (a proxy for hallucination risk)
- **Invalid citations** — citation numbers that don't map to a real
  retrieved chunk (a proxy for citation fabrication)

## Project structure

```
nvidia-rag-bot/
├── src/
│   ├── ingest.py         # SEC EDGAR filing download
│   ├── chunking.py       # section-aware + sliding-window chunking
│   ├── vectorstore.py    # embeddings + FAISS index
│   ├── generator.py      # Groq generation with citation prompting
│   └── rag_pipeline.py   # CLI entry point tying it together
├── backend/main.py       # FastAPI service
├── frontend/app.py       # Streamlit UI
├── eval/
│   ├── eval_dataset.json # hand-written QA pairs for scoring
│   └── evaluate.py       # retrieval + groundedness scoring
├── requirements.txt
└── .env.example
```

## Known limitations

- The eval framework is rule-based (keyword/citation matching), not an LLM
  judge — fast and free, but it can't catch a factual error that's
  correctly cited to the wrong-but-plausible-looking chunk.
- FAISS's `IndexFlatIP` does exact search, which is fine at this corpus
  size (a handful of filings) but wouldn't scale to millions of chunks
  without switching to an approximate index (e.g. HNSW).
- SEC EDGAR filings are HTML with heavy table markup; `html_to_text`
  strips tags but doesn't reconstruct table structure, so numbers inside
  dense financial tables retrieve less reliably than prose.

## Validation

```bash
python -m unittest discover -s tests -v
```

The tests cover chunk boundaries, invalid overlap, section-name normalization,
citation bounds, API request validation, missing indexes, and source evidence.
GitHub Actions runs these tests without API keys or model downloads.

## Sources and errors

The UI displays each retrieved excerpt. Indexes built with this version also
include links to the original SEC filings. Rebuild an older index to add these
links: `python src/rag_pipeline.py build`.
The backend reports missing indexes, authentication failures, rate limits,
connection errors, and generation timeouts. Query length is limited to 4,000
characters and retrieval to 1–10 chunks.

The `/health` endpoint checks that the service responds; it does not verify
Groq credentials or retrieval quality. Run evaluation to assess the pipeline.
The eight-question dataset is a smoke evaluation, not an accuracy benchmark.
A retrieval-only run on the existing six-filing index (2026-10-01, k=5)
matched keywords on 7/8 questions (87.5%) and expected sections on 4/8 (50%).
The section metric counts the question without a section expectation as a pass.
These are loose smoke metrics, not answer accuracy; generation was not evaluated.
The generation checks assess
citation presence and index validity, not whether an excerpt supports a claim.

## Publishing and local data

`.gitignore` excludes `.env`, virtual environments, downloaded filings, and
built indexes. Keep `.env.example` as the configuration template. Never commit
API keys. Rebuild data locally using the documented build command.
Only load indexes you built yourself: the chunk metadata currently uses Python
pickle and is unsafe to load from untrusted sources.

## License

Project code is available under the [MIT License](LICENSE). Downloaded filings
and external model assets are not covered by this project's license.

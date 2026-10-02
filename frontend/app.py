"""
app.py — Streamlit UI for the NVIDIA Filings RAG system.

Run with:
    streamlit run frontend/app.py

Talks to the FastAPI backend over HTTP (make sure it's running first:
    uvicorn backend.main:app --reload --port 8000
).
"""

import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="NVIDIA Filings RAG", page_icon="📊", layout="centered")
st.title("📊 NVIDIA Filings Q&A")
st.caption(
    "Ask questions about NVIDIA's SEC filings (10-K / 10-Q). "
    "Answers use retrieved filing excerpts with citations. Verify claims against the sources."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This tool retrieves relevant excerpts from NVIDIA's SEC filings "
        "using local embeddings + FAISS, then generates a grounded answer "
        "using GPT-OSS 120B (via Groq)."
    )
    st.write("**Try asking:**")
    st.markdown(
        "- What was NVIDIA's Data Center segment revenue?\n"
        "- What risk factors does NVIDIA cite around export controls?\n"
        "- How did gross margin change year-over-year?"
    )
    k = st.slider("Chunks to retrieve (k)", min_value=2, max_value=10, value=5)

question = st.text_input("Your question", placeholder="e.g. What was NVIDIA's revenue growth last quarter?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving filings and generating answer..."):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/query",
                json={"question": question, "k": k},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            st.markdown("### Answer")
            st.write(data["answer"])

            st.markdown("### Sources")
            for s in data["sources"]:
                score_str = f" · similarity {s['score']:.3f}" if s.get("score") is not None else ""
                st.markdown(
                    f"**[{s['index']}]** {s['form']} — {s['report_date']} "
                    f"— *{s['section']}*{score_str}"
                )
                if s.get("url"):
                    st.link_button("Open SEC filing", s["url"])
                with st.expander(f"Excerpt [{s['index']}]"):
                    st.write(s.get("excerpt", "Excerpt unavailable."))
        except requests.exceptions.Timeout:
            st.error("The request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error(
                "Can't reach the backend. Make sure it's running:\n\n"
                "`uvicorn backend.main:app --reload --port 8000`"
            )
        except requests.exceptions.HTTPError as e:
            try:
                detail = e.response.json().get("detail", "Request failed")
            except ValueError:
                detail = f"Request failed (HTTP {e.response.status_code})"
            st.error(f"Backend error: {detail}")
        except requests.exceptions.RequestException:
            st.error("The request failed. Please try again.")

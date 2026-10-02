"""
ingest.py — Pulls NVIDIA's 10-K / 10-Q filings directly from SEC EDGAR.

SEC EDGAR exposes a free, no-auth-required JSON API. We use it in two steps:
  1. Hit the "submissions" endpoint for NVIDIA's CIK to list recent filings.
  2. Download the actual filing document (HTML) for each one we want.

NVIDIA's CIK (Central Index Key) is 0001045810 — a fixed identifier SEC
assigns to every filer.
"""

import os
import time
import json
import requests
from bs4 import BeautifulSoup

NVIDIA_CIK = "0001045810"
SEC_HEADERS = {
    # SEC requires a descriptive User-Agent identifying the requester
    "User-Agent": os.getenv("SEC_USER_AGENT", "NVIDIA RAG research research@example.com")
}
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def get_filing_list(forms=("10-K", "10-Q"), limit=8):
    """Fetch metadata for NVIDIA's recent filings of the given form types."""
    url = f"https://data.sec.gov/submissions/CIK{NVIDIA_CIK}.json"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    filings = []
    for i in range(len(recent["form"])):
        if recent["form"][i] in forms:
            filings.append({
                "form": recent["form"][i],
                "accession_number": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "primary_document": recent["primaryDocument"][i],
                "report_date": recent["reportDate"][i],
            })
        if len(filings) >= limit:
            break
    return filings


def download_filing(filing, out_dir=RAW_DIR):
    """Download and save the raw HTML for a single filing."""
    os.makedirs(out_dir, exist_ok=True)
    acc_no_nodash = filing["accession_number"].replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(NVIDIA_CIK)}/"
        f"{acc_no_nodash}/{filing['primary_document']}"
    )
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()

    fname = f"{filing['form']}_{filing['report_date']}.html"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(resp.text)

    time.sleep(0.2)  # be polite to SEC's rate limits (max 10 req/sec)
    return fpath


def html_to_text(fpath):
    """Strip an SEC filing HTML file down to clean plain text."""
    with open(fpath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines left over from table/HTML structure
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def run(forms=("10-K", "10-Q"), limit=8):
    """Full ingestion pass: list filings, download, convert to text."""
    filings = get_filing_list(forms=forms, limit=limit)
    print(f"Found {len(filings)} filings: "
          f"{[f['form'] + ' ' + f['report_date'] for f in filings]}")

    docs = []
    for filing in filings:
        print(f"Downloading {filing['form']} ({filing['report_date']})...")
        fpath = download_filing(filing)
        text = html_to_text(fpath)
        txt_path = fpath.replace(".html", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        docs.append({
            "form": filing["form"],
            "report_date": filing["report_date"],
            "filing_date": filing["filing_date"],
            "source_file": txt_path,
            "source_url": f"https://www.sec.gov/Archives/edgar/data/{int(NVIDIA_CIK)}/{filing['accession_number'].replace('-', '')}/{filing['primary_document']}",
            "char_count": len(text),
        })

    manifest_path = os.path.join(RAW_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(docs, f, indent=2)
    print(f"Done. {len(docs)} filings saved. Manifest: {manifest_path}")
    return docs


if __name__ == "__main__":
    run()

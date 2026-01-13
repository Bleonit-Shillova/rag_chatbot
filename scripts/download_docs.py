import csv
import os
import re
import requests
from urllib.parse import urlparse
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
CSV_PATH = os.path.join(BASE_DIR, "docs_urls.csv")

def safe_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-_. ]+", "", s)
    s = s.replace(" ", "_")
    return s[:180]

def guess_ext(url: str, content_type: str) -> str:
    if "pdf" in (content_type or "").lower() or url.lower().endswith(".pdf"):
        return ".pdf"
    return ".html"

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in tqdm(rows, desc="Downloading"):
        industry = safe_filename(r["industry"])
        title = safe_filename(r["title"])
        url = r["url"].strip()

        out_dir = os.path.join(RAW_DIR, industry)
        os.makedirs(out_dir, exist_ok=True)

        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            ct = resp.headers.get("Content-Type", "")
            ext = guess_ext(url, ct)

            fname = f"{title}{ext}"
            out_path = os.path.join(out_dir, fname)

            with open(out_path, "wb") as w:
                w.write(resp.content)

        except Exception as e:
            print(f"\nFAILED: {url}\n  -> {e}\n")

if __name__ == "__main__":
    main()

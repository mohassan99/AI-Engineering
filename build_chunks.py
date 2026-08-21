import requests
import json
import html
import re
import os
import time

BASE_URL = "https://api.coverage.cms.gov"
ID_LIST_PATH = "data/raw/lcd_id_list.json"
OUT_PATH = "data/processed/chunks.jsonl"

# Narrative fields worth chunking, per A2-Handoff.md
NARRATIVE_FIELDS = [
    "indication",
    "diagnoses_support",
    "diagnoses_dont_support",
    "coding_guidelines",
    "doc_reqs",
    "bibliography",
    "summary_of_evidence",
    "analysis_of_evidence",
]

CHUNK_SIZE_WORDS = 650   # approximation of ~750-1000 tokens
CHUNK_OVERLAP_WORDS = 90  # ~12-13% overlap


def get_license_token():
    url = f"{BASE_URL}/v1/metadata/license-agreement"
    resp = requests.get(url)
    body = resp.json()
    resp.raise_for_status()
    token = None
    for item in body.get("data", []):
        if isinstance(item, dict) and "Token" in item:
            token = item["Token"]
            break
    return token


def clean_lcd_text(raw):
    if not raw:
        return ""
    unescaped = html.unescape(raw)
    text_only = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", text_only).strip()


def fetch_lcd(token, document_id, document_version):
    url = f"{BASE_URL}/v1/data/lcd/"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"lcdid": document_id, "ver": document_version}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return data[0] if data else None


def build_doc_text(lcd_record):
    """Concatenate cleaned narrative fields into one labeled text blob per LCD."""
    parts = []
    for field in NARRATIVE_FIELDS:
        raw = lcd_record.get(field)
        cleaned = clean_lcd_text(raw)
        if cleaned:
            label = field.replace("_", " ").title()
            parts.append(f"[{label}]\n{cleaned}")
    return "\n\n".join(parts)


def chunk_text(text, size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


if __name__ == "__main__":
    with open(ID_LIST_PATH, "r", encoding="utf-8") as f:
        lcd_list = json.load(f)

    print(f"Loaded {len(lcd_list)} LCD entries from {ID_LIST_PATH}")

    token = get_license_token()
    if token is None:
        print("Could not get token. Aborting.")
        exit(1)

    os.makedirs("data/processed", exist_ok=True)

    total_chunks = 0
    failed = []

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for i, entry in enumerate(lcd_list, start=1):
            display_id = entry["document_display_id"]
            doc_id = entry["document_id"]
            doc_version = entry["document_version"]

            try:
                lcd_record = fetch_lcd(token, doc_id, doc_version)
            except Exception as e:
                print(f"[{i}/{len(lcd_list)}] {display_id}: FAILED to fetch ({e})")
                failed.append(display_id)
                continue

            if lcd_record is None:
                print(f"[{i}/{len(lcd_list)}] {display_id}: no data returned")
                failed.append(display_id)
                continue

            doc_text = build_doc_text(lcd_record)
            if not doc_text:
                print(f"[{i}/{len(lcd_list)}] {display_id}: no narrative text found")
                continue

            chunks = chunk_text(doc_text)
            for j, chunk in enumerate(chunks):
                record = {
                    "id": f"{display_id}_chunk{j}",
                    "source": display_id,
                    "text": chunk,
                }
                out_f.write(json.dumps(record) + "\n")
            total_chunks += len(chunks)

            print(f"[{i}/{len(lcd_list)}] {display_id}: {len(chunks)} chunks")
            time.sleep(0.05)  # light courtesy delay, well under the rate limit

    print(f"\nDone. Total chunks written: {total_chunks} -> {OUT_PATH}")
    if failed:
        print(f"Failed to fetch {len(failed)} documents: {failed}")

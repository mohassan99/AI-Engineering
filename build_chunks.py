"""
Stage A.1 REDO — token-aware, sentence-respecting, field-aware chunking.

Replaces the original word-count chunking (650 words/chunk, ~90-word
overlap, chunked from one concatenated blob per LCD) with chunk
boundaries sized against the REAL embedding model's actual token limit,
never splitting inside a sentence, and chunked PER NARRATIVE FIELD rather
than per flattened document.

BACKGROUND — why this redo exists (full detail in the A.1-redo handoff doc):
  Stage A.4's precision@5 eval scored 0.560 and showed a near-perfect
  correlation between chunks-per-LCD and retrieval success (1 chunk=0%,
  2=29%, 3=67%, 4+=100%). Diagnosis, verified against the real
  all-MiniLM-L6-v2 tokenizer (not assumed):
    - The model's real max_seq_length is 256 tokens. 95% of the original
      650-word chunks (211/222) exceeded that and were silently truncated
      by sentence-transformers before ever being embedded.
    - Every LCD's chunk0 opens with a near-universal "For any item to be
      covered by Medicare, it must..." preamble averaging ~305 tokens --
      OVER the entire budget by itself, for EVERY LCD in the corpus.
      Pulling the actual build_chunks.py from the project's GitHub repo
      revealed WHY: this is simply how CMS's own "indication" field
      conventionally opens.
    - A trailing DMEPOS boilerplate paragraph (SWO/WOPD/POD/coding
      guidelines) is appended once per document before chunking, and can
      be 40-56% of a thin LCD's only embedding. Same discovery: this is
      how the "doc_reqs" (Documentation Requirements) field conventionally
      opens.
    - Long HCPCS code enumerations are token-expensive (~4-5 tokens per
      5-character code -- a single 61-code list measured at 311 tokens,
      OVER budget alone) AND weak at differentiating LCDs: empirically
      measured cosine similarity between two DIFFERENT LCDs' code lists
      was 0.558, HIGHER than the 0.094 similarity between their actual
      coverage-criteria prose. Code lists are expensive AND unhelpful.
    - Chunk boundaries have been observed cutting a code table mid-list
      (e.g. L33318 Knee Orthoses, chunk1->chunk2).

WHY PER-FIELD CHUNKING (changed after finding the real source): the CMS
API returns each LCD's narrative as separate, labeled fields (indication,
diagnoses_support, doc_reqs, bibliography, etc.) -- NOT one blob. The
original build_chunks.py concatenates them into one blob before chunking;
this version chunks each field independently, so a chunk boundary can
never straddle two unrelated fields, and thin fields (like a non-"N/A"
bibliography/summary/analysis section) get their own chunk instead of
being buried in the tail of a much longer one. See chunk_lcd_record()
for the full reasoning.

THIS SCRIPT fixes chunk SIZING/BOUNDARIES. It does NOT remove the
leading/trailing boilerplate text itself or truncate code lists -- those
remain separate, complementary, embedding-input-only fixes that live in
ingest_chunks.py (so the full original text is still what's stored/shown/
sent to the LLM). Even with per-field chunking, a short LCD's lone
"indication" chunk can still open with 300+ tokens of universal preamble
before reaching that LCD's first real sentence -- ingest_chunks.py's
strip_boilerplate_safe() is what handles that.

FETCH/PARSE LOGIC: get_license_token(), clean_lcd_text(), and fetch_lcd()
below are copied verbatim from the project's actual, verified
build_chunks.py (pulled via GitHub, commit as of this session) --
confirmed working, 0 failures across 58 LCDs. Not reconstructed from
assumption.

NEW DEPENDENCY: nltk (for sentence-boundary detection -- handles the
heavy abbreviation use in this legal/medical text, e.g. "Pub. 100-08",
"e.g.", "U.S. Pharmacopeia", which a naive period-based split would
break on).
    pip install nltk
    python -c "import nltk; nltk.download('punkt_tab')"
"""

import requests
import json
import html
import re
import os
import time
import nltk
from sentence_transformers import SentenceTransformer

BASE_URL = "https://api.coverage.cms.gov"
LCD_ID_LIST_PATH = "data/raw/lcd_id_list.json"
CHUNKS_OUTPUT_PATH = "data/processed/chunks.jsonl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Narrative fields worth chunking. Originally pulled from the repo's
# existing build_chunks.py, but CORRECTED after testing against the LIVE
# API this session: "doc_reqs" and "coding_guidelines" came back as EMPTY
# STRINGS for every real record tested (L33738, L33370, L33786, L33611) --
# that content has evidently moved to "associated_info" (confirmed by
# inspecting the raw response: it opens with "<p><strong>DOCUMENTATION
# REQUIREMENTS</strong>..."). This differs from what the ORIGINAL
# build_chunks.py's field list assumed, and from what the existing
# chunks.jsonl corpus was built from -- CMS's API has evidently changed
# since that corpus was built (see the A.1-redo handoff's "field mapping
# drift" section for full detail, including a wording difference: the
# live associated_info boilerplate opens with "Section 1833(e) of the
# Social Security Act precludes payment..." -- NOT the "A Standard
# Written Order (SWO) must be communicated..." phrasing the existing
# corpus and ingest_chunks.py's anchors were built against).
# Old field names kept in the list (harmless if empty) in case some LCDs
# still populate them; associated_info added as the now-correct source.
NARRATIVE_FIELDS = [
    "indication",
    "diagnoses_support",
    "diagnoses_dont_support",
    "coding_guidelines",
    "doc_reqs",
    "associated_info",
    "bibliography",
    "summary_of_evidence",
    "analysis_of_evidence",
]

# Target chunk size in tokens, measured with the REAL embedding-model
# tokenizer (not a word-count proxy). all-MiniLM-L6-v2's max_seq_length
# is 256; 190 leaves margin for [CLS]/[SEP] special tokens and the
# vocabulary variance seen in code-dense passages.
TARGET_TOKENS = 190
OVERLAP_SENTENCES = 1


def chunk_lcd_record(lcd_record, source_id, tokenizer,
                      target_tokens=TARGET_TOKENS, overlap_sentences=OVERLAP_SENTENCES):
    """
    Chunk one LCD by NARRATIVE FIELD, not by first concatenating every
    field into one blob and then chunking that (the original approach).

    WHY THIS CHANGED FROM THE EARLIER DRAFT OF THIS SCRIPT: the earlier
    draft chunked build_doc_text()'s already-concatenated output and
    relied on ingest_chunks.py detecting boilerplate spans after the fact
    via text anchors (e.g. "For any item to be covered by Medicare, it
    must..."). Pulling the actual, verified build_chunks.py from the
    project's GitHub repo showed the real source of that boilerplate:
    it's simply how CMS's own "indication" field conventionally OPENS,
    and how the "doc_reqs" field conventionally opens with its own
    "GENERAL [SWO/WOPD/POD language]" preamble. Since NARRATIVE_FIELDS
    are separate, labeled fields in the raw API response -- before any
    concatenation happens -- chunking per field is both simpler and more
    precise than detecting the same boundaries after they've been fused
    into one blob:
      - Each field becomes its own independent unit for the sentence-aware
        token-budget chunker below, so a chunk boundary can never
        straddle two unrelated fields (e.g. part indication, part
        bibliography).
      - The ingest_chunks.py boilerplate-span-removal logic (leading
        preamble / trailing SWO-POD span) still applies for embedding
        input specifically, since even a clean "indication"-only chunk
        for a short LCD can still open with 300+ tokens of the universal
        preamble before reaching that LCD's actual first sentence.
      - The "bibliography"/"summary_of_evidence"/"analysis_of_evidence"
        fields, which are often "N/A" but occasionally carry real,
        distinctive content (confirmed for L33611's reconsideration-
        history discussion), now naturally become their own chunk(s)
        instead of being fused onto the end of a much longer chunk --
        so genuinely thin LCDs get one more chance at a chunk with real,
        undiluted content, rather than that content being buried at the
        tail of whichever chunk happened to run up against it.

    Returns (chunks, oversized_log) -- same shape as chunk_document() in
    the earlier draft.
    """
    all_chunks = []
    all_oversized = []
    field_idx = 0

    for field in NARRATIVE_FIELDS:
        raw = lcd_record.get(field)
        cleaned = clean_lcd_text(raw)
        if not cleaned or cleaned.upper() in ("N/A", "NA", "NONE"):
            continue

        label = field.replace("_", " ").title()
        labeled_text = f"[{label}] {cleaned}"

        field_chunks, oversized = _chunk_field_text(
            labeled_text, source_id, field_idx, tokenizer,
            target_tokens, overlap_sentences,
        )
        all_chunks.extend(field_chunks)
        all_oversized.extend(oversized)
        field_idx += len(field_chunks)

    return all_chunks, all_oversized


def _chunk_field_text(text, source_id, start_idx, tokenizer,
                       target_tokens, overlap_sentences):
    """
    Split one field's labeled text into chunks by accumulating whole
    sentences up to `target_tokens` (measured with the real tokenizer),
    never splitting a chunk boundary inside a sentence.

    BUG FIXED DURING DEVELOPMENT (documented here so it isn't
    reintroduced): the overlap step must exclude oversized sentences from
    being carried into the next chunk. An earlier version carried
    whatever sentence(s) were last in the flushed chunk regardless of
    size -- if that chunk was a single 700+-token sentence, the *next*
    chunk would inherit that same oversized sentence as its overlap seed,
    producing two oversized chunks in a row instead of one. Verified
    against real corpus text (L33794's insulin-pump inotropic-therapy
    criteria, a single 720-token sentence) before and after the fix.
    """
    def sentence_length(s):
        return len(tokenizer.encode(s, add_special_tokens=False))

    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    oversized_log = []
    current = []
    current_tokens = 0
    idx = start_idx

    def flush():
        nonlocal current, current_tokens, idx
        if not current:
            return [], 0
        chunk_text = " ".join(current)
        n_tokens = sentence_length(chunk_text)
        chunk_id = f"{source_id}_chunk{idx}"
        chunks.append({"id": chunk_id, "source": source_id, "text": chunk_text})
        if n_tokens > target_tokens:
            oversized_log.append((chunk_id, n_tokens))
        idx += 1
        candidate = current[-overlap_sentences:] if overlap_sentences else []
        candidate = [s for s in candidate if sentence_length(s) <= target_tokens]
        return candidate, sum(sentence_length(s) for s in candidate)

    for sentence in sentences:
        n = sentence_length(sentence)
        if current_tokens + n > target_tokens and current:
            current, current_tokens = flush()
        current.append(sentence)
        current_tokens += n

    flush()
    return chunks, oversized_log


def get_license_token():
    """Verified, unchanged from the existing build_chunks.py."""
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
    """Verified, unchanged from the existing build_chunks.py."""
    if not raw:
        return ""
    unescaped = html.unescape(raw)
    text_only = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", text_only).strip()


def fetch_lcd(token, document_id, document_version):
    """Verified, unchanged from the existing build_chunks.py."""
    url = f"{BASE_URL}/v1/data/lcd/"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"lcdid": document_id, "ver": document_version}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return data[0] if data else None


if __name__ == "__main__":
    print("Downloading nltk sentence tokenizer data if not already present...")
    nltk.download("punkt_tab", quiet=True)

    print(f"Loading tokenizer for '{EMBEDDING_MODEL}'...")
    # Uses the SentenceTransformer's underlying tokenizer so token counts
    # here exactly match what ingest_chunks.py's model.encode() will see.
    st_model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer = st_model.tokenizer

    with open(LCD_ID_LIST_PATH, "r", encoding="utf-8") as f:
        lcd_list = json.load(f)
    print(f"Loaded {len(lcd_list)} LCD entries from {LCD_ID_LIST_PATH}")

    token = get_license_token()
    if token is None:
        print("Could not get token. Aborting.")
        raise SystemExit(1)

    os.makedirs("data/processed", exist_ok=True)

    all_chunks = []
    all_oversized = []
    failed = []

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

        doc_chunks, oversized = chunk_lcd_record(lcd_record, display_id, tokenizer)
        if not doc_chunks:
            print(f"[{i}/{len(lcd_list)}] {display_id}: no narrative text found")
            continue

        all_chunks.extend(doc_chunks)
        all_oversized.extend(oversized)
        print(f"[{i}/{len(lcd_list)}] {display_id}: {len(doc_chunks)} chunks")
        time.sleep(0.05)  # light courtesy delay, well under the rate limit

    print(f"\nProduced {len(all_chunks)} chunks from {len(lcd_list)} LCDs.")
    if failed:
        print(f"Failed to fetch {len(failed)} documents: {failed}")

    print(f"\nOversized (unavoidable single-sentence) chunks: {len(all_oversized)}")
    if all_oversized:
        print("These consist of one sentence too long to split without breaking")
        print("mid-sentence. Consider whether the code-run truncation in")
        print("ingest_chunks.py's strip_boilerplate_safe() would reduce these --")
        print("many long single sentences in this corpus are long specifically")
        print("because they contain an embedded HCPCS code enumeration.")
        for cid, n in sorted(all_oversized, key=lambda x: -x[1])[:10]:
            print(f"  {cid}: {n} tokens")

    with open(CHUNKS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    print(f"\nWrote {len(all_chunks)} chunks to {CHUNKS_OUTPUT_PATH}")

import json
import re
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/chunks.jsonl"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "dme_lcds"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# A third embedding-input problem, distinct from both boilerplate spans:
# long HCPCS code enumerations (e.g. a single 61-code list for Power
# Mobility Devices) are both token-EXPENSIVE and semantically WEAK.
# Verified, not assumed:
#   - Tokenization cost: a 5-character HCPCS code costs ~4 subword
#     tokens standing alone (e.g. "E0470" -> ['e','##0','##47','##0']),
#     rising to ~4.8-5.1/code once list-separator commas are counted. A
#     real 61-code enumeration from this corpus measured at 311 tokens --
#     121% of the entire 256-token budget, by itself.
#   - Differentiation power: embedding two DIFFERENT LCDs' full code
#     lists (Power Mobility Devices vs Tracheostomy Care Supplies) with
#     the real all-MiniLM-L6-v2 model gave a cosine similarity of 0.558.
#     The same two LCDs' actual coverage-criteria PROSE gave 0.094 --
#     the prose differentiates these LCDs roughly 6x better than the
#     code lists do. Even more strikingly, each LCD's own code list vs
#     its own prose scored near ZERO (-0.001 and -0.076) -- the model
#     doesn't even treat a code list as "about" its own LCD's topic.
# Net effect: long code enumerations spend a disproportionate share of an
# already-scarce token budget on the least-differentiating content in
# the chunk. This truncates them for EMBEDDING INPUT ONLY -- exactly the
# same pattern as the boilerplate stripping above: the full code list
# stays in the stored/displayed/generation text untouched.
HCPCS_CODE = r"[A-Z][0-9]{4}(?:-[A-Z][0-9]{4})?"
CODE_RUN_PATTERN = re.compile(rf"(?:{HCPCS_CODE})(?:,\s*(?:{HCPCS_CODE})){{5,}}")
CODE_RUN_KEEP = 3  # codes to keep visible before truncating with "..."


def truncate_code_runs(text, keep=CODE_RUN_KEEP):
    """
    Collapse any run of 6+ consecutive comma-separated HCPCS-style codes
    (plain 5-character codes or hyphenated ranges like "A6209-A6215")
    down to the first `keep` codes plus a count marker, e.g.:
        "K0800, K0801, K0802, ... (61 codes total)"
    Short mentions (5 or fewer codes) are left untouched -- those don't
    meaningfully dent the token budget and may still carry some
    differentiating signal (e.g. "covered for E0470 and E0601 devices").
    """
    def replace(match):
        codes = [c.strip() for c in match.group().split(",")]
        if len(codes) <= keep:
            return match.group()
        return f"{', '.join(codes[:keep])}, ... ({len(codes)} codes total)"

    return CODE_RUN_PATTERN.sub(replace, text)


# Stage A.4 fix: every LCD's raw text ends with a near-identical DMEPOS
# boilerplate trailer (Standard Written Order / Proof of Delivery / coding
# guideline language), appended ONCE per document before chunking. In a
# multi-chunk LCD that trailer lands in only one chunk among several and
# gets diluted. In a 1-3 chunk LCD it can be 40-56% of the LCD's only/few
# embeddings, pulling the vector toward generic "DME policy" language and
# away from what's actually distinctive about that LCD. Precision@5 eval
# (A.4) showed this correlating almost perfectly with retrieval failures:
# 1 chunk=0% pass, 2 chunks=29%, 3 chunks=67%, 4+ chunks=100%.
#
# A second, more consequential boilerplate span sits at the START of
# every LCD's chunk0: a near-universal "[Indication] For any item to be
# covered by Medicare, it must..." preamble restating the generic LCD
# "reasonable and necessary" standard. Measured against the real
# all-MiniLM-L6-v2 tokenizer, this preamble alone averages ~305 tokens --
# already OVER the model's 256-token max_seq_length. Since
# sentence-transformers truncates silently at 256 tokens, this means the
# embedding for every single-chunk LCD was based ENTIRELY on this generic
# preamble, with ZERO of the LCD's actual distinctive content ever
# reaching the model. This explains the chunk-count/precision correlation
# far more than the trailing SWO/POD boilerplate does: multi-chunk LCDs
# have later chunks (chunk1+) that don't carry this preamble and so embed
# real content; single/few-chunk LCDs never get past it.
LEADING_START_ANCHOR = "For any item to be covered by Medicare, it must"
LEADING_END_ANCHOR = (
    "are defined by the following coverage indications, limitations "
    "and/or medical necessity."
)

# START_ANCHORS/END_ANCHOR (below) mark the trailing DMEPOS boilerplate
# (Standard Written Order / Proof of Delivery / coding guideline
# language), appended ONCE per document before chunking.
START_ANCHORS = [
    "A Standard Written Order (SWO) must be communicated",
    "must have received a signed Standard Written Order (SWO)",
    "must have received a signed SWO",
    "An item/service is correctly coded when it meets all the coding guidelines",
    "Proof of delivery (POD) is a Supplier Standard",
]
END_ANCHOR = (
    "All services that do not have appropriate proof of delivery from the "
    "supplier shall be denied as not reasonable and necessary."
)

# If stripping would leave less than this many characters, the chunk is
# almost entirely boilerplate to begin with (chunk boundary landed inside
# or right before the trailer, with nothing distinctive on either side).
# Embedding a near-empty string would be a worse, degenerate problem than
# the dilution we're fixing, so fall back to the original (still-diluted)
# text rather than embedding near-nothing.
MIN_STRIPPED_LENGTH = 200


def strip_boilerplate(text):
    """
    Return `text` with the repeated leading and trailing DMEPOS
    boilerplate spans removed, for use as EMBEDDING INPUT ONLY.

    This does not change what's stored/displayed/sent to the LLM at
    generation time -- ingest() below embeds the stripped text but
    upserts the original, full text as the Chroma document. Only the
    vector changes; retrieve()/generate()/answer() in rag_pipeline.py
    are unaffected and need no changes.

    Two spans are removed, each via a start-anchor-to-end-anchor SPAN
    removal (not truncate-to-end-of-string), so real LCD-specific text
    that sometimes sits before/after a boilerplate block is preserved:

    1. LEADING: the "For any item to be covered by Medicare, it
       must..." preamble that opens every LCD's chunk0. This span
       averages ~305 tokens by itself -- already over the embedding
       model's 256-token max_seq_length -- so for any LCD thin enough
       that chunk0 is its only (or first) chunk, this preamble was
       previously consuming the model's *entire* truncated embedding
       window before the LCD's actual distinctive content was ever
       reached.
    2. TRAILING: the SWO/WOPD/POD/coding-guideline trailer appended once
       per document before chunking (see START_ANCHORS/END_ANCHOR above).

    If an END anchor isn't found after its START anchor, the chunk
    simply ends mid-boilerplate at a chunk-split boundary, so truncating
    to end-of-string for that span is safe -- there's nothing real left
    in the chunk to lose.
    """
    # --- leading span ---
    if LEADING_START_ANCHOR in text:
        lead_start = text.find(LEADING_START_ANCHOR)
        lead_end_idx = text.find(LEADING_END_ANCHOR, lead_start)
        if lead_end_idx != -1:
            lead_end = lead_end_idx + len(LEADING_END_ANCHOR)
            text = (text[:lead_start] + " " + text[lead_end:]).strip()
        else:
            # preamble start found but chunk ends before the end anchor --
            # nothing real follows in this chunk, safe to drop to end
            text = text[:lead_start].rstrip()

    # --- trailing span ---
    start_points = [text.find(a) for a in START_ANCHORS if a in text]
    if start_points:
        start = min(start_points)
        end_idx = text.find(END_ANCHOR, start)
        if end_idx != -1:
            end = end_idx + len(END_ANCHOR)
            text = (text[:start] + " " + text[end:]).strip()
        else:
            text = text[:start].rstrip()

    return text


def strip_boilerplate_safe(original_text):
    """
    Wrapper applying both embedding-input-only fixes in sequence, plus the
    MIN_STRIPPED_LENGTH safety fallback:
      1. strip_boilerplate() removes the leading Medicare-purpose preamble
         and trailing SWO/POD span.
      2. truncate_code_runs() collapses long HCPCS code enumerations.
    If the combined result would leave less text than MIN_STRIPPED_LENGTH,
    the chunk was almost entirely boilerplate/codes to begin with --
    embedding a near-empty string would be a worse, more degenerate
    problem than the dilution we're fixing, so fall back to the original
    (still-diluted) text rather than embedding near-nothing.
    """
    stripped = strip_boilerplate(original_text)
    stripped = truncate_code_runs(stripped)
    if len(stripped) < MIN_STRIPPED_LENGTH:
        return original_text
    return stripped


def read_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


if __name__ == "__main__":
    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = read_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads it, ~80MB)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Connecting to Chroma at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # get_or_create so re-runs don't fail if the collection already exists
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [c["id"] for c in chunks]
    full_texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]

    # Embedding input has the boilerplate trailer stripped (Stage A.4 fix);
    # the stored/upserted document text stays the original, full chunk --
    # retrieve() returns full_texts unchanged, generate() still sees the
    # complete LCD excerpt including the SWO/POD language if relevant.
    embed_texts = [strip_boilerplate_safe(t) for t in full_texts]
    stripped_count = sum(1 for a, b in zip(full_texts, embed_texts) if a != b)
    print(f"Boilerplate stripped from embedding input for {stripped_count}/{len(chunks)} chunks.")

    print("Embedding all chunks (this is the slow step, batched)...")
    embeddings = model.encode(embed_texts, show_progress_bar=True, batch_size=32).tolist()

    print("Upserting into Chroma...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=full_texts,
        metadatas=metadatas,
    )

    count = collection.count()
    print(f"\nDone. Collection '{COLLECTION_NAME}' item count: {count}")
    print(f"Expected: {len(chunks)}")
    if count != len(chunks):
        print("WARNING: collection count does not match chunk count — investigate before proceeding.")

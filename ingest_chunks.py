import json
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/chunks.jsonl"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "dme_lcds"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

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
# START_ANCHORS mark the beginning of that trailer. They were chosen and
# checked against all 222 chunks to confirm none of them appear in
# genuine LCD-specific content -- e.g. we deliberately do NOT match on
# the bare word "GENERAL", since several LCDs use "GENERAL" or "GENERAL
# COVERAGE CRITERIA" as a real section header introducing substantive,
# LCD-specific coverage criteria (not boilerplate).
START_ANCHORS = [
    "A Standard Written Order (SWO) must be communicated",
    "must have received a signed Standard Written Order (SWO)",
    "must have received a signed SWO",
    "An item/service is correctly coded when it meets all the coding guidelines",
    "Proof of delivery (POD) is a Supplier Standard",
]

# END_ANCHOR marks where the trailer paragraph reliably ends, right before
# the [Bibliography]/[Summary Of Evidence]/[Analysis Of Evidence] tags.
# A small number of chunks (found during a manual check, e.g. L33611's
# reconsideration-history chunk) have real, LCD-specific text inside those
# tags -- not just "N/A" filler. Using a start+end SPAN removal instead of
# a truncate-to-end-of-string means that real trailing content is kept.
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
    Return `text` with the repeated DMEPOS boilerplate trailer removed,
    for use as EMBEDDING INPUT ONLY.

    This does not change what's stored/displayed/sent to the LLM at
    generation time -- ingest() below embeds the stripped text but
    upserts the original, full text as the Chroma document. Only the
    vector changes; retrieve()/generate()/answer() in rag_pipeline.py
    are unaffected and need no changes.

    Removes the SPAN from the earliest START_ANCHORS match through
    END_ANCHOR (if present), keeping whatever real content comes before
    and after -- rather than truncating everything from the start anchor
    to the end of the string, which would also discard genuine
    LCD-specific text that occasionally follows the trailer (e.g. a
    non-"N/A" Summary/Analysis Of Evidence section). If END_ANCHOR isn't
    found, the chunk simply ends mid-boilerplate at a chunk-split
    boundary, so truncating to end-of-string is safe -- there's nothing
    real left in that chunk to lose.
    """
    start_points = [text.find(a) for a in START_ANCHORS if a in text]
    if not start_points:
        return text
    start = min(start_points)

    end_idx = text.find(END_ANCHOR, start)
    if end_idx != -1:
        end = end_idx + len(END_ANCHOR)
        stripped = (text[:start] + " " + text[end:]).strip()
    else:
        stripped = text[:start].rstrip()

    if len(stripped) < MIN_STRIPPED_LENGTH:
        return text
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
    embed_texts = [strip_boilerplate(t) for t in full_texts]
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

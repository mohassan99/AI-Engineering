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

# Threshold for "is there any real content left after stripping known
# boilerplate anchors" -- used to decide whether a chunk is boilerplate-
# DOMINANT (drop it) vs. just short-but-real (keep it).
#
# NOT an arbitrary number: checked empirically against the real, live
# corpus (all chunks where strip_boilerplate()+truncate_code_runs()
# changed the text at all). Results cluster cleanly in two groups with a
# real gap between them -- nothing landed in between:
#   - genuinely nothing left: 0 chars (fully consumed), 7 chars
#     ("GENERAL"), 12 chars ("[Indication]") -- leftover field labels
#     or section headers with no sentence content at all.
#   - genuinely real content: 77+ chars, all complete sentences (e.g.
#     "DRESSINGS The following are specific guidelines for individual
#     product types.").
# 30 sits inside that gap. This replaces an earlier version of this
# threshold (200 chars) that was calibrated for a different question --
# "is the embedding input near-empty" -- and, when reused to also decide
# whether to DROP a chunk from the corpus, was catching and discarding
# short-but-complete, on-topic sentences (verified case: L33738_chunk2's
# post-strip content, "A facial prosthesis is covered when there is loss
# or absence of facial tissue due to disease, trauma, surgery, or a
# congenital defect. GENERAL", is ~140 chars -- entirely real, the exact
# sentence an eval question asks about -- and was being dropped under
# the 200-char version of this check before this fix).
MIN_REAL_CONTENT = 30


def _remove_boundary_span(text, start_anchors, end_anchor):
    """
    Remove one repeated boilerplate span from `text`, for use as
    EMBEDDING INPUT ONLY. Handles THREE cases depending on which
    anchor(s) actually fall inside THIS chunk -- necessary because, under
    the new ~190-token per-field chunking, a ~305-token boilerplate span
    routinely gets split across two adjacent chunks (verified: L33738's
    chunk0 contains the leading start anchor but not its end anchor; the
    next chunk contains the end anchor but not the start).

    1. BOTH anchors present (start, then end) -- classic case, the whole
       span sits inside one chunk: cut from start anchor through end of
       end anchor. Real text before/after the span is preserved.
    2. ONLY a start anchor present -- the span continues past this
       chunk's boundary into the next chunk. Nothing after the start
       anchor in THIS chunk is real content (it's mid-boilerplate), so
       truncate to end-of-string at the start anchor. This is safe
       specifically because there's nothing to lose here -- distinct
       from case 3, which does have something to preserve.
    3. ONLY the end anchor present -- this chunk is a CONTINUATION of a
       span that started in the PREVIOUS chunk. Everything from the
       start of THIS chunk through the end anchor is still boilerplate,
       so DELETE that prefix segment (not truncate — the whole point is
       that real content follows it) and keep what remains. Before this
       fix, this case wasn't handled at all: the leading branch only
       fired `if start_anchor in text`, so a continuation chunk's
       boilerplate tail was left sitting in the embedding input,
       diluting whatever real content followed it in the same chunk.
    """
    start_points = [text.find(a) for a in start_anchors if a in text]
    start_idx = min(start_points) if start_points else -1
    end_idx = text.find(end_anchor, start_idx if start_idx != -1 else 0)

    if start_idx != -1 and end_idx != -1:
        # case 1: span removal, real text on either side preserved
        end = end_idx + len(end_anchor)
        return (text[:start_idx] + " " + text[end:]).strip()
    if start_idx != -1 and end_idx == -1:
        # case 2: nothing real left after the start anchor in this chunk
        return text[:start_idx].rstrip()
    if start_idx == -1 and end_idx != -1:
        # case 3: continuation chunk -- delete the boilerplate prefix,
        # keep the real content that follows the end anchor
        end = end_idx + len(end_anchor)
        return text[end:].lstrip()
    return text


def strip_boilerplate(text):
    """
    Return `text` with the repeated leading and trailing DMEPOS
    boilerplate spans removed, for use as EMBEDDING INPUT ONLY.

    This does not change what's stored/displayed/sent to the LLM at
    generation time -- ingest() below embeds the stripped text but
    upserts the original, full text as the Chroma document. Only the
    vector changes; retrieve()/generate()/answer() in rag_pipeline.py
    are unaffected and need no changes.

    Two spans are removed via _remove_boundary_span() above, which
    handles a span sitting entirely inside one chunk OR split across a
    chunk boundary in either direction:

    1. LEADING: the "For any item to be covered by Medicare, it
       must..." preamble that opens every LCD's indication field. This
       span averages ~305 tokens by itself -- already over the
       embedding model's 256-token max_seq_length.
    2. TRAILING: the SWO/WOPD/POD/coding-guideline trailer appended once
       per document before chunking (see START_ANCHORS/END_ANCHOR above).
    """
    text = _remove_boundary_span(text, [LEADING_START_ANCHOR], LEADING_END_ANCHOR)
    text = _remove_boundary_span(text, START_ANCHORS, END_ANCHOR)
    return text


def strip_boilerplate_safe(original_text):
    """
    Wrapper applying both embedding-input-only fixes in sequence:
      1. strip_boilerplate() removes the leading Medicare-purpose preamble
         and trailing SWO/POD span (in whichever of the three forms
         applies -- full span, truncate, or continuation-prefix delete;
         see _remove_boundary_span()).
      2. truncate_code_runs() collapses long HCPCS code enumerations.
    Callers are expected to have already run is_boilerplate_dominant()
    and excluded chunks that are near-empty after this same stripping --
    so in normal use this fallback is defensive, not load-bearing. It
    exists in case strip_boilerplate_safe() is ever called directly on
    unfiltered text: if stripping would leave less than MIN_REAL_CONTENT
    characters, fall back to the original (still-diluted) text rather
    than embedding near-nothing.
    """
    stripped = strip_boilerplate(original_text)
    stripped = truncate_code_runs(stripped)
    if len(stripped) < MIN_REAL_CONTENT:
        return original_text
    return stripped


# --- A.1-redo fix (flagged, not yet resolved, in the handoff doc) ---
#
# strip_boilerplate() was written for the OLD ~650-word chunking regime,
# where a full boilerplate span (start anchor -> end anchor) reliably fit
# inside one chunk. Under the NEW ~190-token, per-field, sentence-aware
# chunking, the ~305-token leading preamble no longer fits in one chunk --
# verified against real output: chunk0 of a short LCD's "indication"
# field contains LEADING_START_ANCHOR but not LEADING_END_ANCHOR; the
# next chunk contains the tail end of the preamble (matching
# LEADING_END_ANCHOR) but not the start.
#
# Tracing strip_boilerplate_safe() against that chunk0 case: the span
# removal can't find its end anchor, so it truncates to
# text[:lead_start] -- collapsing the chunk to just its "[Indication] "
# label, a few characters. That trips the MIN_REAL_CONTENT fallback,
# which reverts to the ORIGINAL, unstripped text. Net effect: chunk0
# gets embedded as near-100% boilerplate, unchanged from before this
# fix existed. Not a crash, not silently wrong data -- just a chunk that
# is genuinely, entirely boilerplate, occupying a retrieval slot with
# zero distinctive signal. Confirmed by testing this exact scenario
# (start anchor present, end anchor absent) before writing the fix
# below, not assumed from the handoff description alone.
#
# The fix isn't more span logic (there's no span to remove -- there's
# nothing else in the chunk). It's detection: a single-anchor presence
# check per chunk, independent of whether a full span matched. If a
# chunk contains ANY known boilerplate anchor (leading or trailing) AND
# span-stripping it would leave less than MIN_REAL_CONTENT of real
# content, the chunk is boilerplate-dominant and is DROPPED from the
# corpus entirely (not embedded, not stored) -- rather than indexed as
# dead weight that can only ever occupy a retrieval slot without ever
# being the right answer.
def is_boilerplate_dominant(text, min_real_content=MIN_REAL_CONTENT):
    """
    True if `text` contains a recognized boilerplate anchor and stripping
    known boilerplate spans/anchors from it leaves under
    `min_real_content` characters of real content -- i.e. this chunk IS
    the boilerplate, not a chunk that merely CONTAINS some boilerplate
    alongside real content (that second case is what strip_boilerplate()
    already handles fine via span removal, e.g. the tail-fragment chunk
    that follows a chunk0 like this one).
    """
    has_any_anchor = (
        LEADING_START_ANCHOR in text
        or LEADING_END_ANCHOR in text
        or any(a in text for a in START_ANCHORS)
        or END_ANCHOR in text
    )
    if not has_any_anchor:
        return False

    # Reuse strip_boilerplate()'s span logic where it applies; for the
    # single-anchor case it collapses to (nearly) the field label alone,
    # which is exactly the signal we want.
    stripped = strip_boilerplate(text)
    stripped = truncate_code_runs(stripped)
    return len(stripped) < min_real_content


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

    # A.1-redo fix: drop boilerplate-dominant chunks (see
    # is_boilerplate_dominant() above) BEFORE embedding/storing, rather
    # than let them be embedded as near-100% boilerplate and occupy a
    # retrieval slot. Logged explicitly since dropping corpus content is
    # a visible, reportable decision, not a silent side effect.
    kept, dropped = [], []
    for c in chunks:
        if is_boilerplate_dominant(c["text"]):
            dropped.append(c["id"])
        else:
            kept.append(c)
    print(f"Dropped {len(dropped)}/{len(chunks)} boilerplate-dominant chunks "
          f"(new chunking regime splits the leading preamble across chunk "
          f"boundaries; these chunks are entirely or almost entirely that "
          f"preamble, with no salvageable content of their own).")
    if dropped:
        print(f"  Dropped IDs: {dropped}")
    chunks = kept

    print(f"Connecting to Chroma at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Rebuild the collection fresh rather than upsert into a possibly
    # pre-existing one: the old 650-word chunk IDs and the new per-field,
    # token-budgeted chunk IDs don't reliably superset one another, so an
    # upsert-only approach could leave stale vectors from the old regime
    # sitting in the collection uncontrolled, contaminating the eval.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted pre-existing '{COLLECTION_NAME}' collection.")
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

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

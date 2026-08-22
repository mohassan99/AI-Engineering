# Interview Notes — Chunking Redo (Stage A.1 → A.4)

This documents the full diagnostic chain behind the chunking rebuild, in
the order it was actually found. The README has a condensed version —
this is the detail worth having ready if asked to walk through it.

## Starting point: Stage A.4 eval

25 hand-written questions, each mapped to a specific source LCD, run
through `retrieve()` only (not full generation) at k=5. Baseline result
on the original word-count chunking (650 words/chunk, ~90-word overlap):

**Precision@5 = 14/25 = 0.560**

The failures weren't noise — they tracked chunk count per LCD almost
perfectly:

| Chunks per LCD | Pass rate |
|---|---|
| 1 | 0% (0/4) |
| 2 | 29% (2/7) |
| 3 | 67% (4/6) |
| 4+ | 100% (8/8) |

## Diagnosis, in order

**1. Leading boilerplate.** Every LCD's first chunk opens with a
near-identical "For any item to be covered by Medicare, it must..."
preamble — CMS's standard framing language, not LCD-specific content.
Measured against the real tokenizer, it averages ~305 tokens on its own.

**2. Trailing boilerplate.** A second near-identical block (Standard
Written Order / Proof of Delivery / coding-guideline language) appears
once per document. In a 1-3 chunk LCD it can be 40-56% of that LCD's
only embedded content.

**3. The real bottleneck: the embedding model's actual token limit.**
`all-MiniLM-L6-v2`'s real `max_seq_length` is 256 tokens — confirmed from
the model's own config, not assumed. Measured against the real
tokenizer: **95% of the original chunks (211/222) exceeded 256 tokens**
and were silently truncated by `sentence-transformers` before ever being
embedded. Mean chunk length was 893 tokens — 3.5x over budget. For one
concrete failing case (Facial Prostheses, 1 chunk), the model's actual
embedded window was 100% generic preamble; the real, one-sentence
coverage criterion sat entirely past the truncation point. Removing the
boilerplate spans alone wasn't enough — even after stripping both, 86%
of chunks still exceeded 256 tokens, because 650-word chunking is
oversized for this model independent of what's in the chunk.

**4. HCPCS (Healthcare Common Procedure Coding System) code
enumerations: expensive and weak.** Tested rather than assumed:
- A single 5-character code costs ~4 subword tokens on its own (e.g.
  `E0470` → `['e','##0','##47','##0']`). A real 61-code list from the
  corpus measured 311 tokens — over the entire budget by itself.
- Cosine similarity between two *different* LCDs' code lists measured
  0.558 — higher than the 0.094 similarity between those same two LCDs'
  actual coverage-criteria prose. Code lists were actively worse than
  prose at telling LCDs apart, while costing far more of the scarce
  token budget.
- At least one real case of a code table being split across a chunk
  boundary was confirmed (a Knee Orthoses LCD).

## The fix

Two complementary layers:

**`build_chunks.py` (rebuilt):** chunks by actual token count against
the real tokenizer (target ~190 tokens, leaving margin under 256), never
splits inside a sentence, and chunks **per document field** (indication,
documentation requirements, bibliography, etc.) rather than one
flattened blob per LCD — so a chunk boundary can never straddle two
unrelated sections.

**`ingest_chunks.py` (updated):** strips the leading/trailing boilerplate
spans and truncates long code enumerations — but only from what gets
*embedded*. The full original text is still what's stored and handed to
the LLM at answer time, so nothing is lost from generation, only from
the vector search step.

## A real correction, made along the way

While rebuilding the fetch logic, testing against a small, non-random
sample of LCDs suggested two expected API fields (`doc_reqs`,
`coding_guidelines`) were coming back empty, with their content
apparently living somewhere else. Testing that properly — against all 58
LCDs, not 4 — confirmed it cleanly: 0/58 have those two fields populated,
58/58 have a different field (`associated_info`) populated instead.

Checking the *existing* corpus's own content settled why: those two
fields were never used in the original build at all — there's no
evidence they were ever populated. The boilerplate analyzed above lives
entirely inside the `indication` field. `associated_info` isn't
recovered or relocated content — it's real content (2,500-7,000+
characters per LCD of genuine documentation-requirements detail) the
original corpus never captured, since that field wasn't part of the
original field list to begin with. `NARRATIVE_FIELDS` has been corrected
to include it.

## Known open item for the next session

Because chunks are now much smaller (~190 tokens vs ~650 words), the
~305-token leading preamble no longer fits inside a single chunk — it
spans two. `ingest_chunks.py`'s current boilerplate-removal logic
(designed for the old large-chunk regime) doesn't cleanly match either
resulting chunk. This is a smaller problem than the original one: with
small, per-field chunks, boilerplate and real content mostly separate
into different chunks naturally rather than diluting each other — the
remaining issue is just that a chunk or two per LCD will be *entirely*
boilerplate on its own, wasted rather than harmful. Worth reconsidering
(likely: detect and drop wholly-boilerplate chunks rather than trying to
span-remove within one) before trusting the current logic against the
new chunk boundaries.

## Results

Precision@5 after the rebuild: **[TBD — pending a live run against the
CMS API; not yet executed end-to-end in one sitting]**

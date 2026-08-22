# Interview Notes — Chunking Redo (Stage A.1 → A.4)

This documents the full diagnostic chain behind the chunking rebuild, in
the order it was actually found. The README has a condensed version —
this is the detail worth having ready if asked to walk through it.

## Starting point: Stage A.4 eval

25 hand-written questions, each mapped to a specific source LCD (Local
Coverage Determination), run through `retrieve()` only (not full
generation) at k=5. Baseline result on the original word-count chunking
(650 words/chunk, ~90-word overlap):

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
`sentence-transformers/all-MiniLM-L6-v2`'s real `max_seq_length` is 256
tokens — confirmed from the model's own config, not assumed. Measured
against the real tokenizer: **95% of the original chunks (211/222)
exceeded 256 tokens** and were silently truncated by
`sentence-transformers` before ever being embedded. Mean chunk length
was 893 tokens — 3.5x over budget. For one concrete failing case (Facial
Prostheses, 1 chunk), the model's actual embedded window was 100%
generic preamble; the real, one-sentence coverage criterion sat entirely
past the truncation point. Removing the boilerplate spans alone wasn't
enough — even after stripping both, 86% of chunks still exceeded 256
tokens, because 650-word chunking is oversized for this model
independent of what's in the chunk.

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

## The rebuild

Two complementary layers:

**`build_chunks.py` (rebuilt):** chunks by actual token count against
the real tokenizer (target ~190 tokens, leaving margin under 256), never
splits inside a sentence (`nltk.sent_tokenize`), and chunks **per
document field** (indication, associated info, bibliography, etc.)
rather than one flattened blob per LCD — so a chunk boundary can never
straddle two unrelated sections.

**`ingest_chunks.py` (updated):** strips known boilerplate spans and
truncates long code enumerations — but only from what gets *embedded*.
The full original text is still what's stored and handed to the LLM at
answer time, so nothing is lost from generation, only from the vector
search step.

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
original field list to begin with. `NARRATIVE_FIELDS` was corrected to
include it.

## The split-boilerplate problem, and the fix that actually closed it

Because chunks are now much smaller (~190 tokens vs ~650 words), the
~305-token leading preamble no longer fits inside a single chunk — it
spans two. The original anchor-based `strip_boilerplate()` (built for
the old large-chunk regime, where a full span reliably fit in one chunk)
only handled two cases:

1. Both start and end anchor present in the same chunk → span removal.
2. Start anchor present, no end anchor → truncate to before the start
   (correctly: nothing salvageable follows in that chunk).

It had no handling at all for the third case — **end anchor present,
start anchor absent** — which is exactly what the *second* half of a
split preamble looks like: a chunk that opens mid-boilerplate (carried
over from the previous chunk) and then continues into real content. That
case fell through untouched, leaving a chunk's embedding input diluted
with a boilerplate tail sitting in front of otherwise-good content.

**Fix:** `_remove_boundary_span()` now handles all three cases
symmetrically. Case 3 (end anchor only) *deletes* the boilerplate prefix
— up through the end anchor — and keeps whatever real content follows,
rather than leaving the chunk untouched or discarding it. This is a
content-preserving deletion, distinct from case 2's truncation, which
correctly discards because there's nothing left to preserve.

**A second bug surfaced while building this fix, caught before it did
real damage:** the same length threshold (originally 200 characters) was
being reused for two different questions that don't share an answer —
"is the embedding input near-empty" (a good check) and "should this
chunk be dropped from the corpus entirely" (a bad use of the same
number). Checked empirically against the real corpus, stripped results
cluster cleanly into two groups with nothing in between: genuinely empty
leftovers (0–12 characters — `""`, `"GENERAL"`, `"[Indication]"`) and
genuinely real, complete sentences (77+ characters). The 200-character
threshold sat *inside* the real-content cluster, not the empty one —
so short-but-complete, correct sentences (confirmed case: `L33738`'s
post-strip content, the single sentence stating facial-prosthesis
coverage criteria, ~140 characters) were being misclassified as
boilerplate-dominant and dropped from the corpus outright. Fixed by
introducing `MIN_REAL_CONTENT = 30`, sized to the actual gap in the
data, replacing the reused 200-character constant for the drop decision.

**Incidental resolution, verified rather than assumed:** a second,
shorter recurring preamble variant ("For the items addressed in this
LCD, the 'reasonable and necessary' criteria... are defined by the
following coverage indications...") shares the same closing clause as
the long-form preamble's end anchor. Because case 3 matches on that
shared end anchor regardless of which start phrasing preceded it, this
variant gets stripped too, with no extra code. Confirmed across the full
corpus: 109 chunks contained the variant; 0 contain it after stripping.

## Results

| Stage | Precision@5 |
|---|---|
| Original word-count chunking (650 words/chunk) | 14/25 = **0.560** |
| Boilerplate stripping alone, old chunking | not separately measured — superseded before this run completed |
| Token-budgeted, per-field rechunk + full boilerplate fix | **25/25 = 1.000** |

136 → 150 chunks (8.9% → 9.8% of the 1,527-chunk corpus) were correctly
identified as boilerplate-dominant and dropped once the threshold bug
was fixed; the earlier, buggy version of the drop check had over-pruned
to 197 chunks, discarding some genuinely useful short chunks in the
process before that was caught.

Spot-checked 6 of the hardest cases (the four originally-failing
1-chunk LCDs plus two code-heavy LCDs) directly against retrieved chunk
*text*, not just source-ID match — confirmed each retrieves the actual,
correct coverage-criteria sentence, not a coincidental match.

## Open items, deliberately not fixed this session

- Some short, kept chunks (e.g. "Claims that do not meet coding
  guidelines shall be denied as not reasonable and necessary...") are
  themselves fairly generic, near-identical procedural language repeated
  across many LCDs — real content, not boilerplate by this project's
  definition, but low differentiation. Not addressed; noted for if
  retrieval quality issues ever trace back to this pattern specifically.

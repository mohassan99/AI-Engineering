# AI Engineering Portfolio Project

RAG (Retrieval-Augmented Generation) → Agents (LangGraph) → Prompt Engineering.

A prior-authorization-style RAG pipeline over DME (Durable Medical Equipment)
LCDs (Local Coverage Determinations) from the CMS (Centers for Medicare &
Medicaid Services) Medicare Coverage Database — 58 active LCDs from
Jurisdiction D (Noridian Healthcare Solutions, LLC), filtered to California.

**Status:** Stage A (RAG) complete — Precision@5 = **1.000**. Stage B
(LangGraph agent orchestration) complete — B.1 framework setup, B.2–B.4
orchestrator build-out, two singleton-race bugs fixed, and the
follow-up-offer prompt fix verified across 3 independent re-runs (see
below). Stage C (documented prompt engineering) not yet started.

## Why this project

DME's real-world improper-payment problem is overwhelmingly a documentation
issue, not a fraud issue — the majority of improper DME payments trace back
to paperwork that doesn't sufficiently establish medical necessity, not to
bad actors. This project builds a system that checks exactly that: given a
device and a beneficiary's clinical situation, does the documentation on
file actually satisfy the LCD's coverage criteria?

## Architecture

```
CMS Coverage API
      |
      v
build_chunks.py -----> data/processed/chunks.jsonl
 (per-field, sentence-      (id, source, text)
  aware, ~190 tokens/chunk)
      |
      v
ingest_chunks.py -----> Chroma (local vector DB, "dme_lcds" collection)
 (strips boilerplate/           (id, embedding, full text, source)
  truncates code lists
  from EMBEDDING INPUT
  only -- full text is
  what's stored)
      |
      v
rag_pipeline.py
  retrieve(query, k) --> top-k chunks from Chroma
  generate(query, chunks) --> Claude Sonnet 5, grounded in those chunks
  answer(query, k) --> retrieve() + generate(), returns answer + sources
      |
      v
run_eval.py --> Precision@5 against data/eval/eval_set.jsonl (25 questions)
```

Retrieval and generation are deliberately separated (`retrieve()` /
`generate()` as independent functions wired together by `answer()`) so that
retrieval quality can be measured on its own — Precision@k evaluates
`retrieve()` directly, never the full generated answer, so a good
retrieval score isn't inflated or masked by generation quality.

## Stack, and why

| Component | Choice | Why |
|---|---|---|
| Vector DB | [Chroma](https://www.trychroma.com/) | Local, no account or hosted infra needed for a portfolio-scale demo; still a real, resume-recognized vector DB. |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | Runs locally, no API cost or external dependency for the ingest step; small enough to iterate on quickly. Trade-off: a hard 256-token context window, which drove most of the chunking work below. |
| LLM | Claude Sonnet 5 | Generation and (in Stage B) agent reasoning. |
| Agent framework (Stage B) | LangGraph | More explicit about state and control flow than LangChain's higher-level chains, which makes the orchestration logic easier to walk through in an interview. |

Explicitly out of scope: fine-tuning, MCP (Model Context Protocol), A2A
(Agent-to-Agent). Real, valuable techniques, but out of scope for a
days-long demonstration project.

## How to run it

```bash
git clone https://github.com/mohassan99/AI-Engineering.git
cd AI-Engineering
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"

# .env with ANTHROPIC_API_KEY=... is required for generate()/answer()/run_eval.py's
# downstream use, though run_eval.py itself only exercises retrieve()

python build_chunks.py     # fetches all 58 LCDs from the live CMS API, ~2-3 min
python ingest_chunks.py    # embeds + indexes into local Chroma, ~1-2 min
python run_eval.py         # runs the 25-question Precision@5 eval, <30s
```

`rag_pipeline.py` can also be run directly (`python rag_pipeline.py`) for a
single example query through the full `answer()` pipeline, including the
generated response and cited sources.

## Retrieval evaluation & the chunking rebuild

A 25-question Precision@5 eval (Stage A.4), each question hand-mapped to a
specific source LCD, run against `retrieve()` only (not full generation).

**Baseline** — original word-count chunking (650 words/chunk, ~90-word
overlap, one flattened blob per LCD): **14/25 = 0.560**. The failures
weren't noise; they tracked chunk count per LCD almost perfectly (1
chunk/LCD = 0% pass, 4+ chunks/LCD = 100% pass).

Diagnosis chain (full detail, including a wrong turn that was caught and
corrected mid-session, in [`INTERVIEW_NOTES.md`](./INTERVIEW_NOTES.md)):

1. Every LCD opens with a near-identical, generic "reasonable and
   necessary" preamble — not LCD-specific content.
2. The embedding model's real `max_seq_length` is 256 tokens (confirmed
   from the model's own config). 95% of the original 650-word chunks
   exceeded that and were silently truncated before ever being embedded —
   for thin LCDs, the model's entire embedding window was consumed by the
   generic preamble, with zero of the LCD's actual coverage criteria ever
   reaching the model.
3. Long HCPCS (Healthcare Common Procedure Coding System) code
   enumerations are both token-expensive (a single 61-code list measured
   311 tokens, over budget by itself) and semantically weak (measured
   cosine similarity showed code lists differentiate *different* LCDs from
   each other *worse* than their actual coverage prose does).

**Fix:** chunking rebuilt to size chunks against the real tokenizer
(~190 tokens/chunk, sentence-aware, chunked per document field rather than
one flattened blob), plus an ingest-time fix that strips generic
boilerplate and truncates long code enumerations from what gets
*embedded* — the full original text is still what's stored and handed to
the LLM at generation time, so nothing is lost from answers, only from the
vector search step.

**Result after the rebuild, run live against the CMS API and reproduced
twice (once in a sandboxed run, once independently on the author's own
machine, matching exactly): 25/25 = 1.000.**

Spot-checked directly against retrieved chunk *text* (not just source-ID
match) for the hardest previously-failing cases — confirmed genuinely
correct, substantive retrieval, not a coincidental score.

## Stage B: LangGraph orchestration

### Architecture decision (locked)

A hand-built LangGraph `StateGraph` (Graph API, not the Functional API) with
one `orchestrator` node (Claude + `.bind_tools([retrieve_tool])`) wired into
a `ToolNode`/`tools_condition` loop — not a two-agent "orchestrator +
specialist" split, and not the higher-level `create_agent` abstraction,
which would hide the routing logic later stages need to log and debug. The
tool wraps `retrieve()` only, not `answer()`, so each turn costs exactly one
Claude generation call (inside the orchestrator), not two.

Four-bucket scope routing (in-scope / wrong jurisdiction / non-DME /
non-traditional plan / fully unrelated — five outcomes across four routing
buckets) is folded directly into the orchestrator's own system prompt
rather than handled by a separate upstream classifier node, to avoid
doubling Claude calls for the common case. Locked decision — not revisited
after B.3.

The system also carries no multi-turn conversation state: each query is a
single, independent `graph.invoke()` call. This is why answers must close
definitively rather than end with a follow-up offer or a clarifying
question — there is no later turn in which the system could receive or use
a reply, so either would be unactionable to the user. For genuinely
ambiguous queries (see B.4 below), the model instead states its default
assumption (e.g. "Traditional Medicare, Jurisdiction D") directly in the
answer, so the user can see what was assumed and re-ask if it doesn't
apply to them.

### B.1 — framework setup & sanity check

Before wiring in the real retriever, a throwaway script (gitignored, not
committed) confirmed the basic building blocks work end-to-end against
Claude Sonnet 5: a dummy tool bound via `bind_tools()`, correctly triggered
on an on-topic query and correctly skipped on an off-topic one, wired into
a full `START → llm_call → tools_condition → tool_node → llm_call → END`
loop using LangGraph's prebuilt `ToolNode`/`tools_condition` utilities
(kept as a deliberate, locked design choice even where the current
LangGraph quickstart docs demonstrate a hand-rolled equivalent instead).

Two environment gotchas worth naming, since they're the kind of thing that
looks like a logic bug but isn't:

- **Per-script credential loading.** `ANTHROPIC_API_KEY` lives in a
  gitignored `.env` file, loaded via `python-dotenv`'s `load_dotenv()` —
  but that call has to happen in *every* entry-point script that hits the
  Anthropic API, not just once globally. A new script that skips it fails
  with an authentication error even though the key is "already set."
  (`orchestrator_graph.py` currently gets away without its own explicit
  call only because it imports `agent_wraps_retrieve`, which happens to
  call `load_dotenv()` as an import-time side effect — fragile, worth
  making explicit rather than relying on import order.)
- **Windows terminal encoding.** Git Bash's default terminal encoding
  (`cp1252`) can't render some Unicode formatting characters that
  LangChain's message-printing helpers emit, throwing a
  `UnicodeEncodeError` on otherwise-correct output. Fixed the same way in
  every other entry-point script (`generate()`, the B.1 sanity check,
  `hallucination_check.py`, `sanity_check_toolcalling.py`,
  `test_pipeline.py`) with
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at each
  script's entry point — a real, recurring cross-platform constraint of
  this dev environment, not a one-off.

### Bugs found and fixed

**B.2 — Chroma collection singleton race.** A lazy singleton for the
Chroma collection object was not thread-safe. Fixed with an eager,
unconditional call at module load, before any concurrent access could
occur.

**B.3 — Embedding-model singleton race (same failure class as B.2, different
singleton).** `_get_embedding_model()` in `rag_pipeline.py` was a
non-thread-safe lazy singleton. LangGraph's `ToolNode` runs multiple tool
calls concurrently in worker threads whenever the orchestrator emits more
than one tool call in a single turn — confirmed this happens routinely
(e.g. a Group II oxygen query triggered 2 parallel `retrieve_tool` calls
with different sub-queries in one turn). If that happens before the
embedding model has ever loaded, both threads race past
`if _embedding_model is None`, and the model gets instantiated twice —
visible as duplicate "Loading weights" progress bars.
**Fix:** added an eager, unconditional `_get_embedding_model()` call at
module level in `rag_pipeline.py`, alongside the existing eager
`_get_collection()` call from B.2, so both singletons are warm on the main
thread before `ToolNode` can ever dispatch a worker thread into them.
**Verified:** a full run shows exactly one "Loading weights" print, at
import time, before any query runs — even though the first query still
issues 2 concurrent tool calls.
`_get_anthropic_client()` was deliberately left lazy — it backs
`generate()`/`answer()`, neither of which is in the B.3/B.4 call path (the
tool only calls `retrieve()`), so no concurrent access is possible. Revisit
only if a future stage exposes `generate()`/`answer()` as a second tool.

**Diagnostic-print cleanup.** A temporary `[orchestrator][diag]` print
(`num_tool_calls=...`, `all_calls=...`) used to confirm the parallel-call
behavior above was removed from `orchestrator_graph.py` before B.4 testing.
Its information was folded into the main observability log line instead,
which was also extended to report *every* tool call in a turn
(`queries=[...]`) rather than just the first — the original log silently
dropped additional parallel calls, under-reporting actual retrieval
activity on multi-call turns.

### B.3 validation

Confirmed via a 5-query test run: the in-scope bucket calls the tool and
answers grounded in real retrieved LCD content (Group II oxygen
concentrator criteria, correctly cited to L33797); the wrong-jurisdiction,
non-DME, and non-traditional-plan buckets all open with the verbatim
caveat sentence and make zero tool calls; the fully-unrelated bucket
answers directly with no caveat.

Also confirmed (not a bug): the orchestrator sometimes splits one user
question into multiple parallel `retrieve_tool` calls with different
sub-queries in a single turn. Legitimate multi-call tool use — flagged in
B.3 to watch during B.4 rather than fix immediately.

### B.4: 15-query test set and results

Test set (in `orchestrator_graph.py`'s `__main__` block, mirrored in
`B4_test_log.xlsx`): 4 in-scope, 2 wrong-jurisdiction, 3 non-DME, 2
non-traditional-plan, 2 fully-unrelated, 2 ambiguous/borderline.

**Result: 14/15 correct on routing decision and caveat verbatim-match.**

- All 4 in-scope queries called the tool and produced grounded, cited
  answers. One (power wheelchair documentation) took 3 rounds of tool
  calls (6 sub-queries total) before answering — the deepest multi-call
  chain observed so far — and still closed cleanly.
- All 7 caveat-bucket queries (wrong jurisdiction ×2, non-DME ×3,
  non-traditional-plan ×2) routed correctly (zero tool calls) and opened
  with the exact required caveat sentence.
- Both unrelated queries answered directly with no caveat, as expected.
- Both ambiguous queries (sleep apnea equipment, post-surgical wound vac)
  were resolved as in-scope — a reasonable default given no jurisdiction
  or plan was stated — and each explicitly named its assumed jurisdiction
  and plan type in the answer rather than asking a clarifying question
  (consistent with the no-multi-turn-state constraint above).

**The one failure:** query 10, "Does my Medicare Advantage plan cover a
wheelchair?" — routed and caveated correctly, but closed with "If you'd
like, I can address what traditional Medicare... requires... as a point of
comparison." This is the same follow-up-offer failure pattern B.2 and B.3
each caught once before (knee-replacement test case, two separate B.3
runs). With this as a third, independent reproduction, it clears the B.4
bar ("re-run apparent failures 2–3 times before concluding systematic
rather than noise") and was treated as confirmed rather than a flake.

**Also flagged, not counted as a failure:** three caveat-bucket answers
(wrong-jurisdiction ×2, one non-DME) closed with an advisory redirect
("I'd recommend checking...", "you would need to consult...") rather than
a literal offer ("I can...", "let me know..."). Not the same failure
pattern, but stylistically adjacent — worth revisiting if the fix below
needs broadening.

### Fix applied (pending re-test)

The system prompt's closing instruction was a pure negative ("do not end
with an offer to follow up") with no contrastive example — exactly the
kind of instruction models follow unreliably. Replaced in
`orchestrator_graph.py` with: an explicit "last sentence = last
substantive fact" framing, a named list of banned phrase patterns drawn
directly from the observed failure ("if you'd like", "let me know if...",
"I can also...", "I'd be happy to...", "would you like...", "I can
address..."), and a contrastive wrong/right example.

**Verified.** Re-ran the full 15-query set a second time, plus two
standalone re-runs of query 10 alone (via a short `python -c` snippet
calling `graph.invoke()` directly). The literal follow-up-offer pattern
("If you'd like, I can address...") did not recur anywhere across all
three independent re-runs of query 10, or in the other 14 queries on the
second full-set run. Fix holds.

**New watch-item surfaced by the second run, not yet a confirmed
failure:** a softer "advisory redirect" close ("you would need to
contact...", "you'd need to consult...") — flagged after the first B.4 run
as adjacent-but-not-a-violation (3/15 queries) — appeared in 6/15 queries
on the second run, including one (knee replacement, non-DME) that closed
cleanly on the first run. It uses none of the explicitly banned phrases
("I can", "let me know", "would you like"), so it isn't a spec violation
under the current instruction, but it's the same underlying
leave-the-door-open impulse in softer form, and it's now reproducible
rather than a one-off. Candidate for tightening in Stage C's documented
v1 → v2 iteration, alongside or instead of treating B.4's fix as final.

## Project files

| File | Purpose |
|---|---|
| `build_chunks.py` | Fetches LCDs from the live CMS Coverage API, chunks per field by real token count |
| `ingest_chunks.py` | Embeds chunks (with boilerplate/code-list stripped from embedding input only) and indexes into Chroma |
| `rag_pipeline.py` | `retrieve()` / `generate()` / `answer()` |
| `hallucination_check.py` | Checks whether generated answers are grounded in retrieved text |
| `run_eval.py` | Precision@5 eval runner |
| `data/eval/eval_set.jsonl` | 25 hand-written eval questions, each mapped to an expected source LCD |
| `INTERVIEW_NOTES.md` | Full diagnostic detail behind the chunking rebuild |
| `agent_wraps_retrieve.py` | B.2 — tool wrap (`retrieve_tool`) around `rag_pipeline.retrieve()` |
| `orchestrator_graph.py` | B.3/B.4 — `StateGraph` orchestrator, scope-routing system prompt, observability logging, 15-query B.4 test set |
| `B4_test_log.xlsx` | B.4 — 15-query test log (query / group / bucket / routing correct? / caveat verbatim? / closed cleanly? / notes) |

## Roadmap

- **Stage B: complete.** LangGraph agent orchestration. B.1 framework
  setup and sanity check, B.2–B.4 orchestrator build-out — two
  singleton-race bugs found and fixed, 15-query test set run twice (14/15
  then 15/15 on the literal follow-up-offer check), fix verified across 3
  independent re-runs (see Stage B section above).
- **Stage C:** documented prompt-engineering iteration (v1 → v2) on the
  system's most important prompt. The B.4 follow-up-offer instruction
  (v1, now verified against its original failure mode) is the natural v1
  baseline for this stage — the newly surfaced "advisory redirect" pattern
  (6/15 on B.4's second run) is a candidate for what v2 tightens next.

# AI Engineering Portfolio Project

RAG (Retrieval-Augmented Generation) → Agents (LangGraph) → Prompt Engineering.

Prior-authorization-style RAG agent over DME (Durable Medical Equipment) LCDs
(Local Coverage Determinations) from the CMS Medicare Coverage Database.

**Status:** Stage A (RAG) in progress. Precision@5 evaluation (below) surfaced a chunking issue; Stage A.1 is being rebuilt to fix it before continuing to A.5.

## Retrieval Evaluation & Chunking Iteration

A 25-question precision@5 eval (Stage A.4) against the original word-count
chunking (650 words/chunk) scored **0.560**, with a clear pattern: retrieval
success correlated almost perfectly with how many chunks an LCD (Local
Coverage Determination) had (1 chunk = 0% pass, 4+ chunks = 100% pass).

Diagnosis: the embedding model (`all-MiniLM-L6-v2`) has a hard 256-token
limit, and most chunks (211/222, ~95%) were 2-4x over that limit and
getting silently truncated before ever being embedded — meaning many
LCDs' actual coverage criteria were never seen by the model at all, only
a shared, generic policy preamble that opens most LCDs.

Chunking was rebuilt (`build_chunks.py`) to size chunks against the real
token limit, respect sentence boundaries, and chunk per document field
rather than one flattened blob per LCD, plus a companion fix
(`ingest_chunks.py`) that keeps generic boilerplate and long procedure-code
enumerations out of what gets embedded, while leaving the full original
text available for the LLM to read at answer time.

New precision@5: **[TBD — pending live re-run]**

Full diagnostic detail (including a wrong turn that was caught and
corrected) is in [`INTERVIEW_NOTES.md`](./INTERVIEW_NOTES.md).

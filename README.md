# AI Engineering Portfolio Project

RAG (Retrieval-Augmented Generation) → Agents (LangGraph) → Prompt Engineering.

A prior-authorization-style RAG pipeline over DME (Durable Medical Equipment)
LCDs (Local Coverage Determinations) from the CMS (Centers for Medicare &
Medicaid Services) Medicare Coverage Database — 58 active LCDs from
Jurisdiction D (Noridian Healthcare Solutions, LLC), filtered to California.

**Status:** Stage A (RAG) complete — Precision@5 = **1.000**. Stage B
(LangGraph agent orchestration) and Stage C (documented prompt engineering)
not yet started.

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

## Roadmap

- **Stage B:** LangGraph agent orchestration — an orchestrator agent that
  routes between the RAG retriever and direct-answer paths.
- **Stage C:** documented prompt-engineering iteration (v1 → v2) on the
  system's most important prompt, with a specific, reproduced failure mode
  and re-tested fix.

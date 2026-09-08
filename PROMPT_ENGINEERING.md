# Prompt Engineering: v1 → v2 (Orchestrator Closing-Instruction Fix)

Stage C deliverable. Documents one specific, reproducible prompt failure
found during B.4 testing, the fix, and the re-test result.

## C.1 — Baseline (v1)

The prompt under iteration is the orchestrator's system prompt in
`orchestrator_graph.py` (`prompts/v1_orchestrator_system_prompt.txt`).

**Known imperfection going in:** the closing-instruction paragraph bans a
specific *literal* phrasing pattern — Claude offering to help further in a
later turn ("if you'd like", "let me know if...") — because this system
does not retain conversation state, so any such offer is unactionable.
That instruction was itself a B.4 fix for a follow-up-offer pattern that
had appeared 4 times across B.2–B.4. But it only bans *that* phrasing. It
does not cover a softer, functionally identical pattern: redirecting the
user to contact or consult a different authority as the "real" answer,
instead of treating the general information already given as complete.

## C.2 — Failure Mode

**Failure mode:** on out-of-scope queries (wrong MAC jurisdiction,
non-DME, non-traditional-Medicare-plan), after correctly giving the
caveat and a general, non-corpus answer, the orchestrator's closing
sentence tells the user to go contact or consult an external
authority for the definitive answer — undercutting the "here's what I
can tell you generally" framing by treating it as provisional rather
than complete, and reproducing the same underlying failure the v1 fix
was meant to close (ending the answer by pointing somewhere else for
resolution) via different surface wording that the v1 instruction's
literal phrase list doesn't catch.

**Measured rate**, same 15-query test set, v1 prompt, two full runs:

| Run | Advisory-redirect count |
|---|---|
| B.4 run 1 | 3 / 15 |
| B.4 run 2 | 6 / 15 |

Full raw output: `B4_test_output.txt`, `B4_test_output_2.txt`.

**Example 1 — reproducible on demand (4/4 runs).**
Query: *"Does my Medicare Advantage plan cover a wheelchair?"*
Closing sentence, run 2:
> "...documentation or medical necessity criteria that can differ from
> traditional Medicare's Local Coverage Determinations. **To find out
> whether your specific plan covers a wheelchair, you would need to
> contact your Medicare Advantage plan directly** or review your plan's
> Evidence of Coverage document."

Two additional ad-hoc re-runs of this exact query (captured at the end of
`B4_test_output_2.txt`) produced the same pattern both times — this is
the most reliable repro case.

**Example 2 — wrong-jurisdiction bucket.**
Query: *"Under Jurisdiction K, what's the coverage policy for a CPAP
device?"*
Closing sentence, run 2:
> "...For jurisdiction-specific documentation requirements, fee
> schedules, or supplier standards for Jurisdiction K, **you would need
> to consult** the Noridian or CGS (depending on current contract
> assignment) LCD and Policy Article specific to that jurisdiction."

**Example 3 — the same query drifted between runs (evidence that this
needs multiple re-runs to catch, not one pass).**
Query: *"Is a knee replacement surgery covered under Medicare?"*
- Run 1 (clean): "...Patients are typically responsible for applicable
  deductibles and coinsurance under both Part A and Part B, depending on
  the care setting." — closes definitively, no redirect.
- Run 2 (failed): "...For specific coverage determinations, **it's
  advisable to consult** the applicable National Coverage Determination
  (NCD) or Local Coverage Determination (LCD) related to major joint
  replacement, as well as **verify with Medicare or a healthcare
  provider** based on individual circumstances."

Same prompt, same query, two different closings. The v1 "verified clean
across 3 independent re-runs" check for the *literal* follow-up-offer
phrasing was real, but it wasn't run against this softer pattern — which
is exactly why it went undetected until B.4's second full pass.

## C.3 — v2 (candidate fix)

**Change made — one thing only**, per the Build Guide's own pitfall
warning: the routing rules, caveat text, and tool description are
untouched. Only the closing-instruction paragraph changed
(`prompts/v2_orchestrator_system_prompt.txt`).

v1 banned one *surface form* of the failure (Claude offering more later).
v2 names the underlying *function* instead — ending the answer by
pointing the user toward resolution somewhere else, whether that's
Claude in a later turn, or a different authority right now — and gives
both banned-phrase families plus a second wrong/right example built
directly from Example 1 above, so the model has a concrete instance of
the exact pattern it was producing, not just an abstract rule.

**Re-test status:** pending. `prompts/c3_retest.py` re-runs the 3
queries above against v1 (as committed, for a live baseline) and then v2,
twice each, without modifying `orchestrator_graph.py` — it patches the
module's `SYSTEM_PROMPT` in memory before each call. Run it locally:

```bash
python prompts/c3_retest.py
```

Paste the output back and I'll fill in the result below.

**Result:** _(pending re-test — 0/6 or a measured reduction from the 6/15
v1 rate, plus a check on whether the knee-replacement query stays clean
across both v2 runs)_

## C.4 — Comparative Write-Up

_(pending C.3 result — final pass to fold this into the main README once
v2 is confirmed)_

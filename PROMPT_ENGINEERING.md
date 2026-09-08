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

**Re-test.** `prompts/c3_retest.py` re-ran the 3 example queries from
C.2 against v1 (as committed, for a live same-session baseline) and then
v2, twice each, without modifying `orchestrator_graph.py` — it patches
the module's `SYSTEM_PROMPT` in memory before each call.

| Query | v1 run 1 | v1 run 2 | v2 run 1 | v2 run 2 |
|---|---|---|---|---|
| MA wheelchair | fail | fail | clean | clean |
| Jurisdiction K CPAP | fail | fail | clean | clean |
| Knee replacement | clean | clean | clean | clean |

**Result: v1 4/6 failed. v2 0/6 failed.**

The more informative result isn't the score, it's *how* v2 stayed clean.
v1's Jurisdiction K run 2 failed with: "...Jurisdiction K...is not part
of this system's corpus, **you should verify** jurisdiction-specific
nuances **directly through** the CGS Medicare LCD..." — phrasing that
matches neither "you would need to" nor "it's advisable to consult," the
two families v2's banned-phrase list explicitly names, and doesn't
overlap with either of the two failing transcripts v2 was written from.
If v2 were just a longer enumerated list, this is exactly the kind of
paraphrase that would slip through it. It didn't: both v2 Jurisdiction K
runs closed clean on genuinely different, non-overlapping wording ("as
long as continued use criteria are met," "if the device is being
replaced due to change in medical necessity rather than routine wear"),
with no redirect of any kind. That's the intended effect of naming the
underlying *function* ("don't point the user toward resolution
elsewhere") rather than matching the redirect's previously-seen exact
phrasings — it generalized past the instances it was written from,
instead of covering only those instances.

Knee replacement was clean on both v1 runs in this re-test (it had
previously failed on B.4's second full-set run and was clean on the
first) — consistent with it being the borderline, non-deterministic case
documented in Example 3, not evidence against the fix.

## C.4 — Comparative Write-Up

| | v1 | v2 |
|---|---|---|
| **Prompt file** | `prompts/v1_orchestrator_system_prompt.txt` | `prompts/v2_orchestrator_system_prompt.txt` |
| **Closing instruction** | Bans a list of literal phrases ("if you'd like", "let me know if...", "I can also...") framed as *the* rule | States the underlying function first ("don't end by pointing the user toward resolution somewhere else — yourself later, or another party now"), with phrase families as illustrations, not the boundary |
| **15-query set, advisory-redirect rate** | 3/15 → 6/15 across two runs (non-deterministic, increasing) | Not yet re-run against the full 15-query set — targeted 3-query re-test only |
| **Targeted 3-query re-test (2 runs each)** | 4/6 failed | 0/6 failed |

**Takeaway:** the lesson here isn't "add more banned phrases whenever you
find a new bad one" — that's whack-a-mole, and v1's own history
demonstrates it (the literal follow-up-offer ban didn't anticipate the
advisory-redirect variant at all, despite both being the same underlying
impulse). The lesson is that a negative instruction defined *as* an
enumerated list only ever covers the surface forms it was written
against. The fix that actually generalizes is one that names the
underlying behavior to avoid, with phrase examples serving as
illustrations of that behavior rather than as its complete definition.
v2's result — 0/6 vs. v1's 4/6, including a clean close on wording that
matches none of v2's own banned-phrase examples — is evidence for that
generalization, not just a bigger list winning on coverage.

**Open item, stated plainly rather than glossed over:** this re-test
targeted the 3 examples that motivated the fix, not a full 15-query
regression run. v2 hasn't yet been checked against the in-scope and
fully-unrelated buckets, which weren't part of this targeted test, or
re-run at B.4's own two-pass standard. A full 15-query run on v2 — ideally
twice, matching v1's own two-run standard — is the natural next step
before calling this fully closed, and would also confirm the fix doesn't
introduce a new failure mode on buckets this targeted test didn't touch.

Folded into the main [`README.md`](./README.md)'s Stage C section and
Roadmap.

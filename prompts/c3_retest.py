"""
Stage C.3 — re-test script.

Re-runs the 3 example queries from the C.2 failure-mode write-up against
v2's system prompt, twice each (to catch drift the way B.4 did — the
knee-replacement case was clean on one run and failed on another with the
identical v1 prompt, so a single pass proves nothing either way).

Does NOT modify orchestrator_graph.py. Imports it, then reassigns the
module-level SYSTEM_PROMPT name before invoking graph.invoke(). This works
because orchestrator() looks up SYSTEM_PROMPT as a global at call time, not
at function-definition time — so the patched value is what actually gets
prepended to messages. v1 in the committed file is untouched.

Run from the project root (same place you run orchestrator_graph.py):
    python prompts/c3_retest.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Running this as `python prompts/c3_retest.py` puts prompts/ (this script's
# own folder) on sys.path, not the project root — so orchestrator_graph.py
# (which lives in the root, one level up) isn't importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage

import orchestrator_graph as og  # noqa: E402

V2_PROMPT_PATH = Path(__file__).parent / "v2_orchestrator_system_prompt.txt"
v2_text = V2_PROMPT_PATH.read_text(encoding="utf-8")

# The 3 C.2 failure-mode example queries, same wording as the B.4 test set.
RETEST_QUERIES = [
    ("MA wheelchair (4/4 reproducible in v1)", "Does my Medicare Advantage plan cover a wheelchair?"),
    ("Jurisdiction K CPAP", "Under Jurisdiction K, what's the coverage policy for a CPAP device?"),
    ("Knee replacement (clean on v1 run 1, failed on v1 run 2)", "Is a knee replacement surgery covered under Medicare?"),
]

RUNS_PER_QUERY = 2


def run_with_prompt(label: str, prompt_text: str):
    og.SYSTEM_PROMPT = SystemMessage(content=prompt_text)
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    for name, q in RETEST_QUERIES:
        for run_num in range(1, RUNS_PER_QUERY + 1):
            print(f"\n--- [{name}] run {run_num} ---")
            result = og.graph.invoke({"messages": [{"role": "user", "content": q}]})
            print(og.extract_text(result["messages"][-1].content))


if __name__ == "__main__":
    # Baseline: v1, exactly as committed (og.SYSTEM_PROMPT is already this on
    # import, but set it explicitly so this script is self-contained and the
    # ordering below can't accidentally matter).
    v1_text = Path(__file__).parent.joinpath("v1_orchestrator_system_prompt.txt").read_text(encoding="utf-8")
    run_with_prompt("v1 (baseline, as committed)", v1_text)
    run_with_prompt("v2 (candidate fix)", v2_text)

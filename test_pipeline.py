"""
Stage A.3 -- Manual Pipeline Test

Runs a handful of test questions through the full answer(query) pipeline
and prints the generated answer plus the sources it drew from, so you can
manually confirm groundedness (not hallucinated) before moving to A.4's
formal precision@k eval.

Adjust TEST_QUERIES below to match specific DME (Durable Medical Equipment)
categories/LCDs (Local Coverage Determinations) you know are in the corpus --
the ones below are seeded from what's already been confirmed present
(oxygen therapy / L33797, ankle-foot orthosis / L33686, PAP therapy /
L33800, L33718) plus a couple of generic DME questions.

Run from the project root with the venv active and ANTHROPIC_API_KEY set
(e.g. in a .env file):
    python test_pipeline.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag_pipeline import answer

TEST_QUERIES = [
    "What are the coverage requirements for an oxygen concentrator?",
    "What documentation is required to support medical necessity for an ankle-foot orthosis?",
    "Under what conditions is a CPAP or PAP therapy device covered?",
    "What is the reimbursement policy for portable oxygen systems?",
    "What clinical criteria must be met for continued oxygen therapy coverage?",
]


def main():
    for i, query in enumerate(TEST_QUERIES, start=1):
        print("=" * 70)
        print(f"[{i}] QUERY: {query}")
        print("=" * 70)

        result = answer(query)

        print(f"\nANSWER:\n{result['answer']}\n")
        print(f"SOURCES USED: {result['sources']}")
        print("\n(Manually check: is this answer grounded in the sources above,")
        print("not hallucinated? Note any issues before moving to A.4.)\n")


if __name__ == "__main__":
    main()

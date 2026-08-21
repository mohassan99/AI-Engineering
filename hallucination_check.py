"""
Stage A.3 -- Hallucination / Grounding Check

For each test query, this script:
  1. Runs the full answer(query) pipeline.
  2. Prints the generated answer.
  3. Dumps the RAW retrieved chunk text (chunks_used) -- the exact text
     Claude was given as context, unedited.
  4. Auto-checks a list of specific claims (flagged from a prior manual
     read of the answers) against that raw chunk text and reports
     FOUND / NOT FOUND for each.

"FOUND" means the claim's literal substring appears somewhere in the
retrieved chunks -- i.e. Claude could have read it there. "NOT FOUND"
means Claude produced that specific claim from outside the context it
was given (its own training knowledge), which is a hallucination in the
RAG sense, regardless of whether the claim happens to be true in the
real world.

Limitation: this is a literal substring search. It catches invented
identifiers, citations, and code lists (which either appear verbatim or
don't) but will NOT catch a paraphrased fact -- that still needs a human
read of the dumped chunk text.

Run from the project root with the venv active and ANTHROPIC_API_KEY set:
    python hallucination_check.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag_pipeline import answer

# Each entry: the exact query to run, plus a list of specific substrings
# to check against the raw retrieved chunk text. Pulled from a manual
# read of the A.3 test_pipeline.py run -- these are the claims flagged
# as needing verification.
QUERIES_WITH_CLAIMS_TO_CHECK = [
    {
        "query": "What are the coverage requirements for an oxygen concentrator?",
        "claims": ["A52514", "Policy Article"],
    },
    {
        "query": "What documentation is required to support medical necessity for an ankle-foot orthosis?",
        "claims": ["L1900", "L1902", "L4631"],
    },
    {
        "query": "Under what conditions is a CPAP or PAP therapy device covered?",
        "claims": [],
    },
    {
        "query": "What is the reimbursement policy for portable oxygen systems?",
        "claims": ["Medicare Program Integrity Manual", "Chapter 5"],
    },
    {
        "query": "What clinical criteria must be met for continued oxygen therapy coverage?",
        "claims": ["Continued Coverage", "61-90", "61st day"],
    },
]


def check_claim(claim, chunks_used):
    """
    Case-insensitive literal substring search for `claim` across all
    retrieved chunk text. Returns (found: bool, matching_chunk_ids: list).
    """
    claim_lower = claim.lower()
    matches = [
        c["id"] for c in chunks_used
        if claim_lower in c["text"].lower()
    ]
    return (len(matches) > 0, matches)


def main():
    for i, item in enumerate(QUERIES_WITH_CLAIMS_TO_CHECK, start=1):
        query = item["query"]
        claims = item["claims"]

        print("=" * 70)
        print(f"[{i}] QUERY: {query}")
        print("=" * 70)

        result = answer(query)

        print(f"\nANSWER:\n{result['answer']}\n")
        print(f"SOURCES USED: {result['sources']}")

        # --- Auto-check flagged claims ---
        if claims:
            print("\n" + "-" * 70)
            print("CLAIM CHECK (literal substring search against retrieved chunks):")
            print("-" * 70)
            for claim in claims:
                found, matching_ids = check_claim(claim, result["chunks_used"])
                status = "FOUND" if found else "NOT FOUND -- possible hallucination"
                print(f"  '{claim}': {status}", end="")
                if found:
                    print(f"  (in chunk(s): {matching_ids})")
                else:
                    print()

        # --- Raw retrieved chunk dump ---
        print("\n" + "-" * 70)
        print("RAW RETRIEVED CHUNKS (chunks_used -- exact text sent to Claude):")
        print("-" * 70)
        for chunk in result["chunks_used"]:
            print(f"\n[chunk id={chunk['id']}  source={chunk['source']}  distance={chunk['distance']:.4f}]")
            print(chunk["text"])

        print("\n")


if __name__ == "__main__":
    main()

"""
Stage A.4 -- Evaluation Set & Precision@k

Runs the 25-question eval set (data/eval/eval_set.jsonl) through
retrieve() ONLY (not the full answer() pipeline) -- precision@k measures
retrieval quality, not generation quality. For each question, checks
whether the expected source LCD (Local Coverage Determination) appears
anywhere in the top-k retrieved chunks' source metadata.

Run from the project root with the venv active:
    python run_eval.py

Prints a per-question pass/fail line plus the overall precision@k score,
and writes a full results log to logs/a4_eval_results.txt for the README
and interview reference.
"""

import json
import os

from rag_pipeline import retrieve

EVAL_SET_PATH = "data/eval/eval_set.jsonl"
LOG_PATH = "logs/a4_eval_results.txt"
K = 5  # matches rag_pipeline.py's DEFAULT_K


def read_eval_set(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    eval_items = read_eval_set(EVAL_SET_PATH)
    total = len(eval_items)
    passed = 0

    os.makedirs("logs", exist_ok=True)
    log_lines = []
    log_lines.append(f"Stage A.4 -- Precision@{K} Eval Run")
    log_lines.append(f"Eval set: {EVAL_SET_PATH} ({total} questions)")
    log_lines.append("=" * 70)

    for i, item in enumerate(eval_items, start=1):
        question = item["question"]
        expected_sources = set(item["expected_source"])

        chunks = retrieve(question, k=K)
        retrieved_sources = {c["source"] for c in chunks if c["source"]}

        hit = bool(expected_sources & retrieved_sources)
        if hit:
            passed += 1

        status = "PASS" if hit else "FAIL"
        line = (
            f"[{i:2d}/{total}] {status}  expected={sorted(expected_sources)}  "
            f"retrieved_sources={sorted(retrieved_sources)}\n"
            f"        Q: {question}"
        )
        print(line)
        log_lines.append(line)

    precision_at_k = passed / total if total else 0.0
    summary = f"\nPrecision@{K}: {passed}/{total} = {precision_at_k:.3f}"
    print(summary)
    log_lines.append(summary)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\nFull results written to {LOG_PATH}")


if __name__ == "__main__":
    main()

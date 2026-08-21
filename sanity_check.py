"""
Stage A.2 — Sanity Check

Embeds a single test query and runs a similarity search against the
already-populated 'dme_lcds' Chroma collection. This is a manual,
eyeball-it check — not the formal precision@k eval (that's A.4).

Run from the project root with the venv active:
    python sanity_check.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "dme_lcds"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# A generic, common DME (Durable Medical Equipment) category — should map
# cleanly to one or more specific LCDs (Local Coverage Determinations) in
# the corpus if retrieval is working.
TEST_QUERY = "What are the coverage requirements for an oxygen concentrator?"
TOP_K = 5


def main():
    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Connecting to Chroma at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' item count: {collection.count()}")

    print(f"\nTest query: \"{TEST_QUERY}\"")
    query_embedding = model.encode([TEST_QUERY]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\nTop {TOP_K} results:\n" + "=" * 60)
    for rank, (chunk_id, doc, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        print(f"\n[{rank}] id={chunk_id}  source={meta.get('source')}  distance={dist:.4f}")
        preview = doc[:300].replace("\n", " ")
        print(f"    {preview}...")

    print("\n" + "=" * 60)
    print("Manually review: do these chunks look relevant to the test query?")
    print("If yes, A.2 is complete. If not, note it — this is a flag to")
    print("revisit in A.4 (evaluation set & precision@k), not to fix blind now.")


if __name__ == "__main__":
    main()

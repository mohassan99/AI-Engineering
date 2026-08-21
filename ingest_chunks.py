import json
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/chunks.jsonl"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "dme_lcds"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def read_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


if __name__ == "__main__":
    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = read_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads it, ~80MB)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Connecting to Chroma at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # get_or_create so re-runs don't fail if the collection already exists
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]

    print("Embedding all chunks (this is the slow step, batched)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    print("Upserting into Chroma...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    count = collection.count()
    print(f"\nDone. Collection '{COLLECTION_NAME}' item count: {count}")
    print(f"Expected: {len(chunks)}")
    if count != len(chunks):
        print("WARNING: collection count does not match chunk count — investigate before proceeding.")

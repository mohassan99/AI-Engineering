"""
Stage A.3 -- Retrieval + Generation Pipeline

Wires the confirmed-working 'dme_lcds' Chroma collection (Stage A.2) into a
full RAG (Retrieval-Augmented Generation) pipeline:

    retrieve(query, k)      -> top-k chunks from Chroma
    generate(query, chunks) -> LLM-generated answer grounded in those chunks
    answer(query, k)        -> wires the two together, returns answer + sources

Per the build guide's "No source attribution" pitfall, answer() returns
which chunks/sources were used, not just the final answer text -- this is
required groundwork for Stage A.4's precision@k eval.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "dme_lcds"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-5"
DEFAULT_K = 5

# Lazy singletons -- avoid reloading the embedding model / reconnecting to
# Chroma / re-instantiating the Anthropic client on every call.
_embedding_model = None
_collection = None
_anthropic_client = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _anthropic_client


def retrieve(query, k=DEFAULT_K):
    """
    Embed `query` with the same embedding model used at ingest time
    (all-MiniLM-L6-v2) and search the Chroma collection for the top-k
    most similar chunks.

    Returns a list of dicts: [{"id", "text", "source", "distance"}, ...]
    """
    model = _get_embedding_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    chunks = []
    for chunk_id, text, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "id": chunk_id,
            "text": text,
            "source": meta.get("source"),
            "distance": dist,
        })
    return chunks


def generate(query, chunks):
    """
    Build a prompt that includes the retrieved chunks as context, call
    Claude Sonnet 5, and return the generated answer text.

    NOTE: this returns new LLM-produced text grounded in the chunks -- it
    does NOT return the chunks themselves. That distinction matters for
    precision@k in A.4, which evaluates retrieve(), not generate().
    """
    client = _get_anthropic_client()

    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    system_prompt = (
        "You are a prior-authorization assistant for a Medicare DME "
        "(Durable Medical Equipment) payer. Answer the user's question "
        "using ONLY the LCD (Local Coverage Determination) excerpts "
        "provided below as context. If the context does not contain "
        "enough information to answer confidently, say so explicitly "
        "rather than guessing. Cite the source LCD(s) you relied on."
    )

    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}"

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # response.content is a list of blocks. Claude Sonnet 5 can emit a
    # "thinking" block (its internal reasoning) before the "text" block
    # with the actual answer -- content[0] is not reliably the answer.
    # Concatenate every text block instead of assuming a fixed position.
    text_blocks = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_blocks)


def answer(query, k=DEFAULT_K):
    """
    Full RAG pipeline: retrieve() -> generate().

    Returns a dict with the generated answer AND the sources used:
        {"query": ..., "answer": ..., "sources": [...], "chunks_used": [...]}
    """
    chunks = retrieve(query, k=k)
    generated_answer = generate(query, chunks)
    sources = sorted({c["source"] for c in chunks if c["source"]})

    return {
        "query": query,
        "answer": generated_answer,
        "sources": sources,
        "chunks_used": chunks,
    }


if __name__ == "__main__":
    test_query = "What are the coverage requirements for an oxygen concentrator?"
    result = answer(test_query)
    print(f"Query: {result['query']}\n")
    print(f"Answer:\n{result['answer']}\n")
    print(f"Sources used: {result['sources']}")

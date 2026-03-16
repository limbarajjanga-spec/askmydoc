# rag/retriever.py
from rag.vectorstore import get_collection
from config.constants import TOP_K_RESULTS


def retrieve_similar_chunks(query_vector: list[float],
                            top_k: int = TOP_K_RESULTS,
                            source_filter: str = None) -> list[dict]:
    """
    Retrieves top-k chunks. If source_filter is given,
    only returns chunks from that specific document.
    """
    collection = get_collection()

    if collection.count() == 0:
        raise ValueError("No documents in vector store. Upload a file first.")

    # Build where filter for specific document
    where = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    enriched = []
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        enriched.append({
            "text": chunk,
            "page": meta.get("page", "?"),
            "source": meta.get("source", "?"),
            "score": round(1 - dist, 3)
        })

    print(f"[retriever] Retrieved {len(enriched)} chunks "
          f"from '{source_filter}'")
    return enriched
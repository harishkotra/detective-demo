import chromadb
import os
from ollama_client import embed, llm_complete

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "detective_case"


def get_or_create_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    return collection


def index_docs(docs: list[dict]):
    collection = get_or_create_collection()
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    embeddings = embed(texts)
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=[{"title": d["title"]} for d in docs],
        embeddings=embeddings,
    )
    return collection


def retrieve(collection, query: str, top_k: int = 3) -> list[dict]:
    q_emb = embed([query])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = []
    for i, doc_id in enumerate(results["ids"][0]):
        docs.append(
            {
                "id": doc_id,
                "text": results["documents"][0][i],
                "title": results["metadatas"][0][i]["title"],
                "score": results["distances"][0][i]
                if results.get("distances")
                else None,
            }
        )
    return docs


def answer(docs: list[dict], question: str) -> str:
    context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    prompt = f"""Answer the question based only on the provided documents.

Documents:
{context}

Question: {question}

Answer concisely:"""
    return llm_complete(prompt, temperature=0, max_tokens=256)

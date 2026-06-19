# Detective's Notebook — RAG vs HippoRAG Demo

A self-contained, fully local demo comparing **Standard RAG (ChromaDB)** against **HippoRAG-style retrieval (Knowledge Graph + Personalized PageRank)** on a multi-hop detective mystery.

## The Case: "The Purloined Pearl"

Ten scattered case-file snippets — witness statements, alibi records, and evidence logs about a stolen $2M pearl necklace. Six queries test each system across two dimensions:

| Query | Type | What it tests |
|-------|------|---------------|
| "What time did Detective Miller arrive?" | Simple fact | Single-doc lookup |
| "What is the value of the stolen necklace?" | Simple fact | Single-doc lookup |
| "Where was Mr. Blackwood during the theft?" | Simple fact | Single-doc lookup |
| "Who had both motive and opportunity?" | **Multi-hop** | Cross-document reasoning |
| "How does the 9:15–11:30 timeline connect Victoria Crane?" | **Multi-hop** | Temporal + evidence linking |
| "Given all evidence, who is the most likely thief?" | **Sense-making** | Holistic reasoning |

## How It Works

Both systems use the **same LLM** (`phi4-mini` via Ollama) and **same embeddings** (`nomic-embed-text`). The difference is only in retrieval strategy:

### Standard RAG (`rag/rag_pipeline.py`)
- Embeds all documents with `nomic-embed-text`
- Stores them in ChromaDB
- Retrieves top-3 chunks by cosine similarity
- Feeds chunks + question to LLM

### HippoRAG-Style (`hipporag_style/`)
1. **OpenIE** — Extracts `(subject, relation, object)` triples from each document (`openie.py`)
2. **Knowledge Graph** — Builds a NetworkX graph where entities and passages are nodes, triples are edges (`graph_store.py`)
3. **Graph Retrieval** — Extracts entities from the query, runs **Personalized PageRank** on the graph, ranks passages by their PPR score (`retriever.py`)
4. **QA** — Feeds highest-ranked passages + graph traversal path to LLM (`qa.py`)

## Requirements

- Python ≥ 3.10
- [Ollama](https://ollama.com) running locally
- Models pulled: `phi4-mini`, `nomic-embed-text`

```bash
ollama pull phi4-mini
ollama pull nomic-embed-text
```

## Install

```bash
pip install chromadb networkx requests
```

## Usage

```bash
cd detective-demo
python run_comparison.py
```

The script:
1. Indexes all 10 documents into ChromaDB
2. Extracts triples and builds the knowledge graph
3. Runs all 6 queries through both systems
4. Prints a side-by-side comparison with retrieved documents, graph paths, and answers

## Project Structure

```
detective-demo/
├── data/
│   ├── case_docs.json         # 10 case document snippets
│   └── queries.json           # 6 test queries with expected answers
├── rag/
│   └── rag_pipeline.py        # ChromaDB index, retrieve, answer
├── hipporag_style/
│   ├── openie.py              # Entity/relation triple extraction
│   ├── graph_store.py         # Knowledge graph (NetworkX)
│   ├── retriever.py           # PPR-based graph retrieval
│   └── qa.py                  # QA with graph context
├── ollama_client.py           # Ollama LLM + embedding wrapper
└── run_comparison.py          # Main comparison script
```

## Notes

- First run is slow: each LLM call on CPU takes 1–5 minutes. On a GPU-equipped machine this drops to seconds.
- The `chroma_db/` directory (created at runtime) is ephemeral and should not be committed.
- HippoRAG 2's full implementation requires vLLM/CUDA — this demo implements its core concepts (OpenIE + knowledge graph + PPR retrieval) using lightweight, local-friendly dependencies.

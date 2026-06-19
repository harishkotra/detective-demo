# The Purloined Pearl — RAG vs HippoRAG Detective Demo

A fully local, self-contained Python demo that compares **Standard RAG (ChromaDB + vector similarity)** against **HippoRAG-style retrieval (Knowledge Graph + Personalized PageRank)** on a multi-hop detective mystery.

Built with a murder-mystery-style case ("The Purloined Pearl") — 10 scattered witness statements, alibi records, and evidence logs about a stolen $2M necklace. Six queries test each system across simple fact lookups and complex multi-hop reasoning.

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/your-username/detective-demo
cd detective-demo
python3 -m venv venv && source venv/bin/activate
pip install chromadb networkx requests streamlit scipy

# 2. Start LM Studio (or any OpenAI-compatible server on port 1234)
#    Load a model like qwen/qwen3.5-9b for QA and
#    google/gemma-4-e4b for triple extraction
#    Load text-embedding-nomic-embed-text-v1.5:2 for embeddings

# 3. Run CLI comparison
python run_comparison.py

# 4. Or launch the Streamlit UI
streamlit run streamlit_app.py
```

---

## The Case: "The Purloined Pearl"

Ten case-file snippets form the evidence corpus:

| ID | Title | Key Fact |
|----|-------|----------|
| D1 | Detective's Arrival | Miller arrives at 10 PM; necklace last seen in safe |
| D2 | Mr. Blackwood's Alibi | Owner at charity gala 7–10:30 PM |
| D3 | Groundskeeper's Statement | James Riley near mansion at 9:45 PM |
| D4 | Victoria Crane's Visit | Sister visits 8–9:30 PM, leaves upset after argument |
| D5 | Pawn Shop Report | Woman tries to sell matching necklace at 11:30 PM |
| D6 | License Plate Sighting | Victoria's car on Lombard St at 11:25 PM |
| D7 | Security Footage | Figure in long coat in hallway at 9:15 PM |
| D8 | Victoria's Financial Trouble | Lost job, facing foreclosure |
| D9 | Photo Lineup | Pawn shop owner identifies Victoria |
| D10 | The Long Coat | Mud-stained coat found in Victoria's trunk |

Six queries probe increasing difficulty:

| Query | Type | Tests |
|-------|------|-------|
| "What time did Miller arrive?" | Simple | Single-doc fact lookup |
| "What is the necklace value?" | Simple | Single-doc fact lookup |
| "Where was Mr. Blackwood?" | Simple | Single-doc fact lookup |
| "Who had motive AND opportunity?" | **Multi-hop** | Cross-doc reasoning |
| "How does the 9:15–11:30 timeline connect Victoria?" | **Multi-hop** | Temporal + evidence linking |
| "Who is the most likely thief?" | **Sense-making** | Holistic reasoning |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  run_comparison.py / streamlit_app.py   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─── Standard RAG ─────────────────────────────────┐  │
│  │  Documents ──► embed() ──► ChromaDB (index)      │  │
│  │  Query ──► embed() ──► cosine similarity ──► top-k│  │
│  │  chunks ──► LLM ──► answer                        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─── HippoRAG ─────────────────────────────────────┐  │
│  │  Documents ──► OpenIE triple extraction           │  │
│  │       │                                           │  │
│  │       ▼                                           │  │
│  │  Knowledge Graph (NetworkX)                       │  │
│  │  ┌──────┐   ┌──────────┐   ┌──────┐              │  │
│  │  │Entity│──►│ Passage  │◄──│Entity│              │  │
│  │  │  A   │   │  Node    │   │  B   │              │  │
│  │  └──┬───┘   └──────────┘   └──┬───┘              │  │
│  │     │         ▲               │                   │  │
│  │     └─────────┼───────────────┘                   │  │
│  │           triple edge                              │  │
│  │                                                    │  │
│  │  Query ──► extract entities ──► Personalized PPR   │  │
│  │       on KG ──► rank passages ──► LLM ──► answer   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─── ollama_client.py ─────────────────────────────┐  │
│  │  llm_complete(prompt) —► POST /v1/chat/completions│  │
│  │  embed(texts) —► POST /v1/embeddings              │  │
│  │  Targets: LM Studio / Ollama / any OpenAI API     │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Indexing Phase**: Documents are embedded into Chromadb (standard RAG) and also fed through OpenIE triple extraction to build the knowledge graph.
2. **Retrieval Phase**: A query is processed by both systems independently:
   - **Standard RAG**: Embed query → cosine search ChromaDB → top-3 chunks
   - **HippoRAG**: Extract query entities → PPR on KG → aggregate passage scores → top-3 passages
3. **Answering Phase**: Retrieved passages (+ graph traversal path for HippoRAG) are fed to an LLM that produces the final answer.

---

## Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Vector Store | **ChromaDB** | Persistent vector index with cosine similarity search |
| Knowledge Graph | **NetworkX** | In-memory graph with entity/passage nodes and triple edges |
| PageRank | **NetworkX (pagerank)** | Personalized PageRank for graph-based retrieval |
| LLM Backend | **LM Studio / Ollama** | OpenAI-compatible local inference server |
| Embeddings | **nomic-embed-text-v1.5** | 768-dim text embeddings via LM Studio |
| UI | **Streamlit** | Interactive side-by-side comparison dashboard |
| Structured Extraction | **LLM + regex** | Zero-shot triple/entity extraction via prompting |
| API Client | **requests** | Python HTTP client for LM Studio's OpenAI-compatible API |

---

## Code Walkthrough

### 1. LLM Client (`ollama_client.py`)

Central wrapper that talks to any OpenAI-compatible endpoint. Two main functions:

```python
# Chat completion with reasoning-model fallback
def llm_complete(prompt, model="qwen/qwen3.5-9b", ...):
    resp = requests.post(f"{BASE}/v1/chat/completions", json={...})
    msg = resp.json()["choices"][0]["message"]
    content = msg.get("content", "")
    if not content and "reasoning_content" in msg:
        content = msg["reasoning_content"]
    return content.strip()

# Embedding (accepts list of strings, returns list of vectors)
def embed(texts, model="nomic-embed-text-v1.5:2"):
    resp = requests.post(f"{BASE}/v1/embeddings", json={"input": texts})
    return [d["embedding"] for d in resp.json()["data"]]
```

### 2. Standard RAG (`rag/rag_pipeline.py`)

Documents are embedded and stored in ChromaDB with cosine distance:

```python
def index_docs(docs):
    collection = get_or_create_collection()
    embeddings = embed([d["text"] for d in docs])
    collection.add(ids=ids, documents=texts, embeddings=embeddings)
    return collection

def retrieve(collection, query, top_k=3):
    q_emb = embed([query])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    return [{"id": id, "text": text, "score": dist}
            for id, text, dist in zip(...)]
```

### 3. HippoRAG — Knowledge Graph (`hipporag_style/`)

**OpenIE triple extraction** turns document text into structured facts:

```python
# openie.py — prompt the LLM to extract (subject, relation, object)
def extract_triples(text, doc_id):
    prompt = "Extract triples as JSON array from: " + text
    raw = llm_complete(prompt, model=SMALL_LLM)
    triples = json.loads(re.search(r'\[.*?\]', raw, re.DOTALL).group())
    return [t | {"doc_id": doc_id} for t in triples]
```

**Knowledge Graph** builds a bipartite graph of entities and passages:

```python
# graph_store.py — NetworkX graph
class KnowledgeGraph:
    def add_triple(self, subj, rel, obj, doc_id):
        self.graph.add_edge(f"entity:{subj}", f"passage:{doc_id}", ...)
        self.graph.add_edge(f"entity:{obj}", f"passage:{doc_id}", ...)
        self.graph.add_edge(f"entity:{subj}", f"entity:{obj}",
                            relation=rel, triple=f"({subj}) --[{rel}]-> ({obj})")
```

**Personalized PageRank** retrieves passages relevant to a query:

```python
# retriever.py — PPR with query entities as seeds
def retrieve_hipporag(graph, question, top_k=3):
    entities = extract_query_entities(question)
    personalization = {f"entity:{e}": 1.0 for e in entities}
    pr = nx.pagerank(graph.graph, alpha=0.5, personalization=personalization)
    passage_scores = sum pr scores for passage nodes + entity neighbors
    return sorted passages by score, return top_k
```

### 4. QA (`rag/rag_pipeline.py` and `hipporag_style/qa.py`)

Both approaches feed retrieved context to the LLM:

```python
# Standard RAG
prompt = f"Answer based only on: {documents}\n\nQuestion: {question}"
return llm_complete(prompt)

# HippoRAG — also includes graph traversal path as additional context
prompt = f"Documents: {documents}\nGraph connections: {trace}\nQuestion: {question}"
return llm_complete(prompt)
```

---

## Comparison: Standard RAG vs HippoRAG

| Dimension | Standard RAG | HippoRAG |
|-----------|-------------|----------|
| Retrieval signal | Cosine similarity of embeddings | Personalized PageRank on KG |
| Cross-doc linking | None (each chunk is independent) | Entities connect passages implicitly |
| Multi-hop reasoning | Requires LLM to infer connections | KG provides explicit relation paths |
| Indexing cost | One embedding per document | N LLM calls (one per doc for triple extraction) |
| Retrieval latency | ~100ms (vector search) | ~50ms (PageRank on small graph) |
| Cold-start | Immediate | Requires triple extraction + graph build |
| Interpretability | Black-box similarity scores | Graph traversal paths are inspectable |

### When HippoRAG Wins

- **Multi-hop queries**: "Who had motive AND opportunity?" requires connecting Victoria's financial trouble (D8), her presence in the mansion (D4), and the security footage (D7). Standard RAG might retrieve D8 and D7 but has no explicit link between them. HippoRAG's KG has edges: `Victoria Crane -> mentioned_in -> D8`, `Victoria Crane -> mentioned_in -> D7`, and `Victoria Crane -> argued_with -> Mr. Blackwood`, making the connection explicit.

- **Entity-centric queries**: Questions mentioning "Victoria Crane" activate the entity node, which spreads probability to all connected passages via PPR — naturally aggregating evidence across documents.

### When Standard RAG Wins

- **Simple fact lookups**: "What time did Miller arrive?" — a single document contains the answer. Vector search is faster and requires no graph infrastructure.
- **Semantic similarity**: Queries phrased differently from document text can still match via embedding similarity.

---

## Project Structure

```
detective-demo/
├── data/
│   ├── __init__.py          # load_docs(), load_queries()
│   ├── case_docs.json       # 10 evidence snippets
│   └── queries.json         # 6 test queries with expected answers
├── rag/
│   ├── __init__.py
│   └── rag_pipeline.py      # ChromaDB index, vector retrieve, LLM answer
├── hipporag_style/
│   ├── __init__.py
│   ├── openie.py            # LLM-based triple/entity extraction
│   ├── graph_store.py       # NetworkX knowledge graph builder
│   ├── retriever.py         # PPR-based graph retrieval + path tracing
│   └── qa.py                # QA with graph context augmentation
├── ollama_client.py         # OpenAI-compatible LLM + embedding client
├── run_comparison.py        # CLI comparison runner
├── streamlit_app.py         # Interactive Streamlit dashboard
└── README.md
```

---

## How to Contribute

### Ideas for New Features

1. **Real OpenIE backend** — Replace LLM-based triple extraction with a proper OpenIE tool (e.g., Stanford OpenIE, SPICE) for faster, deterministic extraction.

2. **Hybrid retrieval** — Combine vector similarity scores with PPR scores using weighted fusion (e.g., `score = α * cosine_sim + (1-α) * ppr_score`).

3. **Graph visualization** — Use Pyvis or D3.js to render the knowledge graph interactively in the Streamlit UI, highlighting traversal paths per query.

4. **Larger datasets** — Replace the detective case with a bigger corpus (e.g., Wikipedia articles, legal documents, medical literature) to stress-test both approaches.

5. **HippoRAG L2 (IDF weighting)** — The real HippoRAG paper uses inverse document frequency to weight entity importance. Implement IDF-based entity ranking for PPR seeds.

6. **Multi-query expansion** — Use the LLM to generate multiple sub-queries from the original question, run PPR for each, and aggregate results.

7. **Chunked documents** — Split long documents into smaller passages. Each passage becomes a separate node in both ChromaDB and the KG, with entity edges pointing to specific chunks.

8. **Caching layer** — Cache triple extractions and embeddings to disk so rebuilding is instant on subsequent runs.

9. **Comparison metrics** — Add precision, recall, MRR, and answer-accuracy metrics computed across the query set.

10. **OpenAI / Anthropic backends** — Add support for cloud LLM backends so users can compare with GPT-4 or Claude.

### Pull Request Process

1. Fork the repo and create a feature branch
2. Keep the dependency footprint minimal — prefer stdlib where possible
3. Add or update tests in the relevant module
4. Run `python run_comparison.py` to verify nothing is broken
5. Open a PR with a clear description of the change

---

## Requirements

- Python >= 3.10
- LM Studio (or any OpenAI-compatible inference server) running on `http://127.0.0.1:1234`
- Models loaded in LM Studio:
  - `qwen/qwen3.5-9b` (or any chat model) for QA
  - `google/gemma-4-e4b` (or small model) for triple extraction
  - `text-embedding-nomic-embed-text-v1.5:2` for embeddings

---

## License

MIT

---

## Credits

Built by [Harish Kotra](https://harishkotra.me)

Check out [other builds on DailyBuild](https://dailybuild.xyz)

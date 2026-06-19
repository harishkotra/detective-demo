#!/usr/bin/env python3
"""
The Purloined Pearl — Detective's Notebook Demo
Compares Standard RAG (ChromaDB) vs HippoRAG-style (Knowledge Graph + PPR)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data import load_docs, load_queries
from rag.rag_pipeline import index_docs, retrieve as rag_retrieve, answer as rag_answer
from hipporag_style.openie import extract_triples
from hipporag_style.graph_store import KnowledgeGraph
from hipporag_style.retriever import retrieve_hipporag, trace_path
from hipporag_style.qa import answer_with_retrieved as hipporag_answer


SEPARATOR = "=" * 72


def run_comparison():
    docs = load_docs()
    queries = load_queries()

    # ── Phase 1: Index docs ──
    print("Indexing documents with ChromaDB (Standard RAG)...")
    collection = index_docs(docs)
    print(f"  Indexed {len(docs)} documents\n")

    # ── Phase 2: Build Knowledge Graph (HippoRAG-style) ──
    print("Extracting triples and building knowledge graph (HippoRAG)...")
    all_triples = {}
    for doc in docs:
        print(f"  {doc['id']}: extracting triples...")
        triples = extract_triples(doc["text"], doc["id"])
        all_triples[doc["id"]] = triples
        print(f"    → {len(triples)} triples")

    graph = KnowledgeGraph()
    graph.build_from_docs(docs, all_triples)
    print(f"\n  Knowledge graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges\n")

    # ── Phase 3: Run queries through both systems ──
    for q in queries:
        print(SEPARATOR)
        print(f"QUERY [{q['id']}] ({q['type']}): {q['question']}")
        print(SEPARATOR)

        # ── Standard RAG ──
        print("\n[Standard RAG — ChromaDB + Ollama]")
        rag_docs = rag_retrieve(collection, q["question"], top_k=3)
        for d in rag_docs:
            score_str = f" (score: {d['score']:.4f})" if d["score"] else ""
            print(f"  Retrieved: {d['id']} {score_str}")
            print(f"    \"{d['text'][:80]}...\"")
        rag_result = rag_answer(rag_docs, q["question"])
        print(f"\n  Answer: {rag_result}\n")

        # ── HippoRAG ──
        print("[HippoRAG — Knowledge Graph + PPR]")
        hippo_docs = retrieve_hipporag(graph, q["question"], top_k=3)
        path = trace_path(graph, q["question"])
        for d in hippo_docs:
            print(f"  Retrieved: {d['id']} (score: {d['score']:.6f})")
            print(f"    \"{d['text'][:80]}...\"")
        if path:
            print("\n  Graph traversal path:")
            for line in path:
                print(f"    {line}")
        hippo_result = hipporag_answer(hippo_docs, q["question"], path)
        print(f"\n  Answer: {hippo_result}\n")

        # ── Expected ──
        print(f"[Expected] {q['expected_answer']}")
        print()

    print(SEPARATOR)
    print("DEMO COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    run_comparison()

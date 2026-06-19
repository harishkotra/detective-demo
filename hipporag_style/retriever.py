import json
import re
import networkx as nx
from collections import defaultdict
from ollama_client import llm_complete, embed, DEFAULT_EMBED
from hipporag_style.graph_store import KnowledgeGraph


QUERY_ENTITY_PROMPT = """Identify the key entities (people, places, objects) mentioned in this question.

Return ONLY a JSON array of strings.

Question: {question}
"""


def extract_query_entities(question: str) -> list[str]:
    prompt = QUERY_ENTITY_PROMPT.format(question=question)
    raw = llm_complete(prompt, temperature=0, max_tokens=256)
    json_match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if not json_match:
        return []
    try:
        return json.loads(json_match.group())
    except:
        return []


def personalized_pagerank(graph: nx.Graph, seed_entities: list[str],
                          damping: float = 0.5, max_iter: int = 200) -> dict[str, float]:
    personalization = {}
    for node in graph.nodes():
        personalization[node] = 0.0

    for entity in seed_entities:
        node = f"entity:{entity}"
        if node in personalization:
            personalization[node] = 1.0

    if sum(personalization.values()) == 0:
        return {}

    total = sum(personalization.values())
    for k in personalization:
        personalization[k] /= total

    try:
        pr = nx.pagerank(graph, alpha=damping, personalization=personalization,
                          max_iter=max_iter, tol=1e-6)
        return pr
    except:
        return {}


KEYWORD_ENTITY_PROMPT = """Extract the most important entities (people, places, things) from this text.
Return a comma-separated list of entity names.

Text: {text}
"""


def retrieve_hipporag(graph: KnowledgeGraph, question: str, top_k: int = 5) -> list[dict]:
    entities = extract_query_entities(question)

    if not entities:
        keyword_prompt = KEYWORD_ENTITY_PROMPT.format(text=question)
        raw_keywords = llm_complete(keyword_prompt, temperature=0, max_tokens=128)
        entities = [e.strip() for e in raw_keywords.split(",") if e.strip()]

    pr_scores = personalized_pagerank(graph.graph, entities)

    passage_scores = defaultdict(float)
    for node, score in pr_scores.items():
        if node.startswith("passage:"):
            passage_scores[node] += score
        elif node.startswith("entity:"):
            for neighbor in graph.graph.neighbors(node):
                if neighbor.startswith("passage:"):
                    passage_scores[neighbor] += score * 0.5

    ranked = sorted(passage_scores.items(), key=lambda x: -x[1])
    results = []
    for passage_id, score in ranked[:top_k]:
        doc_id = passage_id.replace("passage:", "")
        text = graph.get_passage_text(passage_id)
        if text:
            results.append({
                "id": doc_id,
                "passage_id": passage_id,
                "text": text,
                "score": score
            })
    return results


def trace_path(graph: KnowledgeGraph, question: str) -> list[str]:
    entities = extract_query_entities(question)
    path_lines = []
    seen = set()
    for entity in entities:
        node = graph.get_entity_node(entity)
        if node and node not in seen:
            seen.add(node)
            path_lines.append(f"Entity: {entity}")
            for neighbor in graph.graph.neighbors(node):
                edge_data = graph.graph.get_edge_data(node, neighbor)
                if "triple" in edge_data and edge_data["triple"] not in seen:
                    path_lines.append(f"  └─ {edge_data['triple']}")
                    seen.add(edge_data["triple"])
    return path_lines

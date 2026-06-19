import re
import networkx as nx
from collections import defaultdict
from hipporag_style.graph_store import KnowledgeGraph


def _all_graph_entity_names(graph: nx.Graph) -> dict[str, str]:
    return {
        n.replace("entity:", "", 1).lower(): n
        for n in graph.nodes()
        if n.startswith("entity:")
    }


def extract_query_entities(question: str) -> list[str]:
    entities = set()
    for match in re.finditer(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", question):
        entities.add(match.group().strip())
    for match in re.finditer(r"\b([A-Z][a-z]+)\b", question):
        entities.add(match.group(1))
    for match in re.finditer(
        r"\b(pearl necklace|mansion|necklace|coat|safe|gala|pawn shop|alibi|"
        r"theft|figure|foreclosure|job|suspect|motive|evidence|thief|"
        r"timeline|timing|opportunity|financial|crime|garden|photo|"
        r"footage|trunk|soil)\b",
        question,
        re.IGNORECASE,
    ):
        entities.add(match.group(1))
    return list(entities)


def _match_entities_to_graph(graph: nx.Graph, question: str) -> list[str]:
    """Find graph entity nodes matching n-grams from the question."""
    entity_names = _all_graph_entity_names(graph)
    words = question.lower().split()
    matched = set()

    for n in range(min(4, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            for gname_lower, gnode in entity_names.items():
                if phrase == gname_lower or (
                    len(phrase) > 2 and (phrase in gname_lower or gname_lower in phrase)
                ):
                    matched.add(gnode)

    for qe in extract_query_entities(question):
        ql = qe.lower()
        if ql in entity_names:
            matched.add(entity_names[ql])
            continue
        for gname_lower, gnode in entity_names.items():
            if len(ql) > 2 and (ql in gname_lower or gname_lower in ql):
                matched.add(gnode)
                break

    return list(matched)


def personalized_pagerank(
    graph: nx.Graph,
    seed_nodes: list[str],
    damping: float = 0.5,
    max_iter: int = 200,
) -> dict[str, float]:
    if not seed_nodes:
        return {}

    personalization = {node: 0.0 for node in graph.nodes()}
    for node in seed_nodes:
        personalization[node] = 1.0

    total = sum(personalization.values())
    if total == 0:
        return {}
    for k in personalization:
        personalization[k] /= total

    try:
        return nx.pagerank(
            graph,
            alpha=damping,
            personalization=personalization,
            max_iter=max_iter,
            tol=1e-6,
        )
    except:
        return {}


def retrieve_hipporag(
    graph: KnowledgeGraph, question: str, top_k: int = 5
) -> list[dict]:
    seed_nodes = _match_entities_to_graph(graph.graph, question)
    pr_scores = personalized_pagerank(graph.graph, seed_nodes)

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
            results.append(
                {"id": doc_id, "passage_id": passage_id, "text": text, "score": score}
            )
    return results


def trace_path(graph: KnowledgeGraph, question: str) -> list[str]:
    matched_nodes = _match_entities_to_graph(graph.graph, question)
    path_lines = []
    seen = set()
    for node in matched_nodes:
        entity_name = node.replace("entity:", "", 1)
        if node not in seen:
            seen.add(node)
            path_lines.append(f"Entity: {entity_name}")
            for neighbor in graph.graph.neighbors(node):
                edge_data = graph.graph.get_edge_data(node, neighbor)
                if "triple" in edge_data and edge_data["triple"] not in seen:
                    path_lines.append(f"  └─ {edge_data['triple']}")
                    seen.add(edge_data["triple"])
    return path_lines

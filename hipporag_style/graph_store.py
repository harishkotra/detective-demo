import json
import networkx as nx
from collections import defaultdict


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.passage_nodes = {}
        self.entity_to_passages = defaultdict(set)
        self.passage_to_entities = defaultdict(set)

    def add_passage(self, doc_id: str, text: str, title: str):
        node_id = f"passage:{doc_id}"
        self.passage_nodes[node_id] = {"text": text, "title": title}
        self.graph.add_node(node_id, type="passage", doc_id=doc_id, title=title)

    def add_entity(self, name: str):
        node_id = f"entity:{name}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, type="entity", name=name)
        return node_id

    def add_triple(self, subj: str, rel: str, obj: str, doc_id: str):
        subj_id = self.add_entity(subj)
        obj_id = self.add_entity(obj)
        passage_id = f"passage:{doc_id}"

        rel_label = rel.lower().replace(" ", "_")

        self.graph.add_edge(subj_id, obj_id, relation=rel_label, triple=f"({subj}) --[{rel}]-> ({obj})")
        self.graph.add_edge(subj_id, passage_id, relation="mentioned_in")
        self.graph.add_edge(obj_id, passage_id, relation="mentioned_in")

        self.entity_to_passages[subj_id].add(passage_id)
        self.entity_to_passages[obj_id].add(passage_id)
        self.passage_to_entities[passage_id].add(subj_id)
        self.passage_to_entities[passage_id].add(obj_id)

    def build_from_docs(self, docs: list[dict], all_triples: dict[str, list[dict]]):
        for doc in docs:
            self.add_passage(doc["id"], doc["text"], doc["title"])

        for doc_id, triples in all_triples.items():
            for t in triples:
                self.add_triple(t["subject"], t["relation"], t["object"], doc_id)

    def get_entity_node(self, name: str):
        node = f"entity:{name}"
        return node if self.graph.has_node(node) else None

    def num_nodes(self):
        return self.graph.number_of_nodes()

    def num_edges(self):
        return self.graph.number_of_edges()

    def get_connected_passages(self, entity_name: str) -> list[str]:
        node = self.get_entity_node(entity_name)
        if not node:
            return []
        passages = set()
        for neighbor in self.graph.neighbors(node):
            if neighbor.startswith("passage:"):
                passages.add(neighbor)
        return sorted(passages)

    def get_passage_text(self, passage_id: str) -> str:
        return self.passage_nodes.get(passage_id, {}).get("text", "")

    def get_triples_between(self, entity_names: list[str]) -> list[str]:
        triples = []
        for s in entity_names:
            for t in entity_names:
                if s >= t:
                    continue
                s_id = self.get_entity_node(s)
                t_id = self.get_entity_node(t)
                if s_id and t_id and self.graph.has_edge(s_id, t_id):
                    edge_data = self.graph.get_edge_data(s_id, t_id)
                    if "triple" in edge_data:
                        triples.append(edge_data["triple"])
        return triples

    def export_graph_json(self):
        nodes = []
        for n, data in self.graph.nodes(data=True):
            nodes.append({"id": n, **data})
        edges = []
        for s, t, data in self.graph.edges(data=True):
            edges.append({"source": s, "target": t, **data})
        return {"nodes": nodes, "edges": edges}

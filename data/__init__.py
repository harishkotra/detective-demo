import json
import os


def load_docs():
    path = os.path.join(os.path.dirname(__file__), "case_docs.json")
    with open(path) as f:
        return json.load(f)


def load_queries():
    path = os.path.join(os.path.dirname(__file__), "queries.json")
    with open(path) as f:
        return json.load(f)

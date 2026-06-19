import json
import re
from ollama_client import llm_complete

TRIPLE_EXTRACTION_PROMPT = """Extract all subject-relation-object triples from the text below.
Each triple must be a factual statement directly supported by the text.

Return ONLY a JSON array of triples, each with "subject", "relation", and "object" keys.
Use lowercase for entity names unless they are proper nouns.

Example:
Text: "Albert Einstein developed the theory of relativity in 1905."
Output: [{{"subject": "Albert Einstein", "relation": "developed", "object": "theory of relativity"}}, {{"subject": "theory of relativity", "relation": "discovered_in", "object": "1905"}}]

Text: {text}
"""


def extract_triples(text: str, doc_id: str) -> list[dict]:
    prompt = TRIPLE_EXTRACTION_PROMPT.format(text=text)
    raw = llm_complete(prompt, model=None, temperature=0, max_tokens=2048)

    json_match = re.search(r"(\[(?:.|\n)*?\])", raw)
    if not json_match:
        return []
    try:
        triples = json.loads(json_match.group(1))
    except json.JSONDecodeError:
        return []
    for t in triples:
        t["doc_id"] = doc_id
    return triples


ENTITY_EXTRACTION_PROMPT = """Identify the key entities mentioned in this text — people, places, objects, organizations.
Return a JSON array of strings.

Text: {text}
"""


def extract_entities(text: str) -> list[str]:
    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
    raw = llm_complete(prompt, model=None, temperature=0, max_tokens=1024)
    json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not json_match:
        return []
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

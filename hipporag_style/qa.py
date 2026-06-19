from ollama_client import llm_complete


def answer_with_retrieved(docs: list[dict], question: str, graph_trace: list[str] = None) -> str:
    context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    trace_section = ""
    if graph_trace:
        trace_section = "\nKnowledge graph connections:\n" + "\n".join(graph_trace)

    prompt = f"""Answer the question using the provided documents and any knowledge graph connections shown.

Documents:
{context}{trace_section}

Question: {question}

Think step by step about how the facts connect, then give a concise answer:"""
    return llm_complete(prompt, temperature=0, max_tokens=512)

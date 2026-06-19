import streamlit as st
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ollama_client import configure
from data import load_docs, load_queries, load_pre_extracted_triples
from rag.rag_pipeline import index_docs, retrieve as rag_retrieve, answer as rag_answer
from hipporag_style.openie import extract_triples as extract_triples_llm
from hipporag_style.graph_store import KnowledgeGraph
from hipporag_style.retriever import retrieve_hipporag, trace_path
from hipporag_style.qa import answer_with_retrieved as hipporag_answer

st.set_page_config(page_title="HippoRAG vs Standard RAG", layout="wide")
st.title("The Purloined Pearl — Detective's Notebook")
st.markdown(
    "**Standard RAG (ChromaDB + vector similarity)** vs **HippoRAG (Knowledge Graph + PPR)**"
)

# ── Sidebar: Inference Config ──

st.sidebar.header("Inference")

PROVIDER_OPTIONS = {
    "lm_studio": "LM Studio (localhost:1234)",
    "ollama": "Ollama (localhost:11434)",
    "openai": "OpenAI API",
    "openrouter": "OpenRouter",
    "featherless": "Featherless.ai",
}

provider_labels = list(PROVIDER_OPTIONS.values())
provider_keys = list(PROVIDER_OPTIONS.keys())

last_provider = st.session_state.get("_last_provider", "lm_studio")
provider_idx = (
    provider_keys.index(last_provider) if last_provider in provider_keys else 0
)
selected_provider_label = st.sidebar.selectbox(
    "Provider", provider_labels, index=provider_idx, key="provider_sel"
)
selected_provider = provider_keys[provider_labels.index(selected_provider_label)]

needs_api_key = selected_provider in ("openai", "openrouter", "featherless")

default_urls = {
    "lm_studio": "http://127.0.0.1:1234",
    "ollama": "http://127.0.0.1:11434",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "featherless": "https://api.featherless.ai/v1",
}

default_models = {
    "lm_studio": (
        "qwen/qwen3.5-9b",
        "google/gemma-4-e4b",
        "text-embedding-nomic-embed-text-v1.5:2",
    ),
    "ollama": ("llama3.2", "llama3.2:1b", "nomic-embed-text"),
    "openai": ("gpt-4o-mini", "gpt-4o-mini", "text-embedding-3-small"),
    "openrouter": (
        "meta-llama/llama-3.2-3b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "none",
    ),
    "featherless": (
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "none",
    ),
}

dlm, dsm, dem = default_models[selected_provider]

base_url = st.sidebar.text_input(
    "Base URL",
    value=st.session_state.get("_last_base_url", default_urls[selected_provider]),
    key="cfg_base_url",
)
api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    key="cfg_api_key",
    placeholder="sk-…" if needs_api_key else "not needed",
    disabled=not needs_api_key,
)
llm_model = st.sidebar.text_input(
    "LLM model",
    value=st.session_state.get("_last_llm", dlm),
    key="cfg_llm",
)
llm_small = st.sidebar.text_input(
    "Small LLM (extraction)",
    value=st.session_state.get("_last_small", dsm),
    key="cfg_small",
)
embed_model = st.sidebar.text_input(
    "Embedding model",
    value=st.session_state.get("_last_embed", dem),
    key="cfg_embed",
)

st.session_state._last_provider = selected_provider
st.session_state._last_base_url = base_url
st.session_state._last_api_key = api_key
st.session_state._last_llm = llm_model
st.session_state._last_small = llm_small
st.session_state._last_embed = embed_model

st.sidebar.divider()
st.sidebar.markdown(
    '<p style="font-size:0.8rem; opacity:0.7">'
    'Built by <a href="https://harishkotra.me">Harish Kotra</a> · '
    '<a href="https://dailybuild.xyz">More builds</a></p>',
    unsafe_allow_html=True,
)

# ── Data Source ──

st.sidebar.header("Data")
data_source = st.sidebar.radio(
    "Documents",
    ["Sample: Purloined Pearl", "Upload your own"],
    index=0,
    key="data_source",
)

docs = None
queries = None

if data_source == "Sample: Purloined Pearl":
    docs = load_docs()
    queries = load_queries()
elif data_source == "Upload your own":
    uploaded = st.sidebar.file_uploader(
        "Upload docs JSON", type="json", key="docs_upload"
    )
    if uploaded:
        docs = json.load(uploaded)
    q_uploaded = st.sidebar.file_uploader(
        "Upload queries JSON (optional)", type="json", key="queries_upload"
    )
    if q_uploaded:
        queries = json.load(q_uploaded)

if not docs:
    st.info("Upload a documents JSON file to get started.")
    st.stop()

with st.expander("View documents", expanded=False):
    for d in docs:
        st.markdown(f"**{d['id']}** — {d['title']}")
        st.markdown(f"> {d['text']}")

# ── Phase 1: ChromaDB Index (fast) ──

st.header("Phase 1: Vector Index (ChromaDB)")
st.caption("Embeds documents into ChromaDB. Takes ~1 second.")

col1, col2 = st.columns([3, 1])
with col1:
    build_vector = st.button(
        "Build Vector Index", type="primary", use_container_width=True
    )
with col2:
    vector_status = st.empty()

if build_vector or st.session_state.get("vector_built"):
    if not st.session_state.get("vector_built"):
        configure(
            provider=selected_provider,
            base_url=base_url,
            api_key=api_key if needs_api_key else None,
            llm_model=llm_model,
            llm_model_small=llm_small,
            embed_model=None if embed_model == "none" else embed_model,
        )
        with st.spinner("Embedding and indexing documents…"):
            t0 = time.time()
            collection = index_docs(docs)
            elapsed = time.time() - t0
        st.session_state.collection = collection
        st.session_state.vector_built = True
        vector_status.success(f"Indexed {len(docs)} docs in {elapsed:.2f}s")
    else:
        vector_status.info("Vector index is ready.")

# ── Phase 2: Knowledge Graph (slow) ──

st.header("Phase 2: Knowledge Graph (HippoRAG)")
st.caption("Build once, query instantly. Uses pre-extracted triples for this sample.")

triple_source = st.radio(
    "Triple extraction method",
    ["Use pre-extracted triples (instant)", "Extract via LLM (slow)"],
    index=0,
    key="triple_source",
    horizontal=True,
)

col1, col2 = st.columns([3, 1])
with col1:
    build_kg = st.button(
        "Build Knowledge Graph", type="primary", use_container_width=True
    )
with col2:
    kg_status = st.empty()

if build_kg or st.session_state.get("kg_built"):
    if not st.session_state.get("kg_built"):
        configure(
            provider=selected_provider,
            base_url=base_url,
            api_key=api_key if needs_api_key else None,
            llm_model=llm_model,
            llm_model_small=llm_small,
            embed_model=None if embed_model == "none" else embed_model,
        )
        if triple_source.startswith("Use pre-extracted"):
            all_triples = load_pre_extracted_triples()
            status_text = "Loaded pre-extracted triples"
        else:
            all_triples = {}
            progress = st.progress(0, text="Extracting triples via LLM…")
            for i, doc in enumerate(docs):
                progress.progress(
                    (i) / len(docs),
                    text=f"Extracting triples from {doc['id']}…",
                )
                triples = extract_triples_llm(doc["text"], doc["id"])
                all_triples[doc["id"]] = triples
            progress.empty()
            status_text = "Extracted triples via LLM"

        t0 = time.time()
        graph = KnowledgeGraph()
        graph.build_from_docs(docs, all_triples)
        elapsed = time.time() - t0

        st.session_state.graph = graph
        st.session_state.kg_built = True
        kg_status.success(
            f"{status_text}. Graph: {graph.num_nodes()} nodes, "
            f"{graph.num_edges()} edges (built in {elapsed:.2f}s)."
        )
    else:
        kg_status.info("Knowledge Graph is ready.")

# ── Query Section ──

if not st.session_state.get("vector_built"):
    st.info("Build the vector index first to run queries.")
    st.stop()

collection = st.session_state.collection
graph = st.session_state.get("graph")
kg_ready = st.session_state.get("kg_built", False)

st.divider()
st.header("Run Queries")

query_options = ["Type your own…"]
if queries:
    query_options.extend([f"{q['id']}: {q['question'][:70]}" for q in queries])
selected = st.selectbox("Select or type a question:", query_options, key="query_sel")

if selected == "Type your own…":
    question = st.text_input("Enter your question:", key="custom_q")
else:
    question = selected.split(": ", 1)[1] if ": " in selected else selected

expected = ""
if queries and question:
    for q in queries:
        if q["question"] == question:
            expected = q["expected_answer"]
            st.caption(f"Type: {q['type']} | Expected: _{expected}_")
            break

if question and st.button("Run", type="primary", use_container_width=True):
    c1, c2 = st.columns(2, gap="large")

    # ── Standard RAG ──
    with c1:
        st.subheader("Standard RAG")
        t_ret_start = time.time()
        rag_docs = rag_retrieve(collection, question, top_k=3)
        t_ret_end = time.time()
        for d in rag_docs:
            score_str = f"score: {d['score']:.4f}" if d["score"] else ""
            with st.container(border=True):
                st.markdown(f"**{d['id']}** — {score_str}")
                st.markdown(f"_{d['text']}_")
        with st.spinner("Thinking…"):
            t_ans_start = time.time()
            rag_result = rag_answer(rag_docs, question)
            t_ans_end = time.time()
        st.info(
            f"Retrieved: {t_ret_end - t_ret_start:.3f}s | "
            f"Answer: {t_ans_end - t_ans_start:.3f}s"
        )
        st.success(f"**{rag_result}**")

    # ── HippoRAG ──
    with c2:
        st.subheader("HippoRAG")
        if kg_ready:
            t_ret_start = time.time()
            hippo_docs = retrieve_hipporag(graph, question, top_k=3)
            path = trace_path(graph, question)
            t_ret_end = time.time()
            for d in hippo_docs:
                with st.container(border=True):
                    st.markdown(f"**{d['id']}** — score: {d['score']:.6f}")
                    st.markdown(f"_{d['text']}_")
            if not hippo_docs:
                st.warning(
                    "No passages retrieved via PPR — graph entities "
                    "didn't match the query. Try rephrasing."
                )
            if path:
                with st.expander("Graph traversal path"):
                    for line in path:
                        st.text(line)
            with st.spinner("Thinking…"):
                t_ans_start = time.time()
                hippo_result = hipporag_answer(hippo_docs, question, path)
                t_ans_end = time.time()
            st.info(
                f"Retrieved: {t_ret_end - t_ret_start:.3f}s | "
                f"Answer: {t_ans_end - t_ans_start:.3f}s"
            )
            st.success(f"**{hippo_result}**")
        else:
            st.info("Build the Knowledge Graph first.")

    # ── Comparison ──
    if expected:
        st.divider()
        rag_correct = expected.lower() in rag_result.lower()
        hippo_correct = expected.lower() in hippo_result.lower() if kg_ready else False
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Expected", expected)
        col_b.metric("RAG match", "✓" if rag_correct else "✗")
        col_c.metric("HippoRAG match", "✓" if hippo_correct else "✗")

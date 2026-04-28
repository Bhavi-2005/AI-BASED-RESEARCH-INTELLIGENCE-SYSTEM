"""Streamlit UI for the Startup-Grade AI Intelligence Suite.

Advanced analytical tools for research evolution, failure detection, and validated gaps.
"""

from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.tokenize import sent_tokenize

from clustering import cluster_texts
from contradiction import check_contradiction
from data_loader import load_and_chunk_documents
from embedding import DEFAULT_EMBEDDING_MODEL, EmbeddingModel
from fetch_papers import fetch_arxiv_papers
from gap_detector import detect_gaps
from generator import DEFAULT_GENERATION_MODEL, AnswerGenerator
from idea_generator import generate_ideas
from pdf_processor import extract_text_from_bytes, extract_text_from_url
from retriever import Retriever
from vector_store import FaissVectorStore

# -- CONSTANTS --
CACHE_ROOT = Path(".rag_cache")
INDEX_ROOT = CACHE_ROOT / "indexes"
FAILURE_KEYWORDS = ["fails", "limitation", "challenge", "poor performance", "bottleneck", "drawback", "unstable"]

# -- UI CONFIG --
st.set_page_config(
    page_title="Intelligence Suite v3 | Startup Edition",
    page_icon="🚀",
    layout="wide",
)

# Custom Styling (Dark/Light Neutral Hybrid)
st.markdown("""
<style>
    .stApp { background-color: #fafbfc; }
    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #eef2f6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.02);
    }
    .status-badge {
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-fast { background: #dcfce7; color: #166534; }
    .badge-balanced { background: #fef9c3; color: #854d0e; }
    .badge-deep { background: #fee2e2; color: #991b1b; }
    
    .timeline-item {
        border-left: 2px solid #4f46e5;
        margin-left: 20px;
        padding-left: 20px;
        padding-bottom: 20px;
        position: relative;
    }
    .timeline-year { font-weight: 800; color: #4f46e5; }
    
    .strategy-leaf {
        padding: 12px;
        background: #f8fafc;
        border-radius: 8px;
        margin-top: 8px;
        border-left: 3px solid #64748b;
    }
</style>
""", unsafe_allow_html=True)

# -- NLTK DATA ENSURANCE --
def ensure_nltk_resources():
    for res in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)

ensure_nltk_resources()

# -- CORE LOGIC --

@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str) -> EmbeddingModel:
    return EmbeddingModel(model_name=model_name)

@st.cache_resource(show_spinner=False)
def get_answer_generator(model_name: str) -> AnswerGenerator:
    return AnswerGenerator(model_name=model_name)

def extract_timeline_data(sources: list[dict[str, Any]]) -> pd.DataFrame:
    """Attempt to extract years from titles or snippet text."""
    data = []
    for s in sources:
        year_match = re.search(r"\b(20\d{2})\b", s["title"])
        year = int(year_match.group(1)) if year_match else 2024
        data.append({"Title": s["title"], "Year": year})
    return pd.DataFrame(data).sort_values("Year")

def extract_failures(text: str) -> list[str]:
    """Find sentences describing research failures or limitations."""
    sentences = sent_tokenize(text)
    found = []
    for s in sentences:
        if any(k in s.lower() for k in FAILURE_KEYWORDS):
            found.append(s.strip())
    return found[:5]

def extract_methods(text: str) -> list[str]:
    methods = ["Transformer", "BERT", "CNN", "LSTM", "GNN", "GAN", "Diffusion", "Attention", "RL"]
    return [m for m in methods if m.lower() in text.lower()]

def highlight_critical_findings(text: str) -> str:
    """Bold critical research keywords for visual impact."""
    keywords = ["limitation", "failure", "challenge", "unresolved", "bottleneck", "contradiction", "misalignment"]
    highlighted = text
    for k in keywords:
        highlighted = re.sub(rf"({k})", r"**\1**", highlighted, flags=re.IGNORECASE)
    return highlighted

def get_failure_intelligence(text: str) -> dict[str, str]:
    """Categorize a failure with severity and technical fix suggestions."""
    t = text.lower()
    if any(x in t for x in ["train", "dataset", "data", "compute", "gpu"]):
        return {
            "label": "🔴 Training & Data Limitation",
            "severity": "High",
            "fix": "Implement Data Augmentation or use Synthetic Dataset generation (e.g., SD-XL or LLM-based) to resolve sparsity.",
            "type": "data"
        }
    if any(x in t for x in ["alignment", "human", "safety", "bias", "fairness"]):
        return {
            "label": "🟡 Alignment & Safety Issue",
            "severity": "Medium",
            "fix": "Integrate RLHF (Reinforcement Learning from Human Feedback) or DPO (Direct Preference Optimization).",
            "type": "alignment"
        }
    return {
        "label": "🔵 Architectural Bottleneck",
        "severity": "High",
        "fix": "Explore LoRA (Low-Rank Adaptation) or FlashAttention for optimized memory throughput.",
        "type": "perf"
    }

def analyze_contradictions(representatives: list[dict[str, Any]], max_pairs: int) -> list[dict[str, Any]]:
    """Compare representative sentences across themes to identify conflicting findings."""
    results = []
    candidate_pairs = list(combinations(representatives[:8], 2))[:max_pairs]
    
    for left, right in candidate_pairs:
        # check_contradiction returns a list of labels/scores
        prediction = check_contradiction(left["record"]["sentence"], right["record"]["sentence"])
        top_result = sorted(prediction, key=lambda x: x["score"], reverse=True)[0]
        label = str(top_result["label"]).replace("LABEL_", "").title()
        
        # Only report if it's a contradiction or strong entailment conflict
        if label.lower() == "contradiction":
            results.append({
                "left_cluster": left["label"],
                "right_cluster": right["label"],
                "label": label,
                "score": float(top_result["score"]),
                "left_record": left["record"],
                "right_record": right["record"],
            })
    return results

def build_or_load_vector_store(source_items, embedding_model, max_pages):
    # (Simplified for single-call reliability)
    import hashlib, io
    sig = hashlib.sha256(str(source_items).encode()).hexdigest()
    index_dir = INDEX_ROOT / sig
    if (index_dir / "index.faiss").exists(): return FaissVectorStore.load(index_dir)
    
    file_payloads = []
    for item in source_items:
        if "content" in item: file_payloads.append((io.BytesIO(item["content"]), item["title"]))
        elif "pdf" in item:
            with st.spinner(f"Extracting {item['title']}..."):
                text = extract_text_from_url(item["pdf"], max_pages=max_pages)
                file_payloads.append((io.BytesIO(text.encode("utf-8")), f"{item['title']}.txt"))
    
    chunks = load_and_chunk_documents(file_payloads)
    embeddings = embedding_model.embed_texts([c.text for c in chunks])
    vs = FaissVectorStore()
    vs.build(chunks, embeddings)
    vs.save(index_dir)
    return vs

# -- MAIN INTERFACE --

def main():
    st.title("🌐 AI Research Intelligence Suite")
    st.caption("Next-Gen Analysis for Research Gaps, Failures, and Strategic Evolution")

    # 1. Preset Control Levels
    with st.sidebar:
        st.header("🎮 Operation Mode")
        mode = st.radio("Intelligence Depth", ["🟢 Fast Mode", "🟡 Balanced", "🔴 Deep Research Mode"])
        
        # Presets
        if mode.startswith("🟢"):
             max_p, n_cl, pgs = 3, 3, 5
             st.markdown('<span class="status-badge badge-fast">Optimal for Quick Browsing</span>', unsafe_allow_html=True)
        elif mode.startswith("🟡"):
             max_p, n_cl, pgs = 5, 5, 10
             st.markdown('<span class="status-badge badge-balanced">Balanced Depth & Speed</span>', unsafe_allow_html=True)
        else:
             max_p, n_cl, pgs = 10, 8, 20
             st.markdown('<span class="status-badge badge-deep">Exhaustive Domain Mapping</span>', unsafe_allow_html=True)
        
        st.divider()
        st.header("📑 Global Filters")
        f_method = st.multiselect("Methodologies", ["Transformer", "BERT", "CNN", "LSTM", "GNN"])
        f_year = st.slider("Year Range", 2018, 2025, (2020, 2025))

    if "sources" not in st.session_state: st.session_state.sources = []
    
    # 2. Input Pipeline
    srch, upld = st.tabs(["🌎 Global Search", "📂 Batch Upload"])
    with srch:
        query = st.text_input("Define Area of Interest", "Zero-shot Reasoning in Large Language Models")
        if st.button("Initial Intelligence Gathering", type="primary"):
            with st.spinner("Fetching papers..."):
                st.session_state.sources = fetch_arxiv_papers(query, max_results=max_p)
    
    with upld:
        files = st.file_uploader("Internal Papers", type=["pdf", "txt"], accept_multiple_files=True)
        if st.button("Process Batch"):
            if files: st.session_state.sources = [{"title": f.name, "content": f.getvalue()} for f in files]

    if not st.session_state.sources:
        st.info("System Idle. Awaiting Research Domain definition.")
        return

    # 3. Model & Indexing Init
    try:
        emb = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
        vector = build_or_load_vector_store(st.session_state.sources, emb, pgs)
        gen = get_answer_generator(DEFAULT_GENERATION_MODEL)
        ret = Retriever(emb, vector, relevance_threshold=0.25)
    except Exception as e:
        st.error(f"Initialization Failed: {e}")
        return

    # 4. Main Workspace
    t_ev, t_gap, t_qa = st.tabs(["📈 Research Evolution", "🔍 Gap & Strategy Tree", "💬 Interactive Insights"])

    with t_ev:
        st.markdown("### 📊 Methodological Trend Evolution")
        df = extract_timeline_data(st.session_state.sources)
        
        # Filtering logic
        df = df[(df["Year"] >= f_year[0]) & (df["Year"] <= f_year[1])]
        
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            for _, row in df.iterrows():
                st.markdown(f"""
                <div class="timeline-item">
                    <div class="timeline-year">{row['Year']}</div>
                    <div>{row['Title'][:80]}...</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_t2:
            st.markdown("#### 🛡️ Failure Intelligence (Frequency & Fixes)")
            all_failures = []
            for ck in vector.chunks: all_failures.extend(extract_failures(ck.text))
            
            if not all_failures: st.info("No explicit failure patterns detected.")
            else:
                # Group and count
                intel_list = [get_failure_intelligence(f) for f in all_failures]
                from collections import Counter
                counts = Counter([i['label'] for i in intel_list])
                
                for label, count in counts.items():
                    st.markdown(f"**{label}** ({count} Mentions)")
                st.divider()

                for f in sorted(list(set(all_failures)))[:6]:
                    intel = get_failure_intelligence(f)
                    with st.container():
                        st.markdown(f"**{intel['label']}** | **Severity: {intel['severity']}**")
                        st.write(highlight_critical_findings(f))
                        st.info(f"💡 **Suggested Fix:** {intel['fix']}")
                        st.divider()

    with t_gap:
        if st.button("Execute Validation Suite", type="primary"):
            with st.spinner("Calculating confidence metrics..."):
                # Analysis preparation
                records = []
                for ck in vector.chunks:
                    for s in sent_tokenize(ck.text)[:10]: records.append({"title": ck.source, "sentence": s})
                
                texts = [r["sentence"] for r in records]
                embeddings = emb.embed_texts(texts)
                labels = cluster_texts(embeddings, n_clusters=n_cl)
                
                # Logic Execution
                reps = [] # Picking centroids (simplified logic)
                for label in set(labels):
                    idx = [i for i, l in enumerate(labels) if l == label][0]
                    reps.append({"label": label, "record": records[idx]})
                
                contras = analyze_contradictions(reps, 3)
                gaps = detect_gaps(labels, contras, records)
                
                cl_map = {l: [r["sentence"] for i, r in enumerate(records) if labels[i] == l] for l in set(labels)}
                methods = list(set([m for ck in vector.chunks for m in extract_methods(ck.text)]))
                strategies = generate_ideas(gaps, cl_map, methods)

                # 5. Display Validated Gaps (Traceback)
                st.markdown("### 🔥 Validated Research Gaps (Pro Evidence Traceback)")
                for g in gaps:
                    if g.get("is_gap"):
                        with st.expander(f"📌 {g['summary']} | CONFIDENCE: {g['confidence']}", expanded=True):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.write(f"**Reasoning:** {g.get('confidence_reasoning', 'Evidence density')}")
                                st.write(f"**Validation factors:** {', '.join(g['reasons'])}")
                            with c2:
                                st.write("**Evidence Traceback:**")
                                for paper in g["supporting_papers"]: st.caption(f"- {paper}")
                            
                            if g["contradicting_claims"]:
                                st.markdown("##### ⚖️ Contradicting Claims Detected")
                                for cc in g["contradicting_claims"]:
                                    st.warning(f"**{cc['label']}** in finding: '{cc['left_record']['sentence'][:100]}...' vs '{cc['right_record']['sentence'][:100]}...'")

                # 6. Display Strategy Tree
                st.markdown("### 💡 Multi-Idea Strategy Tree")
                for s in strategies:
                    with st.container():
                        st.markdown(f"#### Branch: {s['target']}")
                        st.write(f"**Primary Strategy:** {s['primary_idea']}")
                        
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            st.write("**Alternative Approaches:**")
                            for a in s["alternatives"]: st.markdown(f'<div class="strategy-leaf">{a}</div>', unsafe_allow_html=True)
                        with sc2:
                            st.write("**Incremental Improvements:**")
                            for imp in s["improvements"]: st.markdown(f'<div class="strategy-leaf" style="border-left-color: #4f46e5;">{imp}</div>', unsafe_allow_html=True)
                        
                        st.markdown(f"**Project Impact:** {' | '.join(s['impact_tags'])} (Difficulty: {s['difficulty']})")
                        st.divider()

    with t_qa:
        st.markdown("### 💬 Domain Intelligence Discourse")
        chat_q = st.text_area("Ask for deep methodological synthesis...", height=100)
        if st.button("Synthesize Answer"):
            retrieval = ret.retrieve(chat_q, top_k=5)
            ans = gen.generate(chat_q, retrieval.results[:3], retrieval.is_relevant)
            st.info(ans.answer)
            
if __name__ == "__main__":
    main()

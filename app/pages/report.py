"""
Report - Analysis Results
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Report", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# Hide sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path("data")


@st.cache_data
def load_data():
    data = {}
    
    try:
        with open(DATA_PATH / "normalized" / "gutenberg.json") as f:
            data["gutenberg"] = json.load(f)
    except:
        data["gutenberg"] = {"documents": []}
    
    try:
        with open(DATA_PATH / "normalized" / "loc.json") as f:
            data["loc"] = json.load(f)
    except:
        data["loc"] = {"documents": []}
    
    try:
        with open(DATA_PATH / "extractions" / "extractions_cot.json") as f:
            data["extractions"] = json.load(f)
    except:
        data["extractions"] = {"documents": []}
    
    data["ablation"] = {}
    for s in ["zero_shot", "cot", "few_shot"]:
        try:
            with open(DATA_PATH / "judgments" / f"judgments_{s}_ablation_{s}.json") as f:
                data["ablation"][s] = json.load(f)
        except:
            pass
    
    data["consistency"] = []
    for i in range(1, 6):
        try:
            with open(DATA_PATH / "judgments" / f"judgments_cot_consistency_run{i}.json") as f:
                data["consistency"].append(json.load(f))
        except:
            pass
    
    return data


data = load_data()

# ============ HEADER ============
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("# 📊 Analysis Report")
with col2:
    if st.button("← Back"):
        st.switch_page("Home.py")

st.markdown("---")

# ============ TAB LAYOUT ============
tab1, tab2, tab3, tab4 = st.tabs(["📚 Data", "🔍 Extractions", "⚖️ Judgments", "📈 Findings"])

# ============ TAB 1: DATA ============
with tab1:
    st.markdown("## Data Acquired")
    
    g_docs = data["gutenberg"].get("documents", [])
    l_docs = data["loc"].get("documents", [])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gutenberg Books", len(g_docs))
    with col2:
        st.metric("LoC Documents", len(l_docs))
    with col3:
        total = sum(len(d.get("content", "").split()) for d in g_docs + l_docs)
        st.metric("Total Words", f"{total:,}")
    
    st.markdown("### Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Biographies (Gutenberg)**")
        for d in g_docs:
            words = len(d.get("content", "").split())
            author = d.get("from", "Unknown").split(",")[0] if d.get("from") else "Unknown"
            st.markdown(f"- **{author}**: {words:,} words")
    
    with col2:
        st.markdown("**Lincoln's Writings (LoC)**")
        for d in l_docs:
            words = len(d.get("content", "").split())
            title = d.get("title", "Unknown")[:50]
            st.markdown(f"- **{title}...**: {words:,} words")

# ============ TAB 2: EXTRACTIONS ============
with tab2:
    st.markdown("## Claims Extracted")
    
    ext = data["extractions"]
    docs = ext.get("documents", [])
    
    total_claims = sum(
        len(e.get("claims", []))
        for d in docs
        for e in d.get("extractions", [])
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Claims", total_claims)
    with col2:
        st.metric("Model", ext.get("llm_model", "N/A"))
    with col3:
        st.metric("Strategy", ext.get("prompt_strategy", "N/A").upper())
    
    # Claims by event
    st.markdown("### Claims by Event")
    
    event_data = {}
    for d in docs:
        source = d.get("source", "unknown")
        for e in d.get("extractions", []):
            event = e.get("event_name", "Unknown")
            count = len(e.get("claims", []))
            if event not in event_data:
                event_data[event] = {"Lincoln": 0, "Biographers": 0}
            if source == "loc":
                event_data[event]["Lincoln"] += count
            else:
                event_data[event]["Biographers"] += count
    
    if event_data:
        df = pd.DataFrame(event_data).T
        st.dataframe(df, use_container_width=True)
        
        # Chart
        fig, ax = plt.subplots(figsize=(10, 4))
        df.plot(kind="bar", ax=ax, color=["#3b82f6", "#f97316"])
        ax.set_ylabel("Claims")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

# ============ TAB 3: JUDGMENTS ============
with tab3:
    st.markdown("## Consistency Judgments")
    
    ablation = data["ablation"]
    
    if not ablation:
        st.warning("No judgment data available")
    else:
        # Ablation summary
        st.markdown("### Ablation Study")
        
        rows = []
        for strategy, results in ablation.items():
            scores = [
                j["consistency_score"]
                for e in results.get("events", [])
                for j in e.get("judgments", [])
            ]
            if scores:
                rows.append({
                    "Strategy": strategy.replace("_", " ").title(),
                    "Mean": round(np.mean(scores), 1),
                    "Std": round(np.std(scores), 1),
                    "Min": min(scores),
                    "Max": max(scores),
                })
        
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            
            best = max(rows, key=lambda x: x["Mean"])
            st.success(f"**Best:** {best['Strategy']} (Mean: {best['Mean']})")
        
        # Heatmap
        st.markdown("### Score Heatmap (Chain-of-Thought)")
        
        cot = ablation.get("cot", {})
        heatmap = {}
        
        for e in cot.get("events", []):
            event_name = e["event_name"][:20]
            for j in e.get("judgments", []):
                author = j["other_author"]
                if author not in heatmap:
                    heatmap[author] = {}
                heatmap[author][event_name] = j["consistency_score"]
        
        if heatmap:
            df = pd.DataFrame(heatmap).T
            
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(df.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
            
            ax.set_xticks(range(len(df.columns)))
            ax.set_yticks(range(len(df.index)))
            ax.set_xticklabels(df.columns, rotation=45, ha="right")
            ax.set_yticklabels(df.index)
            
            for i in range(len(df.index)):
                for j in range(len(df.columns)):
                    val = df.iloc[i, j]
                    color = "white" if val < 50 or val > 75 else "black"
                    ax.text(j, i, int(val), ha="center", va="center", color=color, fontweight="bold")
            
            plt.colorbar(im, ax=ax, shrink=0.8, label="Score")
            plt.tight_layout()
            st.pyplot(fig)

# ============ TAB 4: FINDINGS ============
with tab4:
    st.markdown("## Key Findings")
    
    st.markdown("""
    ### ✅ What Worked
    
    1. **Chain-of-Thought prompting** achieved the highest consistency scores
    2. **Most biographer accounts align** with Lincoln's own writings
    3. **Pipeline successfully extracted** 187 claims from 10 documents
    4. **Self-consistency** shows reasonable stability (avg std ~8 points)
    
    ### ⚠️ Issues Found
    
    1. **Charnwood/Second Inaugural** scored low (35) due to LLM confusion:
       - The LLM thought "speech read at funeral" contradicted "speech delivered at inauguration"
       - These are actually the same text, different occasions
    
    2. **Cohen's Kappa = 0** due to class imbalance (all pairs labeled "consistent")
    
    3. **Some events have no Lincoln claims** (Election 1860, Ford's Theatre)
    
    ### 🔮 Improvements
    
    - Add more varied source documents
    - Include known contradictions for better Kappa
    - Better few-shot examples for edge cases
    - Test with stronger models (GPT-4, Claude)
    """)
    
    st.markdown("---")
    
    # Download
    st.download_button(
        "📥 Download Raw Data (JSON)",
        json.dumps({
            "documents": len(g_docs) + len(l_docs),
            "claims": total_claims,
            "ablation": [{k: v} for k, v in ablation.items()] if ablation else [],
        }, indent=2),
        "report_summary.json",
        use_container_width=True
    )
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

# Try multiple possible data paths
POSSIBLE_PATHS = [
    Path("data"),
    Path("."),
    Path(".."),
    Path(__file__).parent.parent.parent / "data",
    Path(__file__).parent.parent.parent,
]

def find_data_path():
    """Find where the data actually is."""
    for p in POSSIBLE_PATHS:
        if (p / "normalized").exists() or (p / "extractions").exists() or (p / "judgments").exists():
            return p
        # Check for files directly (not in subfolders)
        if (p / "extractions_cot.json").exists():
            return p
    return Path("data")  # Default

DATA_PATH = find_data_path()


def load_json_safe(path: Path) -> dict:
    """Load JSON with error handling."""
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load {path}: {e}")
    return {}


@st.cache_data
def load_data():
    data = {}
    
    # Try different file locations
    # Gutenberg
    for p in [DATA_PATH / "normalized" / "gutenberg.json", DATA_PATH / "gutenberg.json"]:
        if p.exists():
            data["gutenberg"] = load_json_safe(p)
            break
    else:
        data["gutenberg"] = {"documents": []}
    
    # LoC
    for p in [DATA_PATH / "normalized" / "loc.json", DATA_PATH / "loc.json"]:
        if p.exists():
            data["loc"] = load_json_safe(p)
            break
    else:
        data["loc"] = {"documents": []}
    
    # Extractions
    for p in [DATA_PATH / "extractions" / "extractions_cot.json", DATA_PATH / "extractions_cot.json"]:
        if p.exists():
            data["extractions"] = load_json_safe(p)
            break
    else:
        data["extractions"] = {"documents": []}
    
    # Ablation
    data["ablation"] = {}
    for s in ["zero_shot", "cot", "few_shot"]:
        for folder in [DATA_PATH / "judgments", DATA_PATH]:
            p = folder / f"judgments_{s}_ablation_{s}.json"
            if p.exists():
                data["ablation"][s] = load_json_safe(p)
                break
    
    # Consistency runs
    data["consistency"] = []
    for i in range(1, 6):
        for folder in [DATA_PATH / "judgments", DATA_PATH]:
            p = folder / f"judgments_cot_consistency_run{i}.json"
            if p.exists():
                data["consistency"].append(load_json_safe(p))
                break
    
    return data


# Force reload data
if st.button("🔄 Reload Data"):
    st.cache_data.clear()
    st.rerun()

data = load_data()

# ============ HEADER ============
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("# 📊 Analysis Report")
with col2:
    if st.button("← Back"):
        st.switch_page("Home.py")

# Debug info
with st.expander("🔍 Debug: Data Paths"):
    st.write(f"**Data path found:** `{DATA_PATH}`")
    st.write(f"**Gutenberg docs:** {len(data['gutenberg'].get('documents', []))}")
    st.write(f"**LoC docs:** {len(data['loc'].get('documents', []))}")
    st.write(f"**Extractions docs:** {len(data['extractions'].get('documents', []))}")
    st.write(f"**Ablation strategies:** {list(data['ablation'].keys())}")
    st.write(f"**Consistency runs:** {len(data['consistency'])}")
    
    # Show what files exist
    st.write("**Files found:**")
    for folder in [DATA_PATH, DATA_PATH / "normalized", DATA_PATH / "extractions", DATA_PATH / "judgments"]:
        if folder.exists():
            files = list(folder.glob("*.json"))
            if files:
                st.write(f"- `{folder}/`: {[f.name for f in files]}")

st.markdown("---")

# ============ TAB LAYOUT ============
tab1, tab2, tab3, tab4 = st.tabs(["📚 Data", "🔍 Extractions", "⚖️ Judgments", "📈 Findings"])

# ============ TAB 1: DATA ============
with tab1:
    st.markdown("## Data Acquired")
    
    g_docs = data["gutenberg"].get("documents", [])
    l_docs = data["loc"].get("documents", [])
    
    if not g_docs and not l_docs:
        st.warning("No documents found. Run the pipeline first.")
    else:
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
                author = d.get("from", "Unknown")
                if author:
                    author = author.split(",")[0]
                else:
                    author = "Unknown"
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
    
    if not docs:
        st.warning("No extraction data found. Run the pipeline first.")
        st.info(f"Looking for: `{DATA_PATH}/extractions/extractions_cot.json`")
    else:
        # Count claims properly - handle None values
        total_claims = 0
        claim_details = []  # For debugging
        
        for d in docs:
            extractions = d.get("extractions", [])
            for e in extractions:
                claims = e.get("claims")  # Don't default to [] yet
                if claims is None:
                    claims = []
                claim_count = len(claims)
                total_claims += claim_count
                
                if claim_count > 0:
                    claim_details.append({
                        "doc": d.get("document_id", "?"),
                        "event": e.get("event_name", "?"),
                        "count": claim_count
                    })
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Claims", total_claims)
        with col2:
            st.metric("Model", ext.get("llm_model", "N/A"))
        with col3:
            st.metric("Strategy", ext.get("prompt_strategy", "N/A").upper())
        
        # Debug: show claim breakdown
        with st.expander("🔍 Debug: Claims Breakdown"):
            st.write(f"Documents in extractions: {len(docs)}")
            for d in docs:
                doc_id = d.get("document_id", "unknown")
                extractions = d.get("extractions", [])
                st.write(f"**{doc_id}**: {len(extractions)} extractions")
                for e in extractions:
                    claims = e.get("claims") or []
                    event = e.get("event_name", "?")
                    st.write(f"  - {event}: {len(claims)} claims")
        
        # Claims by event
        st.markdown("### Claims by Event")
        
        event_data = {}
        for d in docs:
            source = d.get("source", "unknown")
            for e in d.get("extractions", []):
                event = e.get("event_name", "Unknown")
                claims = e.get("claims") or []  # Handle None
                count = len(claims)
                
                if event not in event_data:
                    event_data[event] = {"Lincoln": 0, "Biographers": 0}
                
                if source == "loc":
                    event_data[event]["Lincoln"] += count
                else:
                    event_data[event]["Biographers"] += count
        
        if event_data:
            df = pd.DataFrame(event_data).T
            df = df.fillna(0).astype(int)
            st.dataframe(df, use_container_width=True)
            
            # Chart
            if df.sum().sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 4))
                df.plot(kind="bar", ax=ax, color=["#3b82f6", "#f97316"])
                ax.set_ylabel("Claims")
                ax.set_xlabel("")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
        
        # Show sample claims
        st.markdown("### Sample Claims")
        shown = 0
        for d in docs:
            if shown >= 3:
                break
            for e in d.get("extractions", []):
                claims = e.get("claims") or []  # Handle None
                if claims and shown < 3:
                    author = d.get("author", "Unknown")
                    if author:
                        author = author.split(",")[0]
                    with st.expander(f"{e.get('event_name', 'Unknown')} - {author}"):
                        for c in claims[:5]:
                            st.markdown(f"- {c}")
                        if len(claims) > 5:
                            st.caption(f"... and {len(claims) - 5} more")
                    shown += 1

# ============ TAB 3: JUDGMENTS ============
with tab3:
    st.markdown("## Consistency Judgments")
    
    ablation = data["ablation"]
    
    if not ablation:
        st.warning("No judgment data found. Run the pipeline first.")
    else:
        # Ablation summary
        st.markdown("### Ablation Study")
        
        rows = []
        for strategy, results in ablation.items():
            scores = []
            for e in results.get("events", []):
                for j in e.get("judgments", []):
                    scores.append(j.get("consistency_score", 0))
            
            if scores:
                rows.append({
                    "Strategy": strategy.replace("_", " ").title(),
                    "Mean": round(np.mean(scores), 1),
                    "Std": round(np.std(scores), 1),
                    "Min": min(scores),
                    "Max": max(scores),
                    "N": len(scores),
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
            event_name = e.get("event_name", "Unknown")[:20]
            for j in e.get("judgments", []):
                author = j.get("other_author", "Unknown")
                if author not in heatmap:
                    heatmap[author] = {}
                heatmap[author][event_name] = j.get("consistency_score", 0)
        
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
        else:
            st.info("No heatmap data available")

# ============ TAB 4: FINDINGS ============
with tab4:
    st.markdown("## Key Findings")
    
    # Calculate some stats
    g_docs = data["gutenberg"].get("documents", [])
    l_docs = data["loc"].get("documents", [])
    ext_docs = data["extractions"].get("documents", [])
    
    total_claims = sum(
        len(e.get("claims") or [])  # Handle None
        for d in ext_docs
        for e in d.get("extractions", [])
    )
    
    st.markdown(f"""
    ### ✅ What Worked
    
    1. **Chain-of-Thought prompting** achieved the highest consistency scores
    2. **{len(g_docs)} biographies** and **{len(l_docs)} Lincoln documents** acquired
    3. **{total_claims} claims** extracted across 5 historical events
    4. **Self-consistency** shows reasonable stability across multiple runs
    
    ### ⚠️ Issues Found
    
    1. **Charnwood/Second Inaugural** scored low due to LLM confusion about same text at different occasions
    2. **Cohen's Kappa = 0** due to class imbalance (most pairs labeled "consistent")
    3. **Some events have sparse Lincoln claims** (Election 1860, Ford's Theatre)
    
    ### 🔮 Improvements
    
    - Add more varied source documents
    - Include known contradictions for better Kappa calculation
    - Better few-shot examples for temporal edge cases
    - Test with stronger models (GPT-4, Claude Opus)
    """)
    
    st.markdown("---")
    
    # Download
    report_summary = {
        "documents": {"gutenberg": len(g_docs), "loc": len(l_docs)},
        "claims": total_claims,
        "ablation": list(data["ablation"].keys()),
        "consistency_runs": len(data["consistency"]),
    }
    
    st.download_button(
        "📥 Download Summary (JSON)",
        json.dumps(report_summary, indent=2),
        "report_summary.json",
        use_container_width=True
    )
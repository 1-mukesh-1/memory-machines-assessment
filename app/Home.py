"""
Lincoln Analysis Pipeline
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Lincoln Pipeline",
    page_icon="🎩",
    layout="centered",
    initial_sidebar_state="collapsed",  # Hide sidebar
)

# Hide sidebar completely
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    
    .task-card {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        height: 120px;
    }
    .task-card.done {
        background: #ecfdf5;
        border-color: #10b981;
    }
    .task-card.pending {
        background: #f8fafc;
        border-color: #cbd5e1;
    }
    .task-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .task-desc {
        font-size: 0.8rem;
        color: #64748b;
    }
    .arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state properly
if "force_rerun" not in st.session_state:
    st.session_state.force_rerun = False
if "logs" not in st.session_state:
    st.session_state.logs = {"part1": [], "part2": [], "part3": [], "report": []}

DATA_PATH = Path("data")


def check_status(task_id):
    if task_id == "part1":
        return (DATA_PATH / "normalized" / "gutenberg.json").exists()
    elif task_id == "part2":
        return (DATA_PATH / "extractions" / "extractions_cot.json").exists()
    elif task_id == "part3":
        return (DATA_PATH / "judgments").exists() and len(list((DATA_PATH / "judgments").glob("*.json"))) > 0
    elif task_id == "report":
        return check_status("part3")
    return False


# ============ HEADER ============
st.markdown("# 🎩 Lincoln Analysis")
st.markdown("*Comparing Lincoln's writings with biographer accounts*")

st.markdown("---")

# ============ PIPELINE STATUS ============
tasks = [
    ("part1", "Acquire", "Scrape data"),
    ("part2", "Extract", "LLM claims"),
    ("part3", "Judge", "Compare"),
    ("report", "Report", "Results"),
]

# Status row
cols = st.columns([2, 1, 2, 1, 2, 1, 2])

for i, (task_id, title, desc) in enumerate(tasks):
    col_idx = i * 2
    done = check_status(task_id)
    
    with cols[col_idx]:
        icon = "✅" if done else "⬜"
        st.markdown(f"""
        <div class="task-card {'done' if done else 'pending'}">
            <div style="font-size: 1.5rem;">{icon}</div>
            <div class="task-title">{title}</div>
            <div class="task-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Arrow between tasks
    if i < len(tasks) - 1:
        with cols[col_idx + 1]:
            st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

st.markdown("---")

# ============ ACTIONS ============
st.markdown("### Actions")

col1, col2 = st.columns(2)

with col1:
    run_clicked = st.button(
        "▶️ Run Pipeline", 
        type="primary", 
        use_container_width=True,
        help="Run pipeline (skips completed tasks)"
    )
    if run_clicked:
        st.session_state.force_rerun = False
        st.switch_page("pages/run_pipeline.py")

with col2:
    rerun_clicked = st.button(
        "🔄 Re-run All", 
        type="secondary", 
        use_container_width=True,
        help="Delete existing data and re-run everything"
    )
    if rerun_clicked:
        st.session_state.force_rerun = True
        st.switch_page("pages/run_pipeline.py")

# Report button (only if complete)
all_done = all(check_status(t[0]) for t in tasks)
if all_done:
    st.markdown("")
    if st.button("📊 View Report", use_container_width=True):
        st.switch_page("pages/report.py")
else:
    st.info("Complete the pipeline to view the report.")

st.markdown("---")

# ============ STATUS SUMMARY ============
st.markdown("### Current Data")

c1, c2, c3 = st.columns(3)

with c1:
    if check_status("part1"):
        try:
            import json
            with open(DATA_PATH / "normalized" / "gutenberg.json") as f:
                g_count = len(json.load(f).get("documents", []))
            with open(DATA_PATH / "normalized" / "loc.json") as f:
                l_count = len(json.load(f).get("documents", []))
            st.metric("Documents", f"{g_count + l_count}")
        except:
            st.metric("Documents", "✓")
    else:
        st.metric("Documents", "—")

with c2:
    if check_status("part2"):
        try:
            import json
            with open(DATA_PATH / "extractions" / "extractions_cot.json") as f:
                ext = json.load(f)
                claims = sum(len(e.get("claims", [])) for d in ext.get("documents", []) for e in d.get("extractions", []))
            st.metric("Claims", f"{claims}")
        except:
            st.metric("Claims", "✓")
    else:
        st.metric("Claims", "—")

with c3:
    if check_status("part3"):
        j_count = len(list((DATA_PATH / "judgments").glob("*.json")))
        st.metric("Judgments", f"{j_count} runs")
    else:
        st.metric("Judgments", "—")
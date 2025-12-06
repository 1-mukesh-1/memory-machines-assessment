"""
Pipeline Runner
"""

import streamlit as st
import asyncio
import shutil
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Running...", page_icon="🔄", layout="centered", initial_sidebar_state="collapsed")

# Hide sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path("data")

# Get force_rerun flag (default to False if not set)
FORCE_RERUN = st.session_state.get("force_rerun", False)

# Initialize fresh logs for this run
st.session_state.run_logs = []


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.run_logs.append(f"[{ts}] {msg}")


def clear_all_data():
    """Clear all pipeline data at once."""
    cleared = []
    
    if (DATA_PATH / "normalized").exists():
        shutil.rmtree(DATA_PATH / "normalized")
        cleared.append("normalized")
    
    if (DATA_PATH / "extractions").exists():
        shutil.rmtree(DATA_PATH / "extractions")
        cleared.append("extractions")
    
    if (DATA_PATH / "judgments").exists():
        shutil.rmtree(DATA_PATH / "judgments")
        cleared.append("judgments")
    
    # Also clear raw data if exists
    if (DATA_PATH / "raw").exists():
        shutil.rmtree(DATA_PATH / "raw")
        cleared.append("raw")
    
    return cleared


def data_exists(task: str) -> bool:
    if task == "part1":
        return (DATA_PATH / "normalized" / "gutenberg.json").exists()
    elif task == "part2":
        return (DATA_PATH / "extractions" / "extractions_cot.json").exists()
    elif task == "part3":
        return (DATA_PATH / "judgments").exists() and len(list((DATA_PATH / "judgments").glob("*.json"))) > 0
    return False


# ============ HEADER ============
st.markdown("# 🔄 Running Pipeline")

if FORCE_RERUN:
    st.warning("**Re-run mode:** Deleting all existing data...")
    cleared = clear_all_data()
    if cleared:
        log(f"Cleared: {', '.join(cleared)}")
    else:
        log("No existing data to clear")

if st.button("← Cancel"):
    st.switch_page("Home.py")

st.markdown("---")

# ============ PROGRESS ============
progress = st.progress(0)
status_text = st.empty()
log_area = st.empty()


def update_display():
    log_area.code("\n".join(st.session_state.run_logs[-20:]))  # Last 20 lines


# ============ TASK 1: DATA ACQUISITION ============
status_text.markdown("### ⏳ Task 1/3: Data Acquisition")
progress.progress(10)

if data_exists("part1"):
    log("Task 1: Data exists, skipping")
else:
    log("Task 1: Starting data acquisition...")
    update_display()
    
    try:
        from src.part1_acquisition.run import run as run_part1
        from src.part1_acquisition.config import Part1Config
        
        log("Task 1: Fetching Gutenberg books...")
        update_display()
        
        asyncio.run(run_part1(Part1Config()))
        log("Task 1: ✓ Complete")
    except Exception as e:
        log(f"Task 1: ✗ FAILED - {e}")
        update_display()
        st.error(f"Pipeline failed at Task 1: {e}")
        st.stop()

update_display()
progress.progress(33)

# ============ TASK 2: EXTRACTION ============
status_text.markdown("### ⏳ Task 2/3: Event Extraction")

if data_exists("part2"):
    log("Task 2: Extractions exist, skipping")
else:
    log("Task 2: Starting extraction (this may take a few minutes)...")
    update_display()
    
    try:
        from src.part2_extraction.run import run as run_part2
        from src.part2_extraction.config import Part2Config
        
        log("Task 2: Running LLM extraction...")
        update_display()
        
        asyncio.run(run_part2(Part2Config(prompt_strategy="cot")))
        log("Task 2: ✓ Complete")
    except Exception as e:
        log(f"Task 2: ✗ FAILED - {e}")
        update_display()
        st.error(f"Pipeline failed at Task 2: {e}")
        st.stop()

update_display()
progress.progress(66)

# ============ TASK 3: JUDGE ============
status_text.markdown("### ⏳ Task 3/3: LLM Judge")

if data_exists("part3"):
    log("Task 3: Judgments exist, skipping")
else:
    log("Task 3: Running judge experiments...")
    update_display()
    
    try:
        from src.part3_judge.run import run_experiments
        
        log("Task 3: Ablation study + self-consistency...")
        update_display()
        
        asyncio.run(run_experiments())
        log("Task 3: ✓ Complete")
    except Exception as e:
        log(f"Task 3: ✗ FAILED - {e}")
        update_display()
        st.error(f"Pipeline failed at Task 3: {e}")
        st.stop()

update_display()
progress.progress(100)

# ============ DONE ============
status_text.markdown("### ✅ Pipeline Complete!")
log("Done!")
update_display()

# Reset force_rerun flag
st.session_state.force_rerun = False

st.balloons()

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    if st.button("📊 View Report", type="primary", use_container_width=True):
        st.switch_page("pages/report.py")
with col2:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")
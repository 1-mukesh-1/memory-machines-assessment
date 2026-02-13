"""
RAG Q&A Page - Ask questions about Lincoln with source citations.

Streamlit page that uses the LangChain RAG pipeline to answer
natural language questions about Lincoln's writings and biographies.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Ask Lincoln",
    page_icon="❓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_qa_chain():
    """Initialize the Q&A chain (cached across reruns)."""
    from src.rag.qa_chain import LincolnQAChain
    return LincolnQAChain()


# ============ HEADER ============
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("# ❓ Ask Lincoln")
    st.markdown("*Ask questions about Lincoln's writings and biographies — powered by RAG*")
with col2:
    if st.button("← Home"):
        st.switch_page("Home.py")

st.markdown("---")

# ============ MODE SELECTION ============
mode = st.radio(
    "Query mode",
    ["📚 All Sources", "📜 Lincoln's Writings Only", "📖 Biographies Only", "⚖️ Compare Sources"],
    horizontal=True,
    label_visibility="collapsed",
)

# ============ QUERY INPUT ============
query = st.text_input(
    "Your question",
    placeholder="e.g., What did Lincoln say about slavery's role in the Civil War?",
)

# Suggestions
st.markdown("**Try these:**")
suggestions = [
    "What was Lincoln's view on slavery in the Second Inaugural?",
    "How did Lincoln handle the Fort Sumter crisis?",
    "What was the main message of the Gettysburg Address?",
    "How do biographers differ on Lincoln's motivations at Fort Sumter?",
    "Did Lincoln believe the war was divine punishment?",
]
cols = st.columns(len(suggestions))
for i, suggestion in enumerate(suggestions):
    with cols[i]:
        if st.button(suggestion[:30] + "...", key=f"sug_{i}", use_container_width=True):
            query = suggestion

# ============ RUN QUERY ============
if query:
    with st.spinner(f"Searching documents and generating answer..."):
        try:
            qa = get_qa_chain()

            if "Compare" in mode:
                result = qa.compare(query)
            elif "Lincoln's Writings" in mode:
                result = qa.ask_lincoln_only(query)
            elif "Biographies" in mode:
                result = qa.ask_biographers_only(query)
            else:
                result = qa.ask(query)

            # ============ ANSWER ============
            st.markdown("### Answer")
            st.markdown(result["answer"])

            # ============ SOURCES ============
            st.markdown("---")
            st.markdown("### Sources Retrieved")

            for src in result["sources"]:
                source_type = "📜 Lincoln (LoC)" if src["source"] == "loc" else "📖 Biographer (Gutenberg)"
                with st.expander(f"{source_type} — {src['title'][:60]}..."):
                    st.markdown(f"**Document ID:** `{src['doc_id']}`")
                    st.markdown(f"**Author:** {src['author']}")
                    if src.get("date"):
                        st.markdown(f"**Date:** {src['date']}")
                    st.markdown("**Relevant passage:**")
                    st.text(src["chunk_preview"])

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure you've run `python scripts/ingest_vectorstore.py` first.")

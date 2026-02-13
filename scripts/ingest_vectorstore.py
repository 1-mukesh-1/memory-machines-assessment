"""
Ingest Lincoln Documents into Pinecone Vector Store

One-time script to embed and upsert all Lincoln documents
(LoC primary sources + Gutenberg biographies) into Pinecone for RAG.

Usage:
    python scripts/ingest_vectorstore.py
    python scripts/ingest_vectorstore.py --force   # Re-ingest even if index exists

Requires:
    OPENAI_API_KEY and PINECONE_API_KEY in .env
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    from src.rag.vectorstore import VectorStoreManager

    logger.info("=" * 60)
    logger.info("RAG: VECTOR STORE INGESTION")
    logger.info("=" * 60)

    # Validate env
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set in .env")
        sys.exit(1)
    if not os.getenv("PINECONE_API_KEY"):
        logger.error("PINECONE_API_KEY not set in .env")
        sys.exit(1)

    manager = VectorStoreManager()

    # Check if already ingested
    force = "--force" in sys.argv
    try:
        stats = manager.index_stats()
        count = stats.get("total_vector_count", 0)
        if count > 0 and not force:
            logger.info(f"Index already has {count} vectors. Use --force to re-ingest.")
            return
    except Exception:
        pass  # Index doesn't exist yet, will be created

    # Run ingestion
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    count = manager.ingest_all(data_dir)

    logger.info(f"\nDone! Ingested {count} chunks into Pinecone.")
    logger.info("You can now use the RAG Q&A feature.")


if __name__ == "__main__":
    main()

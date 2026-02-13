"""
Vector Store - Pinecone setup, document chunking, and embedding ingestion.

Embeds all Lincoln documents (LoC + Gutenberg) into Pinecone for RAG retrieval.
Each vector stores the chunk text, source metadata, and document provenance.

Usage:
    from src.rag.vectorstore import VectorStoreManager
    manager = VectorStoreManager()
    manager.ingest_all()                    # One-time setup
    docs = manager.similarity_search(query) # Query time
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

# -----------------------------------------------
# Config
# -----------------------------------------------
PINECONE_INDEX_NAME = os.getenv("LINCOLN_PINECONE_INDEX", "lincoln-rag")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


class VectorStoreManager:
    """
    Manages the Pinecone vector store for Lincoln documents.
    
    Handles:
    - Index creation (one-time)
    - Document chunking and embedding ingestion
    - Similarity search for RAG retrieval
    """

    def __init__(
        self,
        index_name: str = None,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.index_name = index_name or PINECONE_INDEX_NAME
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._vectorstore: Optional[PineconeVectorStore] = None

    # -----------------------------------------------
    # Index management
    # -----------------------------------------------
    def ensure_index(self) -> None:
        """Create Pinecone index if it doesn't exist."""
        existing = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing:
            logger.info(f"Creating Pinecone index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
                ),
            )
            # Wait for ready
            import time
            for _ in range(30):
                desc = self.pc.describe_index(self.index_name)
                if desc.status.get("ready", False):
                    break
                time.sleep(2)
            logger.info(f"Index '{self.index_name}' ready")
        else:
            logger.info(f"Index '{self.index_name}' already exists")

    def get_vectorstore(self) -> PineconeVectorStore:
        """Get or create the PineconeVectorStore instance."""
        if self._vectorstore is None:
            self._vectorstore = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
            )
        return self._vectorstore

    def index_stats(self) -> dict:
        """Return current index statistics."""
        index = self.pc.Index(self.index_name)
        return index.describe_index_stats()

    # -----------------------------------------------
    # Document loading
    # -----------------------------------------------
    def load_documents(self, data_dir: str = "data") -> list[Document]:
        """
        Load all Lincoln documents from normalized JSON files
        and split into LangChain Document chunks with metadata.
        """
        documents = []
        data_path = Path(data_dir)

        # Load LoC documents
        loc_path = data_path / "normalized" / "loc.json"
        if loc_path.exists():
            with open(loc_path) as f:
                loc_data = json.load(f)
            for doc in loc_data.get("documents", []):
                documents.extend(self._doc_to_chunks(doc, source="loc"))
            logger.info(f"Loaded {len(loc_data.get('documents', []))} LoC documents")

        # Load Gutenberg documents
        gutenberg_path = data_path / "normalized" / "gutenberg.json"
        if gutenberg_path.exists():
            with open(gutenberg_path) as f:
                gutenberg_data = json.load(f)
            for doc in gutenberg_data.get("documents", []):
                documents.extend(self._doc_to_chunks(doc, source="gutenberg"))
            logger.info(
                f"Loaded {len(gutenberg_data.get('documents', []))} Gutenberg documents"
            )

        logger.info(f"Total chunks created: {len(documents)}")
        return documents

    def _doc_to_chunks(self, doc: dict, source: str) -> list[Document]:
        """Split a single document into LangChain Document chunks."""
        content = doc.get("content", "")
        if not content.strip():
            return []

        metadata = {
            "doc_id": doc.get("id", "unknown"),
            "title": doc.get("title", "Unknown")[:200],
            "source": source,
            "document_type": doc.get("document_type", "Unknown"),
            "date": doc.get("date", ""),
            "author": doc.get("from", "Unknown") or "Unknown",
        }

        chunks = self.splitter.split_text(content)
        return [
            Document(
                page_content=chunk,
                metadata={**metadata, "chunk_index": i},
            )
            for i, chunk in enumerate(chunks)
        ]

    # -----------------------------------------------
    # Ingestion
    # -----------------------------------------------
    def ingest_all(self, data_dir: str = "data", batch_size: int = 100) -> int:
        """
        Load all documents, embed, and upsert into Pinecone.
        
        Returns:
            Number of chunks ingested
        """
        self.ensure_index()
        documents = self.load_documents(data_dir)

        if not documents:
            logger.warning("No documents found to ingest")
            return 0

        logger.info(f"Ingesting {len(documents)} chunks into Pinecone...")
        vectorstore = self.get_vectorstore()

        # Batch upsert
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            vectorstore.add_documents(batch)
            logger.info(f"  Upserted batch {i // batch_size + 1}")

        stats = self.index_stats()
        logger.info(
            f"Ingestion complete. Total vectors: {stats.get('total_vector_count', 'N/A')}"
        )
        return len(documents)

    # -----------------------------------------------
    # Search
    # -----------------------------------------------
    def similarity_search(
        self, query: str, k: int = 5, filter: dict = None
    ) -> list[Document]:
        """
        Retrieve the most relevant document chunks for a query.
        
        Args:
            query: Natural language question
            k: Number of chunks to retrieve
            filter: Optional Pinecone metadata filter
                e.g. {"source": "loc"} for only Lincoln's own writings
        
        Returns:
            List of LangChain Documents with content and metadata
        """
        vectorstore = self.get_vectorstore()
        kwargs = {"k": k}
        if filter:
            kwargs["filter"] = filter
        return vectorstore.similarity_search(query, **kwargs)

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """Search with relevance scores."""
        vectorstore = self.get_vectorstore()
        return vectorstore.similarity_search_with_score(query, k=k)

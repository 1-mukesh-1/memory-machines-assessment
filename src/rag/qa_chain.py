"""
Q&A Chain - LangChain RAG pipeline for answering questions about Lincoln.

Retrieves relevant document chunks from Pinecone, then generates
an answer with source citations using OpenAI via LangChain.

Usage:
    from src.rag.qa_chain import LincolnQAChain
    qa = LincolnQAChain()
    result = qa.ask("What did Lincoln say about slavery in the Second Inaugural?")
    print(result["answer"])
    print(result["sources"])
"""

import os
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks import StdOutCallbackHandler

from src.rag.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)

# -----------------------------------------------
# Prompts
# -----------------------------------------------
QA_SYSTEM_TEMPLATE = """You are a historian specializing in Abraham Lincoln. 
Answer questions using ONLY the provided source documents. 

Rules:
1. Base your answer strictly on the retrieved passages below.
2. Cite which source each claim comes from (e.g., "According to Ketcham's biography..." 
   or "In Lincoln's letter to Truman Smith...").
3. If the sources contain conflicting accounts, note the disagreement.
4. If the sources don't contain enough information, say so honestly.
5. Distinguish between Lincoln's own words (Library of Congress sources) and 
   biographer interpretations (Gutenberg sources).

SOURCES:
{context}

QUESTION: {question}

Provide a thorough, well-cited answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_SYSTEM_TEMPLATE,
    input_variables=["context", "question"],
)

COMPARISON_TEMPLATE = """You are a historian comparing different accounts of Abraham Lincoln.

Given the following source passages about the same topic, analyze:
1. Where do the accounts AGREE?
2. Where do they DIFFER (factual disagreements, interpretive differences, omissions)?
3. Which source appears most reliable and why?

Always distinguish between Lincoln's own writings (LoC sources) and biographer accounts (Gutenberg).

SOURCES:
{context}

QUESTION: {question}

Provide a detailed comparative analysis:"""

COMPARISON_PROMPT = PromptTemplate(
    template=COMPARISON_TEMPLATE,
    input_variables=["context", "question"],
)


class LincolnQAChain:
    """
    RAG Q&A chain for Lincoln documents.
    
    Supports two modes:
    - ask(): General Q&A with source citations
    - compare(): Cross-source comparison analysis
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        retrieval_k: int = 6,
        vectorstore_manager: Optional[VectorStoreManager] = None,
    ):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.retrieval_k = retrieval_k
        self.manager = vectorstore_manager or VectorStoreManager()
        self._qa_chain = None
        self._compare_chain = None

    def _get_retriever(self, source_filter: str = None):
        """Get a LangChain retriever with optional source filtering."""
        vectorstore = self.manager.get_vectorstore()
        search_kwargs = {"k": self.retrieval_k}
        if source_filter:
            search_kwargs["filter"] = {"source": source_filter}
        return vectorstore.as_retriever(search_kwargs=search_kwargs)

    def _build_qa_chain(self, source_filter: str = None) -> RetrievalQA:
        """Build the Q&A chain."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self._get_retriever(source_filter),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT},
        )

    def _build_compare_chain(self) -> RetrievalQA:
        """Build the comparison chain (retrieves from all sources)."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self._get_retriever(),  # No filter — all sources
            return_source_documents=True,
            chain_type_kwargs={"prompt": COMPARISON_PROMPT},
        )

    # -----------------------------------------------
    # Public API
    # -----------------------------------------------
    def ask(
        self,
        question: str,
        source_filter: str = None,
        verbose: bool = False,
    ) -> dict:
        """
        Ask a question about Lincoln with RAG retrieval.
        
        Args:
            question: Natural language question
            source_filter: Optional - "loc" for Lincoln's writings only,
                          "gutenberg" for biographies only, None for all
            verbose: Print chain execution details
        
        Returns:
            {
                "answer": str,
                "sources": [{"doc_id", "title", "source", "author", "chunk_preview"}],
                "question": str
            }
        """
        chain = self._build_qa_chain(source_filter)
        callbacks = [StdOutCallbackHandler()] if verbose else []

        result = chain.invoke(
            {"query": question},
            config={"callbacks": callbacks},
        )

        # Extract source metadata
        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            doc_id = doc.metadata.get("doc_id", "unknown")
            if doc_id not in seen:
                sources.append({
                    "doc_id": doc_id,
                    "title": doc.metadata.get("title", "Unknown"),
                    "source": doc.metadata.get("source", "unknown"),
                    "author": doc.metadata.get("author", "Unknown"),
                    "date": doc.metadata.get("date", ""),
                    "chunk_preview": doc.page_content[:200] + "...",
                })
                seen.add(doc_id)

        return {
            "answer": result.get("result", ""),
            "sources": sources,
            "question": question,
        }

    def compare(self, question: str, verbose: bool = False) -> dict:
        """
        Compare accounts across Lincoln's writings and biographies.
        
        Retrieves from ALL sources and asks the LLM to identify
        agreements, disagreements, and relative reliability.
        
        Args:
            question: Comparative question (e.g., "How do accounts differ 
                     on Lincoln's Fort Sumter decision?")
            verbose: Print chain execution details
        
        Returns:
            Same format as ask()
        """
        chain = self._build_compare_chain()
        callbacks = [StdOutCallbackHandler()] if verbose else []

        result = chain.invoke(
            {"query": question},
            config={"callbacks": callbacks},
        )

        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            doc_id = doc.metadata.get("doc_id", "unknown")
            if doc_id not in seen:
                sources.append({
                    "doc_id": doc_id,
                    "title": doc.metadata.get("title", "Unknown"),
                    "source": doc.metadata.get("source", "unknown"),
                    "author": doc.metadata.get("author", "Unknown"),
                    "date": doc.metadata.get("date", ""),
                    "chunk_preview": doc.page_content[:200] + "...",
                })
                seen.add(doc_id)

        return {
            "answer": result.get("result", ""),
            "sources": sources,
            "question": question,
        }

    def ask_lincoln_only(self, question: str) -> dict:
        """Convenience: query only Lincoln's own writings (LoC)."""
        return self.ask(question, source_filter="loc")

    def ask_biographers_only(self, question: str) -> dict:
        """Convenience: query only biographer accounts (Gutenberg)."""
        return self.ask(question, source_filter="gutenberg")

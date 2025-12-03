"""
Event Extractor - Core extraction logic combining chunking and LLM.
"""

import logging
from typing import Optional

from src.contracts.document import NormalizedDocument
from src.contracts.extraction import (
    EventExtraction, SourceChunk, TemporalDetails
)
from src.part2_extraction.config import EventConfig, ChunkConfig
from src.part2_extraction.chunker import TextChunker, ScoredChunk, should_chunk
from src.part2_extraction.llm import BaseLLM
from src.part2_extraction.prompts import get_extraction_prompt

logger = logging.getLogger(__name__)


class EventExtractor:
    """Extracts event information from documents using LLM."""
    
    def __init__(
        self,
        llm: BaseLLM,
        chunk_config: ChunkConfig = None,
        prompt_strategy: str = "cot",
    ):
        self.llm = llm
        self.chunker = TextChunker(chunk_config or ChunkConfig())
        self.chunk_config = chunk_config or ChunkConfig()
        self.prompt_strategy = prompt_strategy
    
    async def extract_event(
        self,
        document: NormalizedDocument,
        event: EventConfig,
    ) -> EventExtraction:
        """
        Extract information about a single event from a document.
        
        For large documents: chunk → search → extract from top chunks
        For small documents: extract directly
        """
        word_count = len(document.content.split())
        author = document.from_ or "Unknown"
        
        # Small document: process entire content
        if not should_chunk(word_count):
            logger.debug(f"  Processing {document.id} directly ({word_count} words)")
            return await self._extract_from_text(
                document=document,
                event=event,
                text=document.content,
                source_chunks=[],
                author=author,
            )
        
        # Large document: chunk and search
        logger.debug(f"  Chunking {document.id} ({word_count} words)")
        chunks = self.chunker.chunk_text(document.content)
        scored_chunks = self.chunker.search_chunks(chunks, event)
        
        # No relevant chunks found
        if not scored_chunks:
            logger.debug(f"  No chunks matched for {event.id} in {document.id}")
            return EventExtraction(
                event_id=event.id,
                event_name=event.name,
                document_id=document.id,
                author=author,
                claims=[],
                temporal_details=None,
                tone="Neutral",
                source_chunks=[],
            )
        
        # Build context from top chunks
        combined_text = self._combine_chunks(scored_chunks)
        source_chunks = self._build_provenance(scored_chunks)
        
        logger.debug(f"  Found {len(scored_chunks)} chunks for {event.id}")
        
        return await self._extract_from_text(
            document=document,
            event=event,
            text=combined_text,
            source_chunks=source_chunks,
            author=author,
        )
    
    async def _extract_from_text(
        self,
        document: NormalizedDocument,
        event: EventConfig,
        text: str,
        source_chunks: list[SourceChunk],
        author: str,
    ) -> EventExtraction:
        """Run LLM extraction on text."""
        messages = get_extraction_prompt(event, text, self.prompt_strategy)
        
        try:
            result = await self.llm.extract_json(messages, temperature=0.0)
        except Exception as e:
            logger.error(f"  LLM extraction failed for {event.id}/{document.id}: {e}")
            return EventExtraction(
                event_id=event.id,
                event_name=event.name,
                document_id=document.id,
                author=author,
                claims=[],
                temporal_details=None,
                tone="Neutral",
                source_chunks=source_chunks,
            )
        
        # Parse LLM output
        claims = result.get("claims", [])
        temporal = result.get("temporal_details")
        tone = result.get("tone", "Neutral")
        
        # Validate tone
        valid_tones = {"Sympathetic", "Critical", "Neutral", "Admiring", "Defensive", "Other"}
        if tone not in valid_tones:
            tone = "Other"
        
        return EventExtraction(
            event_id=event.id,
            event_name=event.name,
            document_id=document.id,
            author=author,
            claims=claims if isinstance(claims, list) else [],
            temporal_details=TemporalDetails(**temporal) if temporal else None,
            tone=tone,
            source_chunks=source_chunks,
        )
    
    def _combine_chunks(self, scored_chunks: list[ScoredChunk]) -> str:
        """Combine multiple chunks into single text for LLM."""
        texts = []
        for i, sc in enumerate(scored_chunks):
            texts.append(f"[Passage {i+1}]\n{sc.chunk.text}")
        return "\n\n---\n\n".join(texts)
    
    def _build_provenance(self, scored_chunks: list[ScoredChunk]) -> list[SourceChunk]:
        """Build provenance info for chunks."""
        return [
            SourceChunk(
                chunk_index=sc.chunk.index,
                text_preview=sc.chunk.text[:200],
                keyword_matches=sc.matched_keywords,
            )
            for sc in scored_chunks
        ]
"""
Text Chunker - Split documents and find relevant chunks via keyword search.
"""

from dataclasses import dataclass
from src.part2_extraction.config import ChunkConfig, EventConfig


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    index: int
    text: str
    start_char: int
    end_char: int


@dataclass 
class ScoredChunk:
    """A chunk with relevance score."""
    chunk: Chunk
    score: float
    matched_keywords: list[str]


class TextChunker:
    """Splits text into overlapping chunks."""
    
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
    
    def chunk_text(self, text: str) -> list[Chunk]:
        """Split text into overlapping chunks."""
        chunks = []
        chunk_size = self.config.chunk_size_chars
        overlap = self.config.overlap_chars
        step = chunk_size - overlap
        
        start = 0
        index = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind('.', start + step, end)
                if last_period > start + step // 2:
                    end = last_period + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    index=index,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                ))
                index += 1
            
            start = start + step
            if start + overlap >= len(text):
                break
        
        return chunks
    
    def search_chunks(
        self, 
        chunks: list[Chunk], 
        event: EventConfig,
        top_n: int = None
    ) -> list[ScoredChunk]:
        """
        Find chunks relevant to an event using keyword matching.
        
        Scoring:
        - Primary keyword match: +10 points each
        - Secondary keyword match: +3 points each
        """
        top_n = top_n or self.config.top_n_chunks
        scored = []
        
        for chunk in chunks:
            text_lower = chunk.text.lower()
            score = 0.0
            matched = []
            
            # Check primary keywords
            for kw in event.primary_keywords:
                if kw.lower() in text_lower:
                    score += 10
                    matched.append(kw)
            
            # Skip if no primary match
            if score == 0:
                continue
            
            # Check secondary keywords
            for kw in event.secondary_keywords:
                if kw.lower() in text_lower:
                    score += 3
                    matched.append(kw)
            
            scored.append(ScoredChunk(chunk=chunk, score=score, matched_keywords=matched))
        
        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]


def should_chunk(word_count: int, threshold: int = 5000) -> bool:
    """Determine if document needs chunking based on size."""
    return word_count > threshold
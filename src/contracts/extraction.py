"""
Extraction Schema - Contract between Part 2 and Part 3.

Defines the structured output from event extraction.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime, timezone


class SourceChunk(BaseModel):
    """Provenance info for debugging and Part 3 analysis."""
    chunk_index: int
    text_preview: str = Field(description="First 200 chars of chunk")
    keyword_matches: list[str] = Field(default_factory=list)


class TemporalDetails(BaseModel):
    """Temporal information extracted for an event."""
    date: Optional[str] = None
    time: Optional[str] = None


class EventExtraction(BaseModel):
    """
    Extracted information about a single event from a single document.
    
    One per (document_id, event_id) pair.
    """
    event_id: str = Field(description="e.g., 'election_1860', 'fort_sumter'")
    event_name: str = Field(description="Human-readable: 'Election Night 1860'")
    document_id: str = Field(description="e.g., 'gutenberg_6812', 'loc_gettysburg'")
    author: str
    claims: list[str] = Field(default_factory=list, description="Extracted factual claims")
    temporal_details: Optional[TemporalDetails] = None
    tone: Literal["Sympathetic", "Critical", "Neutral", "Admiring", "Defensive", "Other"] = "Neutral"
    source_chunks: list[SourceChunk] = Field(default_factory=list)


class DocumentExtractions(BaseModel):
    """All event extractions from a single document."""
    document_id: str
    document_title: str
    author: str
    source: Literal["gutenberg", "loc"]
    extractions: list[EventExtraction] = Field(default_factory=list)


class ExtractionCollection(BaseModel):
    """Complete extraction output for Part 2."""
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    llm_provider: str
    llm_model: str
    prompt_strategy: str = Field(description="zero_shot | cot | few_shot")
    documents: list[DocumentExtractions] = Field(default_factory=list)
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
    
    def to_file(self, path: str) -> None:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2))
    
    @classmethod
    def from_file(cls, path: str) -> "ExtractionCollection":
        from pathlib import Path
        return cls.model_validate_json(Path(path).read_text())
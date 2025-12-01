"""
Document Schema - Contract between Part 1 and Part 2.

This defines the normalized document structure that Part 1 outputs
and Part 2 consumes. Changes here affect both parts.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime, timezone


class NormalizedDocument(BaseModel):
    """
    Single document in normalized format.
    
    Schema matches assignment requirements:
    - id: unique identifier
    - title: human-readable title
    - reference: path to raw data file
    - document_type: Letter | Speech | Note | Book | Chapter
    - date: as in source (no added/removed precision)
    - place: if known
    - from_: sender (if applicable)
    - to: recipient (if applicable)  
    - content: full source text
    """
    id: str = Field(description="Unique identifier (e.g., 'gutenberg_6812', 'loc_mal0440500')")
    title: str = Field(description="Human-readable title")
    reference: str = Field(description="Path to raw data file")
    document_type: Literal["Letter", "Speech", "Note", "Book", "Chapter", "Address"] = Field(
        description="Type of document"
    )
    date: Optional[str] = Field(default=None, description="Date as in source, no normalization")
    place: Optional[str] = Field(default=None, description="Location if known")
    from_: Optional[str] = Field(default=None, alias="from", description="Sender if applicable")
    to: Optional[str] = Field(default=None, description="Recipient if applicable")
    content: str = Field(description="Full source text")

    class Config:
        populate_by_name = True


class DocumentCollection(BaseModel):
    """
    Collection of documents from a single source.
    
    Used for serialization to JSON files.
    """
    source: Literal["gutenberg", "loc"] = Field(description="Data source identifier")
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    documents: list[NormalizedDocument] = Field(default_factory=list)
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
    
    def to_file(self, path: str) -> None:
        """Save collection to JSON file."""
        from pathlib import Path
        Path(path).write_text(self.model_dump_json(indent=2, by_alias=True))
    
    @classmethod
    def from_file(cls, path: str) -> "DocumentCollection":
        """Load collection from JSON file."""
        from pathlib import Path
        return cls.model_validate_json(Path(path).read_text())
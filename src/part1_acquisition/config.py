"""
Part 1 Configuration - Isolated config for data acquisition.

All URLs, rate limits, and paths specific to Part 1 are defined here.
"""

from dataclasses import dataclass, field


@dataclass
class GutenbergConfig:
    """Project Gutenberg scraping configuration."""
    
    # Book IDs to download (from assignment)
    book_ids: list[str] = field(default_factory=lambda: [
        "6812",   # Lincoln biography
        "6811",   # Lincoln biography  
        "12801",  # Lincoln biography
        "14004",  # Lincoln biography
        "18379",  # Lincoln biography
    ])
    
    # Base URL for plain text downloads
    base_url: str = "https://www.gutenberg.org"
    
    # Requests per minute (Gutenberg is lenient)
    requests_per_minute: int = 30
    
    def get_book_url(self, book_id: str) -> str:
        """Get the ebook page URL for a book ID."""
        return f"{self.base_url}/ebooks/{book_id}"
    
    def get_txt_url(self, book_id: str) -> str:
        """Get plain text download URL for a book ID."""
        return f"{self.base_url}/cache/epub/{book_id}/pg{book_id}.txt"


@dataclass  
class LocConfig:
    """Library of Congress scraping configuration."""
    
    # Documents to download (from assignment)
    documents: list[dict] = field(default_factory=lambda: [
        {
            "id": "mal0440500",
            "name": "Election Night 1860",
            "url": "https://www.loc.gov/item/mal0440500/",
            "type": "item",
            "document_type": "Letter",
        },
        {
            "id": "mal0882800",
            "name": "Fort Sumter Decision",
            "url": "https://www.loc.gov/resource/mal.0882800/",
            "type": "resource",
            "document_type": "Letter",
        },
        {
            "id": "gettysburg",
            "name": "Gettysburg Address",
            "url": "https://www.loc.gov/exhibits/gettysburg-address/ext/trans-nicolay-copy.html",
            "type": "exhibits",
            "document_type": "Speech",
        },
        {
            "id": "mal4361300",
            "name": "Second Inaugural Address",
            "url": "https://www.loc.gov/resource/mal.4361300/",
            "type": "resource",
            "document_type": "Address",
            # Fallback: URL returns 404, use direct transcript
            "fallback_transcript_url": "https://tile.loc.gov/storage-services/service/mss/mal/436/4361300/4361300.pdf",
        },
        {
            "id": "mal4361800",
            "name": "Last Public Address",
            "url": "https://www.loc.gov/resource/mal.4361800/",
            "type": "resource",
            "document_type": "Speech",
        },
    ])
    
    # Rate limit: 20 requests/minute per LoC docs
    requests_per_minute: int = 20


@dataclass
class Part1Config:
    """Combined configuration for Part 1."""
    
    gutenberg: GutenbergConfig = field(default_factory=GutenbergConfig)
    loc: LocConfig = field(default_factory=LocConfig)
    
    # Storage paths
    raw_gutenberg_path: str = "raw/gutenberg"
    raw_loc_path: str = "raw/loc"
    normalized_path: str = "normalized"
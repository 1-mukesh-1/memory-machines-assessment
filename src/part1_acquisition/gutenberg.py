"""
Gutenberg Scraper - Downloads books from Project Gutenberg.

Fetches plain text versions and extracts metadata from ebook pages.
"""

import re
import logging
from bs4 import BeautifulSoup
from src.contracts.document import NormalizedDocument
from src.shared.http import HttpClient
from src.shared.storage import BaseStorage
from src.part1_acquisition.base import BaseScraper
from src.part1_acquisition.config import GutenbergConfig

logger = logging.getLogger(__name__)


class GutenbergScraper(BaseScraper):
    """
    Scraper for Project Gutenberg books.
    
    Downloads plain text (.txt) files and extracts metadata
    from the ebook HTML pages.
    """
    
    def __init__(
        self, 
        client: HttpClient, 
        storage: BaseStorage,
        config: GutenbergConfig = None
    ):
        super().__init__(client, storage)
        self.config = config or GutenbergConfig()
    
    async def fetch_all(self) -> list[NormalizedDocument]:
        """Fetch all configured Gutenberg books."""
        documents = []
        for book_id in self.config.book_ids:
            try:
                doc = await self.fetch_one(book_id)
                documents.append(doc)
                logger.info(f"✓ Fetched Gutenberg book {book_id}: {doc.title}")
            except Exception as e:
                logger.error(f"✗ Failed to fetch book {book_id}: {e}")
        return documents
    
    async def fetch_one(self, book_id: str) -> NormalizedDocument:
        """Fetch a single Gutenberg book by ID."""
        # 1. Get metadata from ebook page
        metadata = await self._fetch_metadata(book_id)
        
        # 2. Download plain text content
        content = await self._fetch_text(book_id)
        
        # 3. Save raw content
        raw_path = f"{self.config.base_url}/ebooks/{book_id}"
        storage_key = f"raw/gutenberg/{book_id}.txt"
        self.storage.save_raw(content, storage_key)
        
        # 4. Build normalized document
        return NormalizedDocument(
            id=f"gutenberg_{book_id}",
            title=metadata.get("title", f"Unknown Book {book_id}"),
            reference=storage_key,
            document_type="Book",
            date=metadata.get("date"),
            place=None,
            from_=metadata.get("author"),
            to=None,
            content=content,
        )
    
    async def _fetch_metadata(self, book_id: str) -> dict:
        """Extract metadata from Gutenberg ebook page."""
        url = self.config.get_book_url(book_id)
        html = await self.client.get_text(url)
        soup = BeautifulSoup(html, "lxml")
        
        metadata = {}
        
        # Extract title
        title_tag = soup.select_one("h1[itemprop='name']") or soup.select_one("h1")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)
        
        # Extract author
        author_tag = soup.select_one("a[itemprop='creator']") or soup.select_one("a[href*='/author/']")
        if author_tag:
            metadata["author"] = author_tag.get_text(strip=True)
        
        # Extract release date
        for row in soup.select("tr"):
            header = row.select_one("th")
            if header and "Release Date" in header.get_text():
                value = row.select_one("td")
                if value:
                    metadata["date"] = value.get_text(strip=True)
                break
        
        return metadata
    
    async def _fetch_text(self, book_id: str) -> str:
        """Download plain text content of book."""
        url = self.config.get_txt_url(book_id)
        
        try:
            content = await self.client.get_text(url)
        except Exception:
            # Fallback: try alternate URL pattern
            alt_url = f"{self.config.base_url}/files/{book_id}/{book_id}-0.txt"
            content = await self.client.get_text(alt_url)
        
        return self._clean_gutenberg_text(content)
    
    def _clean_gutenberg_text(self, text: str) -> str:
        """
        Light cleaning of Gutenberg text.
        
        Preserves original formatting but removes Gutenberg header/footer.
        """
        # Find start marker
        start_markers = [
            r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG",
            r"\*\*\*START OF (THE|THIS) PROJECT GUTENBERG",
        ]
        
        for pattern in start_markers:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Find end of the line containing the marker
                start_idx = text.find("\n", match.end())
                if start_idx != -1:
                    text = text[start_idx + 1:]
                break
        
        # Find end marker
        end_markers = [
            r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG",
            r"\*\*\*END OF (THE|THIS) PROJECT GUTENBERG",
            r"End of (the )?Project Gutenberg",
        ]
        
        for pattern in end_markers:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                text = text[:match.start()]
                break
        
        return text.strip()
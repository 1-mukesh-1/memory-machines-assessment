"""
Library of Congress Scraper - Downloads Lincoln documents from LoC.

Handles 3 different URL patterns:
- /item/ → JSON API with transcript
- /resource/ → JSON API with transcript  
- /exhibits/ → HTML scraping

Also handles broken URLs with fallback to direct transcript URLs.
"""

import re
import logging
from bs4 import BeautifulSoup
from src.contracts.document import NormalizedDocument
from src.shared.http import HttpClient
from src.shared.storage import BaseStorage
from src.part1_acquisition.base import BaseScraper
from src.part1_acquisition.config import LocConfig

logger = logging.getLogger(__name__)


class LocScraper(BaseScraper):
    """
    Scraper for Library of Congress Lincoln documents.
    
    Handles multiple URL patterns and fallback strategies
    for broken links.
    """
    
    def __init__(
        self, 
        client: HttpClient, 
        storage: BaseStorage,
        config: LocConfig = None
    ):
        super().__init__(client, storage)
        self.config = config or LocConfig()
    
    async def fetch_all(self) -> list[NormalizedDocument]:
        """Fetch all configured LoC documents."""
        documents = []
        for doc_config in self.config.documents:
            try:
                doc = await self.fetch_one(doc_config["id"])
                documents.append(doc)
                logger.info(f"✓ Fetched LoC document: {doc.title}")
            except Exception as e:
                logger.error(f"✗ Failed to fetch {doc_config['name']}: {e}")
        return documents
    
    async def fetch_one(self, item_id: str) -> NormalizedDocument:
        """Fetch a single LoC document by ID."""
        # Find config for this document
        doc_config = next(
            (d for d in self.config.documents if d["id"] == item_id), 
            None
        )
        if not doc_config:
            raise ValueError(f"Unknown document ID: {item_id}")
        
        url_type = doc_config["type"]
        
        # Route to appropriate handler
        if url_type == "exhibits":
            return await self._fetch_exhibits(doc_config)
        else:
            return await self._fetch_api(doc_config)
    
    async def _fetch_api(self, doc_config: dict) -> NormalizedDocument:
        """Fetch document using LoC JSON API."""
        url = doc_config["url"].rstrip("/") + "/?fo=json"
        
        try:
            data = await self.client.get_json(url)
        except Exception as e:
            # Try fallback if available
            if "fallback_transcript_url" in doc_config:
                logger.warning(f"API failed, using fallback for {doc_config['name']}")
                return await self._fetch_fallback(doc_config)
            raise
        
        # Extract metadata
        item = data.get("item", {})
        resources = data.get("resources", [])
        
        title = item.get("title", doc_config["name"])
        date = self._extract_date(item)
        
        # Get transcript content
        content = await self._fetch_transcript(resources, doc_config)
        
        # Save raw content
        storage_key = f"raw/loc/{doc_config['id']}.txt"
        self.storage.save_raw(content, storage_key)
        
        return NormalizedDocument(
            id=f"loc_{doc_config['id']}",
            title=title,
            reference=storage_key,
            document_type=doc_config.get("document_type", "Letter"),
            date=date,
            place=self._extract_place(item),
            from_=self._extract_contributor(item, "from"),
            to=self._extract_contributor(item, "to"),
            content=content,
        )
    
    async def _fetch_transcript(self, resources: list, doc_config: dict) -> str:
        """Fetch transcript from resources list."""
        # Look for transcript URL in resources
        transcript_url = None
        for resource in resources:
            if "fulltext_file" in resource:
                transcript_url = resource["fulltext_file"]
                break
        
        if not transcript_url:
            # Try constructing URL from ID pattern
            doc_id = doc_config["id"]
            # Pattern: mal0440500 -> /mss/mal/044/0440500/0440500.xml
            if doc_id.startswith("mal"):
                num = doc_id[3:]  # Remove 'mal' prefix
                prefix = num[:3]
                transcript_url = f"https://tile.loc.gov/storage-services/service/mss/mal/{prefix}/{doc_id}/{doc_id}.xml"
        
        if transcript_url:
            content = await self.client.get_text(transcript_url)
            return self._parse_transcript(content, transcript_url)
        
        raise ValueError(f"No transcript found for {doc_config['name']}")
    
    def _parse_transcript(self, content: str, url: str) -> str:
        """Parse transcript content based on format."""
        if url.endswith(".xml"):
            return self._parse_xml_transcript(content)
        elif url.endswith(".txt"):
            return content.strip()
        elif url.endswith(".pdf"):
            # For PDF, we'd need OCR - log warning
            logger.warning("PDF transcript requires manual processing")
            return content
        return content
    
    def _parse_xml_transcript(self, xml_content: str) -> str:
        """Extract text from LoC XML transcript format."""
        soup = BeautifulSoup(xml_content, "lxml-xml")
        
        # Try common XML structures
        # 1. TEI format
        body = soup.find("body") or soup.find("text")
        if body:
            return body.get_text(separator="\n", strip=True)
        
        # 2. Simple text content
        return soup.get_text(separator="\n", strip=True)
    
    async def _fetch_exhibits(self, doc_config: dict) -> NormalizedDocument:
        """Fetch document from exhibits HTML page."""
        html = await self.client.get_text(doc_config["url"])
        soup = BeautifulSoup(html, "lxml")
        
        # Extract transcript text (specific to Gettysburg Address page)
        content = self._extract_exhibits_content(soup)
        
        # Save raw content
        storage_key = f"raw/loc/{doc_config['id']}.txt"
        self.storage.save_raw(content, storage_key)
        
        return NormalizedDocument(
            id=f"loc_{doc_config['id']}",
            title=doc_config["name"],
            reference=storage_key,
            document_type=doc_config.get("document_type", "Speech"),
            date="November 19, 1863",  # Known date for Gettysburg Address
            place="Gettysburg, Pennsylvania",
            from_="Abraham Lincoln",
            to=None,
            content=content,
        )
    
    def _extract_exhibits_content(self, soup: BeautifulSoup) -> str:
        """Extract transcript from exhibits HTML page."""
        content_parts = []
        
        # Footer patterns to exclude
        footer_patterns = [
            "About|Press|Jobs",
            "Inspector General",
            "Accessibility",
            "USA.gov",
            "External Link Disclaimer",
        ]
        
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if not text or len(text) < 50:
                continue
            # Skip footer content
            if any(pattern in text for pattern in footer_patterns):
                continue
            content_parts.append(text)
        
        return "\n\n".join(content_parts)
    
    async def _fetch_fallback(self, doc_config: dict) -> NormalizedDocument:
        """Fetch using fallback transcript URL (for broken main URLs)."""
        fallback_url = doc_config["fallback_transcript_url"]
        
        # For PDF, we need special handling
        if fallback_url.endswith(".pdf"):
            # Log that manual processing may be needed
            logger.warning(f"PDF fallback for {doc_config['name']} - extracting text")
            # Try to get text from PDF (basic extraction)
            response = await self.client.get(fallback_url)
            content = f"[PDF content from: {fallback_url}]\n\nManual transcription required."
        else:
            content = await self.client.get_text(fallback_url)
            content = self._parse_transcript(content, fallback_url)
        
        storage_key = f"raw/loc/{doc_config['id']}.txt"
        self.storage.save_raw(content, storage_key)
        
        return NormalizedDocument(
            id=f"loc_{doc_config['id']}",
            title=doc_config["name"],
            reference=storage_key,
            document_type=doc_config.get("document_type", "Speech"),
            date=None,
            place=None,
            from_="Abraham Lincoln",
            to=None,
            content=content,
        )
    
    def _extract_date(self, item: dict) -> str | None:
        """Extract date from item metadata."""
        if "date" in item:
            return item["date"]
        dates = item.get("dates", [])
        if dates and isinstance(dates[0], str):
            return dates[0]
        return None
    
    def _extract_place(self, item: dict) -> str | None:
        """Extract place from item metadata."""
        locations = item.get("location", [])
        if locations:
            if isinstance(locations[0], str):
                return locations[0]
            elif isinstance(locations[0], dict):
                return list(locations[0].keys())[0]
        return None
    
    def _extract_contributor(self, item: dict, role: str) -> str | None:
        """Extract contributor by role."""
        contributors = item.get("contributor", [])
        # For Lincoln papers, assume Lincoln is the author
        if role == "from":
            return "Abraham Lincoln"
        return None
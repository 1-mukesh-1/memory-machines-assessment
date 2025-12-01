"""
Base Scraper - Abstract interface for all scrapers.

All scrapers inherit from this and implement fetch_all().
"""

from abc import ABC, abstractmethod
from src.contracts.document import NormalizedDocument
from src.shared.http import HttpClient
from src.shared.storage import BaseStorage


class BaseScraper(ABC):
    """
    Abstract base class for document scrapers.
    
    Args:
        client: HTTP client for making requests
        storage: Storage backend for saving raw files
    """
    
    def __init__(self, client: HttpClient, storage: BaseStorage):
        self.client = client
        self.storage = storage
    
    @abstractmethod
    async def fetch_all(self) -> list[NormalizedDocument]:
        """
        Fetch and normalize all documents from this source.
        
        Returns:
            List of normalized documents
        """
        pass
    
    @abstractmethod
    async def fetch_one(self, item_id: str) -> NormalizedDocument:
        """
        Fetch and normalize a single document.
        
        Args:
            item_id: Identifier for the document
            
        Returns:
            Normalized document
        """
        pass
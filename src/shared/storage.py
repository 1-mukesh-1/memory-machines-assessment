"""
Storage - Abstract file storage with local implementation.

Swap LocalStorage for S3Storage later without changing scraper code.

Usage:
    storage = LocalStorage(base_path="data")
    storage.save_raw("content", "raw/gutenberg/6812.txt")
    storage.save_json({"key": "value"}, "normalized/gutenberg.json")
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from datetime import datetime
import json


def json_serializer(obj):
    """Custom JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class BaseStorage(ABC):
    """Abstract storage interface for file operations."""
    
    @abstractmethod
    def save_raw(self, content: str, key: str) -> str:
        """Save raw text content. Returns the saved path."""
        pass
    
    @abstractmethod
    def load_raw(self, key: str) -> str:
        """Load raw text content."""
        pass
    
    @abstractmethod
    def save_json(self, data: Any, key: str) -> str:
        """Save JSON data. Returns the saved path."""
        pass
    
    @abstractmethod
    def load_json(self, key: str) -> Any:
        """Load JSON data."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if file exists."""
        pass


class LocalStorage(BaseStorage):
    """
    Local filesystem storage implementation.
    
    Args:
        base_path: Root directory for all storage (default: "data")
    """
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _resolve(self, key: str) -> Path:
        """Resolve key to full path."""
        path = self.base_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def save_raw(self, content: str, key: str) -> str:
        """Save raw text content."""
        path = self._resolve(key)
        path.write_text(content, encoding="utf-8")
        return str(path)
    
    def load_raw(self, key: str) -> str:
        """Load raw text content."""
        path = self._resolve(key)
        return path.read_text(encoding="utf-8")
    
    def save_json(self, data: Any, key: str) -> str:
        """Save JSON data with pretty printing."""
        path = self._resolve(key)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=json_serializer), 
            encoding="utf-8"
        )
        return str(path)
    
    def load_json(self, key: str) -> Any:
        """Load JSON data."""
        path = self._resolve(key)
        return json.loads(path.read_text(encoding="utf-8"))
    
    def exists(self, key: str) -> bool:
        """Check if file exists."""
        return (self.base_path / key).exists()
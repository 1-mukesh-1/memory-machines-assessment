"""
HTTP Client - Shared async HTTP client with rate limiting and retries.

Usage:
    async with HttpClient(requests_per_minute=20) as client:
        response = await client.get("https://example.com")
"""

import httpx
import asyncio
from typing import Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class HttpClient:
    """
    Async HTTP client with built-in rate limiting and retry logic.
    
    Args:
        requests_per_minute: Max requests per minute (default: 20 for LoC)
        max_retries: Number of retry attempts (default: 3)
        timeout: Request timeout in seconds (default: 30)
    """
    requests_per_minute: int = 20
    max_retries: int = 3
    timeout: float = 30.0
    
    _client: Optional[httpx.AsyncClient] = field(default=None, repr=False)
    _semaphore: Optional[asyncio.Semaphore] = field(default=None, repr=False)
    
    @property
    def _delay(self) -> float:
        """Delay between requests to respect rate limit."""
        return 60.0 / self.requests_per_minute
    
    async def __aenter__(self) -> "HttpClient":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        self._semaphore = asyncio.Semaphore(1)  # Sequential requests
        return self
    
    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
    
    async def get(
        self, 
        url: str, 
        params: Optional[dict] = None,
        headers: Optional[dict] = None
    ) -> httpx.Response:
        """
        GET request with rate limiting and retries.
        
        Raises:
            httpx.HTTPStatusError: After all retries exhausted
        """
        async with self._semaphore:
            for attempt in range(self.max_retries):
                try:
                    logger.debug(f"GET {url} (attempt {attempt + 1})")
                    response = await self._client.get(
                        url, 
                        params=params, 
                        headers=headers,
                        follow_redirects=True
                    )
                    
                    # Handle rate limiting (429)
                    if response.status_code == 429:
                        wait_time = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited. Waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    response.raise_for_status()
                    await asyncio.sleep(self._delay)  # Rate limit delay
                    return response
                    
                except httpx.HTTPStatusError as e:
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries}: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
                except httpx.RequestError as e:
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(f"Request error, retry {attempt + 1}: {e}")
                    await asyncio.sleep(2 ** attempt)
        
        raise RuntimeError("Unreachable")

    async def get_text(self, url: str) -> str:
        """GET request returning text content."""
        response = await self.get(url)
        return response.text
    
    async def get_json(self, url: str) -> dict:
        """GET request returning JSON content."""
        response = await self.get(url)
        return response.json()
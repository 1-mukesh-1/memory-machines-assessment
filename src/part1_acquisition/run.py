"""
Part 1 Entry Point - Run the complete acquisition pipeline.

Usage:
    from src.part1_acquisition.run import run
    gutenberg, loc = asyncio.run(run())
    
CLI:
    python -m src.part1_acquisition.run
"""

import asyncio
import logging
from datetime import datetime, timezone

from src.contracts.document import DocumentCollection
from src.shared.http import HttpClient
from src.shared.storage import LocalStorage
from src.part1_acquisition.config import Part1Config
from src.part1_acquisition.gutenberg import GutenbergScraper
from src.part1_acquisition.loc import LocScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


async def run(config: Part1Config = None) -> tuple[DocumentCollection, DocumentCollection]:
    """
    Run the complete Part 1 acquisition pipeline.
    
    Steps:
        1. Fetch all Gutenberg books
        2. Fetch all LoC documents
        3. Save normalized JSON files
    
    Returns:
        Tuple of (gutenberg_collection, loc_collection)
    """
    config = config or Part1Config()
    storage = LocalStorage(base_path="data")
    
    logger.info("=" * 60)
    logger.info("PART 1: DATA ACQUISITION & NORMALIZATION")
    logger.info("=" * 60)
    
    # 1. Fetch Gutenberg books
    logger.info("\n[1/2] Fetching Project Gutenberg books...")
    async with HttpClient(requests_per_minute=config.gutenberg.requests_per_minute) as client:
        gutenberg_scraper = GutenbergScraper(client, storage, config.gutenberg)
        gutenberg_docs = await gutenberg_scraper.fetch_all()
    
    gutenberg_collection = DocumentCollection(
        source="gutenberg",
        scraped_at=datetime.now(timezone.utc),
        documents=gutenberg_docs,
    )
    
    # 2. Fetch LoC documents
    logger.info("\n[2/2] Fetching Library of Congress documents...")
    async with HttpClient(requests_per_minute=config.loc.requests_per_minute) as client:
        loc_scraper = LocScraper(client, storage, config.loc)
        loc_docs = await loc_scraper.fetch_all()
    
    loc_collection = DocumentCollection(
        source="loc",
        scraped_at=datetime.now(timezone.utc),
        documents=loc_docs,
    )
    
    # 3. Save normalized collections
    logger.info("\n[3/3] Saving normalized datasets...")
    
    gutenberg_path = f"{config.normalized_path}/gutenberg.json"
    storage.save_json(gutenberg_collection.model_dump(by_alias=True), gutenberg_path)
    logger.info(f"✓ Saved {len(gutenberg_docs)} Gutenberg docs → {gutenberg_path}")
    
    loc_path = f"{config.normalized_path}/loc.json"
    storage.save_json(loc_collection.model_dump(by_alias=True), loc_path)
    logger.info(f"✓ Saved {len(loc_docs)} LoC docs → {loc_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ACQUISITION COMPLETE")
    logger.info(f"  Gutenberg: {len(gutenberg_docs)}/5 books")
    logger.info(f"  LoC: {len(loc_docs)}/5 documents")
    logger.info("=" * 60)
    
    return gutenberg_collection, loc_collection


if __name__ == "__main__":
    asyncio.run(run())
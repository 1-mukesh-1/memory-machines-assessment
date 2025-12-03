"""
Part 2 Entry Point - Run the complete extraction pipeline.

Usage:
    python -m src.part2_extraction.run
    
Environment:
    LLM_PROVIDER=anthropic|openai|ollama
    LLM_MODEL=claude-sonnet-4-20250514|gpt-4o|llama3
    ANTHROPIC_API_KEY=...
    OPENAI_API_KEY=...
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.contracts.document import DocumentCollection
from src.contracts.extraction import (
    ExtractionCollection, DocumentExtractions
)
from src.part2_extraction.config import Part2Config
from src.part2_extraction.llm import get_llm
from src.part2_extraction.extractor import EventExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


async def run(config: Part2Config = None) -> ExtractionCollection:
    """
    Run the complete Part 2 extraction pipeline.
    
    Steps:
        1. Load normalized documents from Part 1
        2. For each (document, event) pair, extract claims
        3. Save extraction results
    """
    config = config or Part2Config()
    
    logger.info("=" * 60)
    logger.info("PART 2: EVENT EXTRACTION")
    logger.info("=" * 60)
    
    # Initialize LLM
    llm = get_llm()
    logger.info(f"Using LLM: {llm.provider}/{llm.model}")
    logger.info(f"Prompt strategy: {config.prompt_strategy}")
    
    extractor = EventExtractor(
        llm=llm,
        chunk_config=config.chunking,
        prompt_strategy=config.prompt_strategy,
    )
    
    # Load documents
    logger.info("\n[1/3] Loading normalized documents...")
    gutenberg = DocumentCollection.from_file(config.input_gutenberg)
    loc = DocumentCollection.from_file(config.input_loc)
    
    all_docs = [
        (doc, "gutenberg") for doc in gutenberg.documents
    ] + [
        (doc, "loc") for doc in loc.documents
    ]
    logger.info(f"Loaded {len(all_docs)} documents")
    
    # Extract events
    logger.info("\n[2/3] Extracting events...")
    document_extractions = []
    
    total_pairs = len(all_docs) * len(config.events)
    processed = 0
    
    for doc, source in all_docs:
        logger.info(f"\nProcessing: {doc.title[:50]}...")
        
        doc_extractions = DocumentExtractions(
            document_id=doc.id,
            document_title=doc.title,
            author=doc.from_ or "Unknown",
            source=source,
        )
        
        for event in config.events:
            processed += 1
            logger.info(f"  [{processed}/{total_pairs}] {event.name}")
            
            extraction = await extractor.extract_event(doc, event)
            doc_extractions.extractions.append(extraction)
            
            # Log summary
            claim_count = len(extraction.claims)
            if claim_count > 0:
                logger.info(f"    → {claim_count} claims, tone: {extraction.tone}")
            else:
                logger.info(f"    → No claims found")
        
        document_extractions.append(doc_extractions)
    
    # Build collection
    collection = ExtractionCollection(
        extracted_at=datetime.now(timezone.utc),
        llm_provider=llm.provider,
        llm_model=llm.model,
        prompt_strategy=config.prompt_strategy,
        documents=document_extractions,
    )
    
    # Save results
    logger.info("\n[3/3] Saving extraction results...")
    output_file = f"{config.output_path}/extractions_{config.prompt_strategy}.json"
    collection.to_file(output_file)
    logger.info(f"✓ Saved to {output_file}")
    
    # Summary
    total_claims = sum(
        len(e.claims) 
        for d in document_extractions 
        for e in d.extractions
    )
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"  Documents processed: {len(all_docs)}")
    logger.info(f"  Event pairs: {total_pairs}")
    logger.info(f"  Total claims extracted: {total_claims}")
    logger.info("=" * 60)
    
    return collection


async def run_all_strategies(config: Part2Config = None) -> dict[str, ExtractionCollection]:
    """Run extraction with all prompt strategies for Part 3 comparison."""
    config = config or Part2Config()
    results = {}
    
    for strategy in ["zero_shot", "cot", "few_shot"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING STRATEGY: {strategy}")
        logger.info("=" * 60)
        
        config.prompt_strategy = strategy
        results[strategy] = await run(config)
    
    return results


if __name__ == "__main__":
    asyncio.run(run())
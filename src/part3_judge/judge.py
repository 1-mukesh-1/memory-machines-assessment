"""
Judge Engine - Core comparison logic.
"""

import logging
from typing import Optional

from src.contracts.extraction import ExtractionCollection, EventExtraction
from src.contracts.judgment import (
    JudgmentResult, Contradiction, EventJudgments, JudgmentCollection
)
from src.part2_extraction.llm import BaseLLM
from src.part3_judge.prompts import get_judge_prompt

logger = logging.getLogger(__name__)


class JudgeEngine:
    """Compares Lincoln's claims against other authors."""
    
    def __init__(self, llm: BaseLLM, prompt_strategy: str = "cot"):
        self.llm = llm
        self.prompt_strategy = prompt_strategy
    
    async def judge_pair(
        self,
        event_name: str,
        event_id: str,
        lincoln_claims: list[str],
        other_claims: list[str],
        lincoln_doc_id: str,
        other_doc_id: str,
        other_author: str,
        temperature: float = 0.0,
    ) -> JudgmentResult:
        """Judge a single Lincoln vs Other pair."""
        
        messages = get_judge_prompt(
            event_name=event_name,
            lincoln_claims=lincoln_claims,
            other_claims=other_claims,
            other_author=other_author,
            strategy=self.prompt_strategy,
        )
        
        try:
            result = await self.llm.extract_json(messages, temperature=temperature)
        except Exception as e:
            logger.error(f"LLM judgment failed: {e}")
            return JudgmentResult(
                event_id=event_id,
                event_name=event_name,
                lincoln_doc_id=lincoln_doc_id,
                other_doc_id=other_doc_id,
                other_author=other_author,
                consistency_score=50,
                contradictions=[],
                reasoning=f"Error: {e}",
            )
        
        # Parse contradictions
        contradictions = []
        for c in result.get("contradictions", []):
            contradictions.append(Contradiction(
                type=c.get("type", "factual"),
                lincoln_claim=c.get("lincoln_claim"),
                other_claim=c.get("other_claim"),
                explanation=c.get("explanation", ""),
            ))
        
        return JudgmentResult(
            event_id=event_id,
            event_name=event_name,
            lincoln_doc_id=lincoln_doc_id,
            other_doc_id=other_doc_id,
            other_author=other_author,
            consistency_score=result.get("consistency_score", 50),
            contradictions=contradictions,
            reasoning=result.get("reasoning"),
        )


def build_comparison_pairs(extractions: ExtractionCollection) -> list[dict]:
    """
    Build all valid (Lincoln, Other) pairs from extractions.
    
    Returns list of dicts with keys:
        event_id, event_name, lincoln_doc_id, lincoln_claims,
        other_doc_id, other_author, other_claims
    """
    pairs = []
    
    # Separate by source
    loc_docs = [d for d in extractions.documents if d.source == "loc"]
    gutenberg_docs = [d for d in extractions.documents if d.source == "gutenberg"]
    
    # For each event, find Lincoln doc with claims
    event_ids = ["fort_sumter", "gettysburg_address", "second_inaugural"]
    event_names = {
        "fort_sumter": "Fort Sumter Decision",
        "gettysburg_address": "Gettysburg Address",
        "second_inaugural": "Second Inaugural Address",
    }
    
    for event_id in event_ids:
        # Find Lincoln's claims for this event
        lincoln_doc = None
        lincoln_claims = []
        
        for doc in loc_docs:
            ext = next((e for e in doc.extractions if e.event_id == event_id), None)
            if ext and ext.claims:
                lincoln_doc = doc
                lincoln_claims = ext.claims
                break
        
        if not lincoln_claims:
            continue
        
        # Find all Gutenberg docs with claims for this event
        for doc in gutenberg_docs:
            ext = next((e for e in doc.extractions if e.event_id == event_id), None)
            if ext and ext.claims:
                pairs.append({
                    "event_id": event_id,
                    "event_name": event_names[event_id],
                    "lincoln_doc_id": lincoln_doc.document_id,
                    "lincoln_claims": lincoln_claims,
                    "other_doc_id": doc.document_id,
                    "other_author": doc.author.split(",")[0],  # First name part
                    "other_claims": ext.claims,
                })
    
    return pairs
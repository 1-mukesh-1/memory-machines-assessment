"""
LangChain Judge - Replaces direct LLM calls in Part 3 with LangChain chains.

Benefits over the existing judge.py:
- Structured output parsing via Pydantic (no manual JSON extraction)
- Built-in retry logic and error handling
- Prompt templates with variable injection
- Easy to swap LLM providers
- Chain composition for multi-step reasoning

Usage:
    from src.part3_judge.langchain_judge import LangChainJudge
    judge = LangChainJudge()
    result = await judge.judge_pair(event_name, lincoln_claims, other_claims, ...)
"""

import os
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from pydantic import BaseModel, Field

from src.contracts.judgment import (
    JudgmentResult,
    Contradiction,
    EventJudgments,
    JudgmentCollection,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------
# Pydantic models for structured LLM output
# -----------------------------------------------
class ContradictionOutput(BaseModel):
    """Structured contradiction from LLM."""
    type: str = Field(description="One of: factual, interpretive, omission")
    lincoln_claim: Optional[str] = Field(
        default=None, description="Lincoln's claim, or null for omission by Lincoln"
    )
    other_claim: Optional[str] = Field(
        default=None, description="Other author's claim, or null for omission by other"
    )
    explanation: str = Field(description="Why this is a contradiction")


class JudgmentOutput(BaseModel):
    """Structured judgment output that LangChain will parse."""
    consistency_score: int = Field(
        description="Score from 0-100 indicating consistency between accounts",
        ge=0,
        le=100,
    )
    contradictions: list[ContradictionOutput] = Field(
        default_factory=list,
        description="List of contradictions found. Empty list if none.",
    )
    reasoning: str = Field(
        description="Brief summary of the analysis explaining the score"
    )


# -----------------------------------------------
# Prompt Templates
# -----------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are a historical accuracy evaluator comparing first-person accounts 
by Abraham Lincoln with third-party biographer accounts of the same events.

Your task is to assess CONSISTENCY - how well the biographer's claims align with Lincoln's own words.

CONTRADICTION TYPES:
- factual: Disagreement on verifiable facts (dates, numbers, names, locations, sequences)
- interpretive: Different characterization of motivations, emotions, or significance
- omission: Important information in one source that contradicts or undermines the other

SCORING RUBRIC:
- 0-20: Major factual contradictions that fundamentally misrepresent the event
- 21-40: Multiple factual errors or significant interpretive disagreements
- 41-60: Some factual discrepancies or notable interpretive differences
- 61-80: Minor discrepancies, mostly aligned with some interpretive variance
- 81-100: Strong alignment, differences are trivial or complementary details

NOTE: Biographers adding context Lincoln didn't mention is NOT a contradiction.
Only flag omissions when one source contradicts or undermines the other.

{format_instructions}"""

COT_HUMAN_PROMPT = """Compare these accounts of "{event_name}":

LINCOLN'S CLAIMS:
{lincoln_claims}

{other_author}'S CLAIMS:
{other_claims}

---

Think step by step:

1. ALIGNMENT CHECK: Which claims from {other_author} align with Lincoln's account?

2. CONTRADICTION SCAN: Are there any direct factual disagreements?
   Look for conflicting dates, names, numbers, or sequences.

3. INTERPRETATION DIFFERENCES: Does {other_author} characterize motivations 
   or significance differently than Lincoln's account suggests?

4. SIGNIFICANT OMISSIONS: Does either source omit something that contradicts 
   or undermines the other? (Adding context is NOT an omission.)

5. SCORE: Based on the above, assign a consistency score (0-100).

After your analysis, provide your final answer in the required format."""

ZERO_SHOT_HUMAN_PROMPT = """Compare these accounts of "{event_name}":

LINCOLN'S CLAIMS:
{lincoln_claims}

{other_author}'S CLAIMS:
{other_claims}

Provide your judgment in the required format."""

FEW_SHOT_HUMAN_PROMPT = """Here is an example judgment:

EXAMPLE - High consistency (score: 92):
Lincoln says the address was at Gettysburg about a civil war testing liberty.
Biographer adds the date (Nov 19, 1863) and that Everett spoke first.
Result: No contradictions. Biographer adds complementary context.

EXAMPLE - With contradiction (score: 65):
Lincoln says Chew arrived April 8. Biographer says the messenger arrived April 6.
Result: Factual contradiction on arrival date (2-day discrepancy).

---

Now compare these accounts of "{event_name}":

LINCOLN'S CLAIMS:
{lincoln_claims}

{other_author}'S CLAIMS:
{other_claims}

Provide your judgment in the required format."""


# -----------------------------------------------
# LangChain Judge
# -----------------------------------------------
class LangChainJudge:
    """
    LLM Judge using LangChain for structured output and chain composition.
    
    Drop-in replacement for the existing JudgeEngine in judge.py.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        prompt_strategy: str = "cot",
    ):
        self.model_name = model
        self.default_temperature = temperature
        self.prompt_strategy = prompt_strategy
        self.parser = PydanticOutputParser(pydantic_object=JudgmentOutput)
        self._chains: dict[str, LLMChain] = {}

    def _get_llm(self, temperature: float = None) -> ChatOpenAI:
        """Create LLM instance with specified temperature."""
        return ChatOpenAI(
            model=self.model_name,
            temperature=temperature if temperature is not None else self.default_temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    def _get_human_template(self, strategy: str) -> str:
        """Get the human prompt template for the given strategy."""
        templates = {
            "zero_shot": ZERO_SHOT_HUMAN_PROMPT,
            "cot": COT_HUMAN_PROMPT,
            "few_shot": FEW_SHOT_HUMAN_PROMPT,
        }
        return templates.get(strategy, COT_HUMAN_PROMPT)

    def _build_chain(self, strategy: str, temperature: float = None) -> LLMChain:
        """Build a LangChain chain for judging."""
        llm = self._get_llm(temperature)
        human_template = self._get_human_template(strategy)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                JUDGE_SYSTEM_PROMPT,
                partial_variables={
                    "format_instructions": self.parser.get_format_instructions()
                },
            ),
            HumanMessagePromptTemplate.from_template(human_template),
        ])

        return LLMChain(llm=llm, prompt=prompt)

    def _format_claims(self, claims: list[str]) -> str:
        """Format a list of claims as numbered text."""
        return "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(claims))

    # -----------------------------------------------
    # Core judging
    # -----------------------------------------------
    async def judge_pair(
        self,
        event_name: str,
        event_id: str,
        lincoln_claims: list[str],
        other_claims: list[str],
        lincoln_doc_id: str,
        other_doc_id: str,
        other_author: str,
        temperature: float = None,
        strategy: str = None,
    ) -> JudgmentResult:
        """
        Judge a single (Lincoln doc, Other doc) pair.
        
        Compatible with the existing JudgeEngine.judge_pair() interface.
        """
        strategy = strategy or self.prompt_strategy
        chain = self._build_chain(strategy, temperature)

        try:
            raw_output = await chain.ainvoke({
                "event_name": event_name,
                "lincoln_claims": self._format_claims(lincoln_claims),
                "other_claims": self._format_claims(other_claims),
                "other_author": other_author,
            })

            # Parse structured output
            parsed: JudgmentOutput = self.parser.parse(raw_output["text"])

            # Convert to existing contract types
            contradictions = [
                Contradiction(
                    type=c.type,
                    lincoln_claim=c.lincoln_claim,
                    other_claim=c.other_claim,
                    explanation=c.explanation,
                )
                for c in parsed.contradictions
            ]

            return JudgmentResult(
                event_id=event_id,
                event_name=event_name,
                lincoln_doc_id=lincoln_doc_id,
                other_doc_id=other_doc_id,
                other_author=other_author,
                consistency_score=parsed.consistency_score,
                contradictions=contradictions,
                reasoning=parsed.reasoning,
            )

        except Exception as e:
            logger.error(f"LangChain judgment failed for {event_id}/{other_author}: {e}")
            # Fallback: return a neutral score with error info
            return JudgmentResult(
                event_id=event_id,
                event_name=event_name,
                lincoln_doc_id=lincoln_doc_id,
                other_doc_id=other_doc_id,
                other_author=other_author,
                consistency_score=50,
                contradictions=[],
                reasoning=f"LangChain error: {e}",
            )

    async def judge_pair_sync(self, **kwargs) -> JudgmentResult:
        """Synchronous wrapper for judge_pair."""
        import asyncio
        return await self.judge_pair(**kwargs)


# -----------------------------------------------
# Integration with existing Part 3 pipeline
# -----------------------------------------------
async def run_langchain_judge(
    extractions_path: str = "data/extractions/extractions_cot.json",
    output_path: str = "data/judgments",
    strategy: str = "cot",
    temperature: float = 0.0,
    run_id: str = None,
    model: str = "gpt-4o-mini",
) -> JudgmentCollection:
    """
    Run the LangChain judge on all comparison pairs.
    
    Drop-in replacement for src.part3_judge.run.run_judge()
    that uses LangChain instead of direct LLM calls.
    """
    import uuid
    from datetime import datetime, timezone
    from src.contracts.extraction import ExtractionCollection
    from src.part3_judge.judge import build_comparison_pairs

    run_id = run_id or f"langchain_{uuid.uuid4().hex[:8]}"

    logger.info("=" * 60)
    logger.info("PART 3: LLM JUDGE (LangChain)")
    logger.info("=" * 60)

    # Load extractions
    logger.info(f"Loading extractions from {extractions_path}...")
    extractions = ExtractionCollection.from_file(extractions_path)
    pairs = build_comparison_pairs(extractions)
    logger.info(f"Found {len(pairs)} comparison pairs")

    # Initialize LangChain judge
    judge = LangChainJudge(
        model=model,
        temperature=temperature,
        prompt_strategy=strategy,
    )
    logger.info(f"LangChain Judge: model={model}, strategy={strategy}, temp={temperature}")

    # Run judgments
    events_map = {}
    for pair in pairs:
        event_id = pair["event_id"]
        if event_id not in events_map:
            events_map[event_id] = {
                "event_name": pair["event_name"],
                "judgments": [],
            }

        logger.info(f"  {pair['event_name']} — Lincoln vs {pair['other_author']}")

        result = await judge.judge_pair(
            event_name=pair["event_name"],
            event_id=event_id,
            lincoln_claims=pair["lincoln_claims"],
            other_claims=pair["other_claims"],
            lincoln_doc_id=pair["lincoln_doc_id"],
            other_doc_id=pair["other_doc_id"],
            other_author=pair["other_author"],
            temperature=temperature,
        )

        events_map[event_id]["judgments"].append(result)
        logger.info(
            f"    → Score: {result.consistency_score}, "
            f"Contradictions: {len(result.contradictions)}"
        )

    # Build collection
    events = [
        EventJudgments(
            event_id=eid,
            event_name=data["event_name"],
            judgments=data["judgments"],
        )
        for eid, data in events_map.items()
    ]

    collection = JudgmentCollection(
        judged_at=datetime.now(timezone.utc),
        llm_provider="openai",
        llm_model=model,
        prompt_strategy=strategy,
        temperature=temperature,
        run_id=run_id,
        events=events,
    )

    # Save
    output_file = f"{output_path}/judgments_{strategy}_{run_id}.json"
    collection.to_file(output_file)
    logger.info(f"Saved to {output_file}")

    return collection

"""
Judgment Schema - Contract between Part 3 Judge and Part 4 Report.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime, timezone


class Contradiction(BaseModel):
    """A specific inconsistency between claim sets."""
    type: Literal["factual", "interpretive", "omission"]
    lincoln_claim: Optional[str] = Field(
        default=None,
        description="Lincoln's claim (None for omission by Lincoln)"
    )
    other_claim: Optional[str] = Field(
        default=None,
        description="Other author's claim (None for omission by other)"
    )
    explanation: str = Field(description="Why this is a contradiction")


class JudgmentResult(BaseModel):
    """Comparison result for one (Lincoln doc, Other doc, Event) triple."""
    event_id: str
    event_name: str
    lincoln_doc_id: str
    other_doc_id: str
    other_author: str
    
    consistency_score: int = Field(ge=0, le=100)
    contradictions: list[Contradiction] = Field(default_factory=list)
    reasoning: Optional[str] = Field(
        default=None,
        description="LLM reasoning (for CoT strategy)"
    )


class EventJudgments(BaseModel):
    """All judgments for a single event."""
    event_id: str
    event_name: str
    judgments: list[JudgmentResult] = Field(default_factory=list)


class JudgmentCollection(BaseModel):
    """Complete judgment output for a single run."""
    judged_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    llm_provider: str
    llm_model: str
    prompt_strategy: str
    temperature: float
    run_id: str = Field(description="Unique identifier for this run")
    
    events: list[EventJudgments] = Field(default_factory=list)
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
    
    def to_file(self, path: str) -> None:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2))
    
    @classmethod
    def from_file(cls, path: str) -> "JudgmentCollection":
        from pathlib import Path
        return cls.model_validate_json(Path(path).read_text())


# --- Experiment Results ---

class ManualLabel(BaseModel):
    """Human label for Cohen's Kappa."""
    event_id: str
    lincoln_doc_id: str
    other_doc_id: str
    other_author: str
    label: Literal["consistent", "contradictory"]


class KappaResult(BaseModel):
    """Cohen's Kappa calculation result."""
    manual_labels: list[bool]  # True = consistent
    llm_labels: list[bool]
    kappa: float
    agreement_pct: float
    interpretation: str  # "slight", "fair", "moderate", "substantial", "perfect"


class SelfConsistencyResult(BaseModel):
    """Variance analysis for one pair across multiple runs."""
    event_id: str
    other_author: str
    scores: list[int]
    mean: float
    std_dev: float


class ExperimentSummary(BaseModel):
    """Summary of all Part 3 experiments."""
    ablation: dict[str, float]  # strategy -> avg score
    self_consistency: list[SelfConsistencyResult]
    kappa: KappaResult
"""
Experiments - Ablation, Self-Consistency, Cohen's Kappa calculations.
"""

import statistics
from typing import Literal

from src.contracts.judgment import (
    JudgmentCollection, KappaResult, SelfConsistencyResult, ManualLabel
)


# Manual labels from human evaluation (all consistent)
MANUAL_LABELS: list[ManualLabel] = [
    ManualLabel(event_id="fort_sumter", lincoln_doc_id="loc_mal0882800", 
                other_doc_id="gutenberg_6811", other_author="Ketcham", label="consistent"),
    ManualLabel(event_id="fort_sumter", lincoln_doc_id="loc_mal0882800",
                other_doc_id="gutenberg_12801", other_author="Morse", label="consistent"),
    ManualLabel(event_id="fort_sumter", lincoln_doc_id="loc_mal0882800",
                other_doc_id="gutenberg_14004", other_author="Browne", label="consistent"),
    ManualLabel(event_id="fort_sumter", lincoln_doc_id="loc_mal0882800",
                other_doc_id="gutenberg_18379", other_author="Charnwood", label="consistent"),
    ManualLabel(event_id="gettysburg_address", lincoln_doc_id="loc_gettysburg",
                other_doc_id="gutenberg_6811", other_author="Ketcham", label="consistent"),
    ManualLabel(event_id="gettysburg_address", lincoln_doc_id="loc_gettysburg",
                other_doc_id="gutenberg_12801", other_author="Morse", label="consistent"),
    ManualLabel(event_id="gettysburg_address", lincoln_doc_id="loc_gettysburg",
                other_doc_id="gutenberg_14004", other_author="Browne", label="consistent"),
    ManualLabel(event_id="gettysburg_address", lincoln_doc_id="loc_gettysburg",
                other_doc_id="gutenberg_18379", other_author="Charnwood", label="consistent"),
    ManualLabel(event_id="second_inaugural", lincoln_doc_id="loc_mal4361300",
                other_doc_id="gutenberg_6811", other_author="Ketcham", label="consistent"),
    ManualLabel(event_id="second_inaugural", lincoln_doc_id="loc_mal4361300",
                other_doc_id="gutenberg_12801", other_author="Morse", label="consistent"),
    ManualLabel(event_id="second_inaugural", lincoln_doc_id="loc_mal4361300",
                other_doc_id="gutenberg_14004", other_author="Browne", label="consistent"),
    ManualLabel(event_id="second_inaugural", lincoln_doc_id="loc_mal4361300",
                other_doc_id="gutenberg_18379", other_author="Charnwood", label="consistent"),
]


def calculate_kappa(
    judgments: JudgmentCollection,
    threshold: int = 50
) -> KappaResult:
    """
    Calculate Cohen's Kappa between LLM judgments and manual labels.
    
    Args:
        judgments: LLM judgment results
        threshold: Score >= threshold is "consistent"
    """
    manual_labels = []
    llm_labels = []
    
    for manual in MANUAL_LABELS:
        # Find matching LLM judgment
        for event in judgments.events:
            if event.event_id != manual.event_id:
                continue
            for j in event.judgments:
                if j.other_doc_id == manual.other_doc_id:
                    manual_labels.append(manual.label == "consistent")
                    llm_labels.append(j.consistency_score >= threshold)
                    break
    
    if len(manual_labels) == 0:
        return KappaResult(
            manual_labels=[], llm_labels=[],
            kappa=0.0, agreement_pct=0.0, interpretation="no data"
        )
    
    # Calculate agreement
    agreements = sum(m == l for m, l in zip(manual_labels, llm_labels))
    agreement_pct = agreements / len(manual_labels) * 100
    
    # Cohen's Kappa calculation
    kappa = _cohens_kappa(manual_labels, llm_labels)
    
    # Interpret kappa
    if kappa < 0:
        interpretation = "poor"
    elif kappa < 0.21:
        interpretation = "slight"
    elif kappa < 0.41:
        interpretation = "fair"
    elif kappa < 0.61:
        interpretation = "moderate"
    elif kappa < 0.81:
        interpretation = "substantial"
    else:
        interpretation = "perfect"
    
    return KappaResult(
        manual_labels=manual_labels,
        llm_labels=llm_labels,
        kappa=kappa,
        agreement_pct=agreement_pct,
        interpretation=interpretation,
    )


def _cohens_kappa(labels1: list[bool], labels2: list[bool]) -> float:
    """Calculate Cohen's Kappa for two binary label lists."""
    n = len(labels1)
    if n == 0:
        return 0.0
    
    # Observed agreement
    po = sum(l1 == l2 for l1, l2 in zip(labels1, labels2)) / n
    
    # Expected agreement by chance
    p1_true = sum(labels1) / n
    p2_true = sum(labels2) / n
    pe = (p1_true * p2_true) + ((1 - p1_true) * (1 - p2_true))
    
    # Handle edge case: perfect expected agreement
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    
    kappa = (po - pe) / (1 - pe)
    return round(kappa, 4)


def calculate_self_consistency(
    runs: list[JudgmentCollection]
) -> list[SelfConsistencyResult]:
    """
    Calculate score variance across multiple runs.
    
    Args:
        runs: Multiple judgment runs (same prompt, different seeds)
    """
    if not runs:
        return []
    
    # Build map: (event_id, other_doc_id) -> list of scores
    score_map: dict[tuple, list[int]] = {}
    
    for run in runs:
        for event in run.events:
            for j in event.judgments:
                key = (j.event_id, j.other_doc_id, j.other_author)
                if key not in score_map:
                    score_map[key] = []
                score_map[key].append(j.consistency_score)
    
    # Calculate stats for each pair
    results = []
    for (event_id, other_doc_id, other_author), scores in score_map.items():
        if len(scores) < 2:
            continue
        results.append(SelfConsistencyResult(
            event_id=event_id,
            other_author=other_author,
            scores=scores,
            mean=round(statistics.mean(scores), 2),
            std_dev=round(statistics.stdev(scores), 2),
        ))
    
    return results


def calculate_ablation_summary(
    results: dict[str, JudgmentCollection]
) -> dict[str, dict]:
    """
    Summarize ablation results across prompt strategies.
    
    Args:
        results: strategy name -> JudgmentCollection
    
    Returns:
        Dict with avg_score, std_dev per strategy
    """
    summary = {}
    
    for strategy, judgments in results.items():
        scores = []
        for event in judgments.events:
            for j in event.judgments:
                scores.append(j.consistency_score)
        
        if scores:
            summary[strategy] = {
                "avg_score": round(statistics.mean(scores), 2),
                "std_dev": round(statistics.stdev(scores) if len(scores) > 1 else 0, 2),
                "min": min(scores),
                "max": max(scores),
                "n": len(scores),
            }
    
    return summary
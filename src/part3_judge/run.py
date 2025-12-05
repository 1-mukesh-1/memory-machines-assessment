"""
Part 3 Entry Point - Run judge and experiments.

Usage:
    python -m src.part3_judge.run
    python -m src.part3_judge.run --experiments
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from src.contracts.extraction import ExtractionCollection
from src.contracts.judgment import JudgmentCollection, EventJudgments
from src.part2_extraction.llm import get_llm
from src.part3_judge.judge import JudgeEngine, build_comparison_pairs
from src.part3_judge.experiments import (
    calculate_kappa, calculate_self_consistency, calculate_ablation_summary
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


async def run_judge(
    extractions_path: str = "data/extractions/extractions_cot.json",
    output_path: str = "data/judgments",
    strategy: str = "cot",
    temperature: float = 0.0,
    run_id: str = None,
) -> JudgmentCollection:
    """Run judge on all comparison pairs."""
    
    run_id = run_id or str(uuid.uuid4())[:8]
    
    logger.info("=" * 60)
    logger.info("PART 3: LLM JUDGE")
    logger.info("=" * 60)
    
    # Load extractions
    logger.info("\n[1/3] Loading extractions...")
    extractions = ExtractionCollection.from_file(extractions_path)
    pairs = build_comparison_pairs(extractions)
    logger.info(f"Found {len(pairs)} comparison pairs")
    
    # Initialize judge
    llm = get_llm()
    logger.info(f"Using LLM: {llm.provider}/{llm.model}")
    logger.info(f"Strategy: {strategy}, Temperature: {temperature}")
    
    judge = JudgeEngine(llm, prompt_strategy=strategy)
    
    # Run judgments
    logger.info("\n[2/3] Running judgments...")
    
    # Group by event
    events_map = {}
    for pair in pairs:
        event_id = pair["event_id"]
        if event_id not in events_map:
            events_map[event_id] = {
                "event_name": pair["event_name"],
                "judgments": []
            }
        
        logger.info(f"  {pair['event_name']} - Lincoln vs {pair['other_author']}")
        
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
        logger.info(f"    → Score: {result.consistency_score}, Contradictions: {len(result.contradictions)}")
    
    # Build collection
    events = [
        EventJudgments(
            event_id=eid,
            event_name=data["event_name"],
            judgments=data["judgments"]
        )
        for eid, data in events_map.items()
    ]
    
    collection = JudgmentCollection(
        judged_at=datetime.now(timezone.utc),
        llm_provider=llm.provider,
        llm_model=llm.model,
        prompt_strategy=strategy,
        temperature=temperature,
        run_id=run_id,
        events=events,
    )
    
    # Save
    logger.info("\n[3/3] Saving results...")
    output_file = f"{output_path}/judgments_{strategy}_{run_id}.json"
    collection.to_file(output_file)
    logger.info(f"✓ Saved to {output_file}")
    
    return collection


async def run_experiments(
    extractions_path: str = "data/extractions/extractions_cot.json",
    output_path: str = "data/judgments",
) -> dict:
    """Run all experiments: ablation, self-consistency, kappa."""
    
    logger.info("=" * 60)
    logger.info("PART 3: EXPERIMENTS")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. Ablation Study (3 strategies)
    logger.info("\n[1/3] ABLATION STUDY")
    logger.info("-" * 40)
    
    ablation_results = {}
    for strategy in ["zero_shot", "cot", "few_shot"]:
        logger.info(f"\nRunning strategy: {strategy}")
        judgments = await run_judge(
            extractions_path=extractions_path,
            output_path=output_path,
            strategy=strategy,
            temperature=0.0,
            run_id=f"ablation_{strategy}",
        )
        ablation_results[strategy] = judgments
    
    ablation_summary = calculate_ablation_summary(ablation_results)
    logger.info("\nAblation Summary:")
    for strategy, stats in ablation_summary.items():
        logger.info(f"  {strategy}: avg={stats['avg_score']}, std={stats['std_dev']}")
    
    results["ablation"] = ablation_summary
    
    # 2. Self-Consistency (5 runs with temp=0.7)
    logger.info("\n[2/3] SELF-CONSISTENCY")
    logger.info("-" * 40)
    
    consistency_runs = []
    for i in range(5):
        logger.info(f"\nRun {i+1}/5 (temperature=0.7)")
        judgments = await run_judge(
            extractions_path=extractions_path,
            output_path=output_path,
            strategy="cot",
            temperature=0.7,
            run_id=f"consistency_run{i+1}",
        )
        consistency_runs.append(judgments)
    
    consistency_results = calculate_self_consistency(consistency_runs)
    logger.info("\nSelf-Consistency Results:")
    for r in consistency_results:
        logger.info(f"  {r.event_id}/{r.other_author}: mean={r.mean}, std={r.std_dev}")
    
    results["self_consistency"] = [r.model_dump() for r in consistency_results]
    
    # 3. Cohen's Kappa
    logger.info("\n[3/3] COHEN'S KAPPA")
    logger.info("-" * 40)
    
    # Use the CoT ablation run for kappa
    kappa_result = calculate_kappa(ablation_results["cot"])
    logger.info(f"Agreement: {kappa_result.agreement_pct}%")
    logger.info(f"Kappa: {kappa_result.kappa} ({kappa_result.interpretation})")
    
    results["kappa"] = kappa_result.model_dump()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENTS COMPLETE")
    logger.info("=" * 60)
    
    return results


async def run(experiments: bool = False):
    """Main entry point."""
    if experiments:
        return await run_experiments()
    else:
        return await run_judge()


if __name__ == "__main__":
    import sys
    experiments = "--experiments" in sys.argv
    asyncio.run(run(experiments=experiments))
"""
CLI Entry Point - Trigger pipeline stages from command line.

Usage:
    python -m src.cli part1           # Run Part 1 only
    python -m src.cli part2           # Run Part 2 only
    python -m src.cli part2 --all     # Run Part 2 with all strategies
    python -m src.cli all             # Run all parts
"""

import asyncio
import typer

app = typer.Typer(help="Lincoln Historiographical Divergence Pipeline")


@app.command()
def part1():
    """Run Part 1: Data Acquisition & Normalization."""
    from src.part1_acquisition.run import run
    asyncio.run(run())


@app.command()
def part2(
    strategy: str = typer.Option("cot", help="Prompt strategy: zero_shot, cot, few_shot"),
    all_strategies: bool = typer.Option(False, "--all", help="Run all strategies"),
):
    """Run Part 2: Event Extraction."""
    from src.part2_extraction.run import run, run_all_strategies
    from src.part2_extraction.config import Part2Config
    
    if all_strategies:
        asyncio.run(run_all_strategies())
    else:
        config = Part2Config(prompt_strategy=strategy)
        asyncio.run(run(config))


@app.command()
def part3():
    """Run Part 3: LLM Judge. (Not implemented)"""
    typer.echo("Part 3 not yet implemented")
    raise typer.Exit(1)


@app.command()
def part4():
    """Run Part 4: Reporting. (Not implemented)"""
    typer.echo("Part 4 not yet implemented")
    raise typer.Exit(1)


@app.command()
def all():
    """Run complete pipeline (Parts 1-4)."""
    typer.echo("Running complete pipeline...")
    part1()
    part2(strategy="cot", all_strategies=False)
    # part3(), part4() when implemented


if __name__ == "__main__":
    app()
"""
CLI Additions — New commands for RAG and LangChain judge.

Add these commands to your existing src/cli.py file.

Usage:
    python -m src.cli ingest-rag          # Embed documents into Pinecone
    python -m src.cli ask "your question"  # RAG Q&A
    python -m src.cli part3-langchain      # Run judge with LangChain
"""

import asyncio
import typer

# Create a new Typer group (or add to existing app)
rag_app = typer.Typer(help="RAG & LangChain features")


@rag_app.command("ingest-rag")
def ingest_rag(
    force: bool = typer.Option(False, "--force", help="Re-ingest even if vectors exist"),
):
    """Embed all Lincoln documents into Pinecone for RAG."""
    from src.rag.vectorstore import VectorStoreManager

    manager = VectorStoreManager()

    if not force:
        try:
            stats = manager.index_stats()
            count = stats.get("total_vector_count", 0)
            if count > 0:
                typer.echo(f"Index already has {count} vectors. Use --force to re-ingest.")
                return
        except Exception:
            pass

    count = manager.ingest_all("data")
    typer.echo(f"✓ Ingested {count} chunks into Pinecone")


@rag_app.command("ask")
def ask_question(
    question: str = typer.Argument(..., help="Question about Lincoln"),
    source: str = typer.Option(None, help="Filter: 'loc' or 'gutenberg'"),
    compare: bool = typer.Option(False, "--compare", help="Compare across sources"),
):
    """Ask a question about Lincoln using RAG."""
    from src.rag.qa_chain import LincolnQAChain

    qa = LincolnQAChain()

    if compare:
        result = qa.compare(question)
    else:
        result = qa.ask(question, source_filter=source)

    typer.echo(f"\n{'='*60}")
    typer.echo(f"Question: {result['question']}")
    typer.echo(f"{'='*60}\n")
    typer.echo(result["answer"])
    typer.echo(f"\n{'='*60}")
    typer.echo(f"Sources ({len(result['sources'])}):")
    for s in result["sources"]:
        icon = "📜" if s["source"] == "loc" else "📖"
        typer.echo(f"  {icon} {s['doc_id']} — {s['author']}")


@rag_app.command("part3-langchain")
def part3_langchain(
    strategy: str = typer.Option("cot", help="Prompt strategy: zero_shot, cot, few_shot"),
    temperature: float = typer.Option(0.0, help="LLM temperature"),
    model: str = typer.Option("gpt-4o-mini", help="OpenAI model"),
    experiments: bool = typer.Option(False, "--experiments", help="Run ablation + consistency"),
):
    """Run Part 3 judge using LangChain (structured output)."""
    from src.part3_judge.langchain_judge import run_langchain_judge

    if experiments:
        # Run ablation across strategies
        for strat in ["zero_shot", "cot", "few_shot"]:
            typer.echo(f"\n--- Running {strat} ---")
            asyncio.run(run_langchain_judge(
                strategy=strat,
                temperature=0.0,
                run_id=f"langchain_ablation_{strat}",
                model=model,
            ))
        # Self-consistency runs
        for i in range(5):
            typer.echo(f"\n--- Consistency run {i+1}/5 ---")
            asyncio.run(run_langchain_judge(
                strategy="cot",
                temperature=0.7,
                run_id=f"langchain_consistency_run{i+1}",
                model=model,
            ))
    else:
        asyncio.run(run_langchain_judge(
            strategy=strategy,
            temperature=temperature,
            model=model,
        ))


# -----------------------------------------------
# To integrate with existing cli.py, add this
# at the bottom of your src/cli.py:
#
#   from src.cli_langchain import rag_app
#   app.add_typer(rag_app, name="rag")
#
# Then usage becomes:
#   python -m src.cli rag ingest-rag
#   python -m src.cli rag ask "What about Fort Sumter?"
#   python -m src.cli rag part3-langchain --experiments
# -----------------------------------------------

if __name__ == "__main__":
    rag_app()

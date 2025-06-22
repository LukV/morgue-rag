"""CLI interface for Rue Morgue Revisited benchmark."""

import random
from typing import Any

import numpy as np
import typer
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from rich.console import Console
from rich.panel import Panel

from config import settings
from graph_rag import GraphRAG
from prompts import COMPARISON_TEMPLATE
from rag import TraditionalRAG

# Load environment variables
load_dotenv()

app = typer.Typer(
    help="🕵️ Rue Morgue Revisited: Compare RAG vs GraphRAG for murder mystery solving",
    rich_markup_mode="rich",
)
console = Console()


def set_seed(seed: int) -> None:
    """Seed all randomness sources."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002


def display_response(result: dict[str, Any], title: str) -> None:
    """Display the response in a formatted panel."""
    panel = Panel(
        result["response"],
        title=f"[bold]{title}[/bold]",
        title_align="left",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)

    if settings.verbose:
        if "retrieved_chunks" in result:
            console.print("\n[dim]Retrieved Chunks:[/dim]")
            if not result["retrieved_chunks"]:
                console.print("[red]No chunks retrieved.[/red]")
            for i, chunk in enumerate(result["retrieved_chunks"], 1):
                console.print(
                    f"[dim]{i}. {chunk['source']} (score: {chunk['score']:.3f})[/dim]"
                )
                console.print(f"   {chunk['text'][:200]}...")

        elif "retrieved_triples" in result:
            console.print("\n[dim]Retrieved Triples:[/dim]")
            if not result["retrieved_triples"]:
                console.print("[red]No triples retrieved.[/red]")
            for triple in result["retrieved_triples"][:10]:
                console.print(f"[dim]- {triple}[/dim]")


@app.command()
def rag(
    query: str = typer.Argument(
        "Who committed the murders?", help="The question to ask about the mystery"
    ),
    verbose: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--verbose",
        "-v",
        help="Show retrieved chunks and detailed output",
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Random seed for reproducibility"
    ),
) -> None:
    """Use traditional RAG (chunk-based retrieval) to solve the mystery."""
    console.print("[bold]🔍 Traditional RAG Approach[/bold]\n")

    settings.verbose = verbose
    if seed is not None:
        settings.seed = seed
        set_seed(seed)

    try:
        rag_system = TraditionalRAG()
        rag_system.load_documents()

        console.print(f"[yellow]Question:[/yellow] {query}\n")
        result = rag_system.query(query)
        display_response(result, "Traditional RAG Analysis")

    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@app.command()
def graph(
    query: str = typer.Argument(
        "Who committed the murders?", help="The question to ask about the mystery"
    ),
    verbose: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--verbose",
        "-v",
        help="Show retrieved triples and detailed output",
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Random seed for reproducibility"
    ),
) -> None:
    """Use GraphRAG (RDF knowledge graph) to solve the mystery."""
    console.print("[bold]🕸️ GraphRAG Approach[/bold]\n")

    settings.verbose = verbose
    if seed is not None:
        settings.seed = seed
        set_seed(seed)

    try:
        graph_system = GraphRAG()
        graph_system.load_graph()

        console.print(f"[yellow]Question:[/yellow] {query}\n")
        result = graph_system.query(query)
        display_response(result, "GraphRAG Analysis")

    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@app.command()
def compare(
    query: str = typer.Argument(
        "Who committed the murders?", help="The question to ask about the mystery"
    ),
    verbose: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--verbose",
        "-v",
        help="Show detailed output from both approaches",
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Random seed for reproducibility"
    ),
) -> None:
    """Compare RAG and GraphRAG approaches side by side."""
    console.print("[bold]⚖️ Comparing RAG vs GraphRAG[/bold]\n")

    # Update settings
    settings.verbose = verbose
    if seed is not None:
        settings.seed = seed

    try:
        console.print(f"[yellow]Question:[/yellow] {query}\n")

        # Run Traditional RAG
        console.print("[dim]Running Traditional RAG...[/dim]")
        rag_system = TraditionalRAG()
        rag_system.load_documents()
        rag_result = rag_system.query(query)

        # Run GraphRAG
        console.print("[dim]Running GraphRAG...[/dim]")
        graph_system = GraphRAG()
        graph_system.load_graph()
        graph_result = graph_system.query(query)

        # Display individual results
        console.print("\n" + "=" * 80 + "\n")
        display_response(rag_result, "Traditional RAG")
        console.print("\n" + "=" * 80 + "\n")
        display_response(graph_result, "GraphRAG")
        console.print("\n" + "=" * 80 + "\n")

        # Create comparison
        from llama_index.llms.ollama import Ollama

        from llama_index.llms.base import BaseLLM

        # Use configured LLM provider
        llm: BaseLLM
        if settings.llm_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key required for comparison")
            llm = OpenAI(
                model=settings.openai_model,
                temperature=0.0,
                api_key=settings.openai_api_key,
            )
        else:  # ollama
            llm = Ollama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0.0,
                context_window=8192,
                request_timeout=settings.ollama_timeout,
                additional_kwargs={
                    "keep_alive": settings.ollama_keep_alive,
                    "num_predict": settings.max_tokens,
                },
            )

        comparison_prompt = COMPARISON_TEMPLATE.format(
            question=query,
            rag_response=rag_result["response"],
            graph_response=graph_result["response"],
        )

        comparison = llm.complete(comparison_prompt)

        comparison_panel = Panel(
            comparison.text,
            title="[bold green]Comparative Analysis[/bold green]",
            title_align="left",
            border_style="green",
            padding=(1, 2),
        )
        console.print(comparison_panel)

    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@app.command()
def info() -> None:
    """Display information about the benchmark and configuration."""
    # Determine which embedding model is being used
    if settings.embedding_provider == "openai":
        embedding_info = f"OpenAI: {settings.openai_embedding_model}"
    else:
        embedding_info = f"HuggingFace: {settings.huggingface_embedding_model} (free)"

    # Determine which LLM is being used
    if settings.llm_provider == "openai":
        llm_info = f"OpenAI: {settings.openai_model}"
    else:
        llm_info = f"Ollama: {settings.ollama_model} (local)"

    info_text = f"""
[bold]🕵️ Rue Morgue Revisited Benchmark[/bold]

A creative benchmark comparing traditional RAG with GraphRAG using a
rewritten version of Edgar Allan Poe's "The Murders in the Rue Morgue"
where C. Auguste Dupin is the murderer.

[yellow]Current Configuration:[/yellow]
• LLM Provider: {settings.llm_provider}
• LLM Model: {llm_info}
• Embedding Provider: {settings.embedding_provider}
• Embedding Model: {embedding_info}
• Chunk Size: {settings.chunk_size}
• Top K: {settings.top_k}
• Data Directory: {settings.data_dir}

[yellow]Available Commands:[/yellow]
• [green]morgue-rag rag[/green] - Use traditional chunk-based RAG
• [green]morgue-rag graph[/green] - Use RDF knowledge graph approach
• [green]morgue-rag compare[/green] - Compare both approaches
• [green]morgue-rag info[/green] - Show this information

[dim]For completely free operation:
• Set LLM_PROVIDER=ollama (requires Ollama running locally)
• Set EMBEDDING_PROVIDER=huggingface (default)
• No API keys required![/dim]
"""
    console.print(Panel(info_text, title="🕵️ Benchmark Info", border_style="blue"))


if __name__ == "__main__":
    app()

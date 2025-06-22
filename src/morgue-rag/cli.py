from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(help="🕵️‍♀️ Detective Morgue")
console = Console()


@app.command()
def say(word: str) -> None:
    """Echo a word back in a friendly way."""
    typer.echo(f"{word} you say?")


if __name__ == "__main__":
    app()

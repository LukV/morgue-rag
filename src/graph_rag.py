"""GraphRAG implementation using RDF and SPARQL."""

from typing import Any

from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from rdflib import Graph, Namespace
from rich.console import Console

from config import settings
from prompts import REASONING_PROMPT_TEMPLATE

console = Console()


class GraphRAG:
    """Graph-based RAG using RDF knowledge graph."""

    def __init__(self) -> None:
        """Initialize GraphRAG components."""
        self.graph = Graph()
        
        # Configure LLM based on provider
        if settings.llm_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI LLM")
            self.llm = OpenAI(
                model=settings.openai_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                api_key=settings.openai_api_key,
                seed=settings.seed,
            )
        else:  # ollama
            self.llm = Ollama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=settings.temperature,
                context_window=8192,
                request_timeout=settings.ollama_timeout,
                additional_kwargs={
                    "keep_alive": settings.ollama_keep_alive,
                    "num_predict": settings.max_tokens,
                },
            )

        # Define namespaces
        self.MORGUE = Namespace("http://example.org/morgue#")
        self.graph.bind("morgue", self.MORGUE)

    def load_graph(self) -> None:
        """Load the RDF knowledge graph."""
        console.print("[yellow]Loading RDF knowledge graph...[/yellow]")

        # Parse the turtle file
        self.graph.parse(settings.rdf_file, format="turtle")

        # Count triples
        num_triples = len(self.graph)
        console.print(f"[green]Loaded {num_triples} RDF triples[/green]")

    def extract_relevant_subgraph(self, question: str) -> list[tuple[str, str, str]]:
        """Extract relevant triples based on the question."""
        # Keywords to search for in the question
        keywords = self._extract_keywords(question)

        relevant_triples = []

        # Query for triples containing keywords
        for s, p, o in self.graph:
            triple_text = f"{s} {p} {o}".lower()
            if any(keyword.lower() in triple_text for keyword in keywords):
                relevant_triples.append((str(s), str(p), str(o)))

        # Also get connected triples (one hop)
        subjects = {t[0] for t in relevant_triples}
        objects = {t[2] for t in relevant_triples if isinstance(t[2], str) and t[2].startswith("http")}

        for s, p, o in self.graph:
            if str(s) in subjects or str(s) in objects:
                triple = (str(s), str(p), str(o))
                if triple not in relevant_triples:
                    relevant_triples.append(triple)

        return relevant_triples[:settings.top_k * 3]  # Return more triples than chunks

    def _extract_keywords(self, question: str) -> list[str]:
        """Extract keywords from the question."""
        # Simple keyword extraction - can be enhanced
        keywords = ["murder", "dupin", "sailor", "orangutan", "motive", "alibi", "witness"]

        # Add words from the question
        words = question.lower().split()
        for word in words:
            if len(word) > 4 and word not in ["committed", "murders", "who's", "what's"]:
                keywords.append(word)

        return keywords

    def triples_to_natural_language(self, triples: list[tuple[str, str, str]]) -> str:
        """Convert RDF triples to natural language."""
        facts = []

        for s, p, o in triples:
            # Extract readable names from URIs
            subject = self._extract_name(s)
            predicate = self._extract_name(p)
            object_val = self._extract_name(o)

            # Convert to natural language based on predicate
            if "type" in predicate:
                facts.append(f"{subject} is a {object_val}")
            elif "name" in predicate:
                facts.append(f"{subject} is named {object_val}")
            elif "knows" in predicate:
                facts.append(f"{subject} knows {object_val}")
            elif "present" in predicate or "location" in predicate:
                facts.append(f"{subject} was present at {object_val}")
            elif "alibi" in predicate:
                facts.append(f"{subject} claims alibi: {object_val}")
            elif "motive" in predicate:
                facts.append(f"{subject} has motive: {object_val}")
            elif "owns" in predicate or "possesses" in predicate:
                facts.append(f"{subject} owns {object_val}")
            else:
                facts.append(f"{subject} {predicate} {object_val}")

        return "\n".join(facts)

    def _extract_name(self, uri: str) -> str:
        """Extract readable name from URI or return literal value."""
        if uri.startswith("http"):
            # Extract the fragment or last part of the URI
            if "#" in uri:
                return uri.split("#")[-1].replace("_", " ")
            return uri.split("/")[-1].replace("_", " ")
        return uri.strip('"')

    def query(self, question: str) -> dict[str, Any]:
        """Query the graph-based system with a question."""
        # Extract relevant subgraph
        relevant_triples = self.extract_relevant_subgraph(question)

        # Convert to natural language
        context = self.triples_to_natural_language(relevant_triples)

        # Create prompt with graph context
        prompt = f"""Based on the following facts from the knowledge graph:

{context}

{REASONING_PROMPT_TEMPLATE.format(question=question)}"""

        # Query the LLM
        response = self.llm.complete(prompt)

        return {
            "response": response.text,
            "retrieved_triples": relevant_triples,
            "context": context,
            "method": "graph_rag",
        }

    def run_sparql_query(self, sparql: str) -> list[dict[str, Any]]:
        """Run a SPARQL query on the graph."""
        results = []
        for row in self.graph.query(sparql):
            results.append({str(var): str(val) for var, val in row.asdict().items()})
        return results

"""Traditional RAG implementation using LlamaIndex."""

from typing import Any

from llama_index.core import Settings as LlamaSettings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from rich.console import Console

from config import settings
from prompts import REASONING_PROMPT_TEMPLATE

console = Console()


class TraditionalRAG:
    """Traditional chunk-based RAG implementation."""

    def __init__(self) -> None:
        """Initialize RAG components."""
        # Configure LLM based on provider
        if settings.llm_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI LLM")
            LlamaSettings.llm = OpenAI(
                model=settings.openai_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                api_key=settings.openai_api_key,
                seed=settings.seed,
            )
            console.print(f"[cyan]Using OpenAI LLM: {settings.openai_model}[/cyan]")
        else:  # ollama
            LlamaSettings.llm = Ollama(
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
            console.print(f"[cyan]Using Ollama LLM: {settings.ollama_model}[/cyan]")
            console.print(f"[dim]Timeout: {settings.ollama_timeout}s, Keep alive: {settings.ollama_keep_alive}[/dim]")
        
        # Configure embedding model based on provider
        if settings.embedding_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            LlamaSettings.embed_model = OpenAIEmbedding(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key,
            )
            console.print(f"[cyan]Using OpenAI embeddings: {settings.openai_embedding_model}[/cyan]")
        else:  # huggingface
            LlamaSettings.embed_model = HuggingFaceEmbedding(
                model_name=settings.huggingface_embedding_model,
                trust_remote_code=True,
            )
            console.print(f"[cyan]Using HuggingFace embeddings: {settings.huggingface_embedding_model}[/cyan]")

        self.index: VectorStoreIndex | None = None
        self.query_engine: RetrieverQueryEngine | None = None

    def load_documents(self) -> None:
        """Load and index the mystery narrative."""
        console.print("[yellow]Loading mystery narrative...[/yellow]")

        # Load all markdown files
        reader = SimpleDirectoryReader(
            input_dir=str(settings.markdown_dir),
            filename_as_id=True,
            recursive=False,
        )
        documents = reader.load_data()

        # Parse documents into nodes
        parser = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        nodes = parser.get_nodes_from_documents(documents)

        # Create index
        self.index = VectorStoreIndex(nodes)

        # Create retriever and query engine
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=settings.top_k,
        )

        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
        )

        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        console.print(f"[green]Indexed {len(nodes)} text chunks[/green]")

    def query(self, question: str) -> dict[str, Any]:
        """Query the RAG system with a question."""
        if not self.query_engine:
            raise ValueError("Documents not loaded. Call load_documents() first.")

        # Format the prompt with reasoning instructions
        formatted_prompt = REASONING_PROMPT_TEMPLATE.format(question=question)

        # Query the index
        response = self.query_engine.query(formatted_prompt)

        # Extract retrieved chunks for transparency
        retrieved_chunks = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                retrieved_chunks.append({
                    "text": node.node.get_content(),
                    "score": node.score,
                    "source": node.node.metadata.get("file_name", "unknown"),
                })

        return {
            "response": str(response),
            "retrieved_chunks": retrieved_chunks,
            "method": "traditional_rag",
        }

    def get_context_window(self, question: str) -> list[str]:
        """Get the context chunks that would be used for a query."""
        if not self.index:
            raise ValueError("Documents not loaded. Call load_documents() first.")

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=settings.top_k,
        )

        nodes = retriever.retrieve(question)
        return [node.node.get_content() for node in nodes]

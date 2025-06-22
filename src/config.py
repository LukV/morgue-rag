from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for the application."""

    # Provider Configuration
    embedding_provider: Literal["openai", "huggingface"] = Field(
        "huggingface", 
        description="Embedding provider to use (openai or huggingface)"
    )
    llm_provider: Literal["openai", "ollama"] = Field(
        "openai",
        description="LLM provider to use (openai or ollama)"
    )
    
    # API Configuration
    openai_api_key: str | None = Field(None, description="OpenAI API key (required for OpenAI provider)")
    openai_model: str = Field("gpt-4", description="OpenAI model name")
    openai_embedding_model: str = Field("text-embedding-3-small", description="OpenAI embedding model")
    
    # Ollama Configuration
    ollama_model: str = Field("deepseek-r1:latest", description="Ollama model name")
    ollama_base_url: str = Field("http://localhost:11434", description="Ollama API base URL")
    ollama_timeout: float = Field(300.0, description="Ollama request timeout in seconds")
    ollama_keep_alive: str = Field("5m", description="How long to keep model in memory")
    
    # HuggingFace Configuration
    huggingface_embedding_model: str = Field(
        "BAAI/bge-small-en-v1.5",
        description="HuggingFace embedding model (free, no API key required)"
    )

    # RAG Configuration
    chunk_size: int = Field(512, description="Text chunk size for RAG")
    chunk_overlap: int = Field(50, description="Overlap between chunks")
    top_k: int = Field(5, description="Number of chunks to retrieve")

    # Paths
    data_dir: Path = Field(Path("data"), description="Data directory path")
    markdown_dir: Path = Field(Path("data/markdown"), description="Markdown files directory")
    rdf_file: Path = Field(Path("data/rdf/rue_morgue_kg.ttl"), description="RDF knowledge graph file")
    vector_db_path: Path = Field(Path("data/.vectorstore"), description="Vector database path")

    # Model Parameters
    temperature: float = Field(0.0, description="Model temperature for deterministic output")
    max_tokens: int = Field(2000, description="Maximum tokens in response")
    seed: int | None = Field(None, description="Random seed for reproducibility")

    # Output Configuration
    verbose: bool = Field(False, description="Enable verbose output")

    @field_validator("data_dir", "markdown_dir", "rdf_file")
    @classmethod
    def validate_paths(cls, v: Path) -> Path:  # noqa: D102
        return v.absolute()

    class Config:
        """Configuration class for environment settings."""

        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings: Settings = Settings() # type: ignore[call-arg]

# 🐶 GraphRAGGLE

GraphRAGGLE is a prototype for exploring **Graph-Based Retrieval-Augmented Generation (GraphRAG)** with a real-world use case: *recognized books*.

Starting from a curated CSV seed list (title, author, recognition list, year, summary), it builds a small but semantically rich **knowledge graph**, enhances it with linked data from **Wikidata** and **StoryGraph**, and allows head-to-head comparison with traditional tabular RAG pipelines.

---

## ✨ Project Features

### 📚 Seed Data Ingestion
- Load a curated CSV of ~300 recognized books (Booker, Pulitzer, NYT Notables, etc.)
- Normalize book data into clean, semantically meaningful RDF triples
- Output to **Turtle (`.ttl`)** or **JSON-LD** for downstream KG processing

### 🧠 Graph Construction & Enrichment
- Construct RDF-based knowledge graphs using schema.org and custom namespaces
- Normalize recognitions (e.g., `nyt-100-notable`, `booker-prize`) as URIs
- Enrich the graph with:
  - 🔗 **Wikidata**: author info, genres, translations, awards
  - 📖 **Open Library**: subjects, covers, additional metadata
  - 🧬 **LLM-derived nodes**: moods, styles, fine-grained genres (e.g., "page-turner", "teenage angst")

### 🔁 Graph vs. Traditional RAG Comparison
- Generate classic RAG-ready text chunks from the same source
- Extract subgraphs from the KG to serve as retrieval context
- Evaluate and compare:
  - 🔍 Accuracy of retrieved facts
  - 🗣️ Explainability of answers
  - ✅ Validity of sources
  - 🔄 Stability across queries

### 🛠️ Developer Workflow
- Modular, `uv`-managed Python project with CLI
- Opinionated setup with:
  - `typer` for CLI
  - `rdflib` for RDF
  - `ruff`, `mypy`, `pre-commit`, `commitizen` for quality
- Easy to extend with additional transformers, enrichers, or RAG pipelines

---

## 🧪 Targeted Research Questions

- When and why does GraphRAG produce more *explainable* or *valid* answers than traditional chunked RAG?
- Can semantically sparse but **structured** graphs outperform noisy embedding search?
- Which node types (e.g., moods, prizes, author backgrounds) improve the **precision of retrieval**?
- Can LLMs perform meaningful inference when guided by KG paths?
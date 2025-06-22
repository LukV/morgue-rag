# Rue Morgue Revisited: A RAG vs. GraphRAG Benchmark

Can an LLM solve a murder mystery — and explain its reasoning?

## 🕵️ Overview

This project is a creative benchmark designed to compare traditional Retrieval-Augmented Generation (RAG) with GraphRAG, using a narrative approach grounded in classic detective fiction. Inspired by Edgar Allan Poe’s 1841 short story The Murders in the Rue Morgue, we reimagined the plot to test whether modern language models can not only identify the culprit, but explain their reasoning.

The result is a structured, inference-heavy benchmark where a large language model (LLM) must solve a subtly rewritten mystery, picking up on contradictions, motive, and clues — like a true detective.

## 🧠 Why This Benchmark?

Poe called his original story a “tale of ratiocination” — a story centered on deliberate, logical reasoning. That makes it a perfect fit for testing LLM-based systems’ ability to reason through:

- Multi-hop deduction
- Conflicting evidence
- Entity-based inference
- Red herrings

While both RAG and GraphRAG systems may arrive at the correct answer, we’re especially interested in explainability: can the model justify its answer in terms of facts and logic?

## 📚 The Setup

We created a new version of The Murders in the Rue Morgue in which the famous detective, C. Auguste Dupin, is secretly the murderer. The reader (and the LLM) is presented with the same mystery structure, but subtle inconsistencies in the narrative provide evidence of Dupin’s guilt.

## 🔧 Data Artifacts

- rue_morgue_revisited.md: The rewritten mystery narrative
- rue_morgue_kg.ttl: RDF knowledge graph in Turtle format (extracted from the story)

## 🧱 RAG vs. GraphRAG

We compare:

1. Traditional RAG

- Chunks the story text
- Embeds the chunks
- Retrieves and injects chunks into prompts
- LLM uses free-text to reason

2. GraphRAG

- Uses RDF triples from the knowledge graph
- Retrieves relevant facts via SPARQL or graph traversal
- Injects graph context into the prompt
- LLM reasons based on structured relationships

## ⚖️ Evaluation Focus: Explainability

Rather than measuring accuracy alone, we measure how well the system explains its answer:

- 🔍 Which clues were used?
- 🔗 Were they connected logically?
- ❌ Did the system rely on hallucinated or irrelevant facts?
- 📜 Does the explanation reflect real reasoning chains?

Each response is expected to follow a structured format:

```
**Conclusion**: [Character Name]
**Clues Used**:
1. ...
2. ...
**Reasoning Chain**:
Step-by-step explanation...
```

## 🧠 Knowledge Graph Construction

To create the knowledge graph:

	1.	We used an LLM to extract entities (characters, places, objects)
	2.	We manually reviewed and connected them as RDF triples
	3.	These were serialized as rue_morgue_kg.ttl

The graph allows queries like:

- Who was present near the crime scene?
- Who lacked an alibi?
- What items were linked to which characters?
- What motives were stated or implied?

## 🧨 Key Story Modifications

We carefully rewrote the story to plant two critical inference edges:

- Motive for Dupin: He believed Paris had become decadent, and loosed the orangutan as a form of ideological terror.
- Lack of alibi: Dupin separated from the narrator the night of the murders, and small inconsistencies point to him being present at the scene.

These clues are subtly planted in the narrative, not directly stated — forcing the model to reason rather than recall.

## 🚀 Usage

Use the CLI (morgue-rag) to run:

morgue-rag rag --query "Who committed the murders?"
morgue-rag graph --query "Who committed the murders?"
morgue-rag compare --query "Who committed the murders?"

Each command:

- Retrieves context (chunks or subgraph)
- Prompts the LLM with structured instructions
- Returns a conclusion + explanation

## 🧰 Technologies

- Typer CLI framework
- DuckDBfor RAG
- rdflib for RDF
- OpenAI for generation

## 📎 Reference
- Wikipedia – The Murders in the Rue Morgue
- Edgar Allan Poe – Original 1841 text via Project Gutenberg

⸻

Rue Morgue Revisited is part of ongoing research into explainable AI and narrative evaluation of language model reasoning.

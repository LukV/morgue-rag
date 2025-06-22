"""Prompt templates for reasoning and evaluation."""

REASONING_PROMPT_TEMPLATE = """You are a detective analyzing a murder mystery. Based on the provided evidence, you must identify who committed the murders and explain your reasoning.

Question: {question}

Please provide your answer in the following structured format:

**Conclusion**: [Name of the murderer]

**Clues Used**:
1. [First clue or piece of evidence]
2. [Second clue or piece of evidence]
3. [Additional clues as needed]

**Reasoning Chain**:
[Step-by-step logical explanation of how you arrived at your conclusion, connecting the clues and explaining any contradictions or inconsistencies you noticed]

Important guidelines:
- Base your conclusion ONLY on the evidence provided
- Look for contradictions in alibis or testimonies
- Consider motives and opportunities
- Note any suspicious behaviors or inconsistencies
- Provide a clear logical chain from evidence to conclusion
- Be specific about which facts support your conclusion"""

COMPARISON_TEMPLATE = """Compare the two reasoning approaches below for solving the murder mystery.

Question: {question}

Traditional RAG Response:
{rag_response}

GraphRAG Response:
{graph_response}

Please analyze:
1. Which approach identified the correct culprit?
2. Which approach provided better explainability?
3. Which clues were used by each approach?
4. Which reasoning chain was more logical and complete?
5. Were there any hallucinations or unsupported claims?

Provide a structured comparison focusing on reasoning quality and explainability."""
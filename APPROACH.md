# Approach Document: SHL Conversational Assessment Recommender

## 1. Problem Statement

Build a conversational AI agent that recommends SHL assessment products from a fixed catalog of 377 individual test solutions. The system must support multi-turn conversations including clarification, recommendation, refinement, comparison, and confirmation — while never hallucinating assessments that don't exist in the catalog.

## 2. Architecture Overview

The system follows a **Retrieval-Augmented Generation (RAG)** pattern with a stateless, agentic conversation loop:

```
POST /chat (messages[]) → State Reconstruction → Hybrid Retrieval → LLM Reasoning → Post-process Validation → JSON Response
```

### 2.1 Stateless Design

Every request contains the full conversation history. On each call, the agent:
1. Parses prior assistant messages to reconstruct the current shortlist
2. Classifies the user's latest intent (clarify / recommend / refine / compare / confirm / off-topic)
3. Builds a retrieval query from accumulated user context
4. Retrieves relevant catalog entries
5. Calls the LLM with grounding context
6. Validates every recommendation against the catalog before responding

### 2.2 Hybrid Retrieval

Two complementary retrieval methods are fused using **Reciprocal Rank Fusion (RRF)**:

- **Dense (Semantic)**: `all-MiniLM-L6-v2` sentence-transformer embeddings indexed in FAISS (Inner Product). Captures meaning — "hiring plant operators for a chemical facility" matches safety-focused assessments.
- **Sparse (Keyword)**: BM25 over tokenized catalog entries. Captures exact terms — "Java", "OPQ32r", "HIPAA" match directly.
- **Fusion**: RRF with 60% semantic / 40% keyword weighting merges both ranked lists into a single relevance-ordered result set.

This hybrid approach ensures high recall: semantic search catches conceptual matches while BM25 catches exact product names and technical terms.

### 2.3 LLM Integration

**Model**: Gemini 2.0 Flash (with flash-lite fallback)  
**Temperature**: 0.3 (low for deterministic, grounded output)  
**Output**: Structured JSON via `response_mime_type: application/json`

The LLM receives:
- A system prompt encoding behavioral rules (when to clarify vs recommend, how to handle comparisons, scope guards)
- Up to 20 retrieved catalog entries with full metadata
- The complete conversation history

### 2.4 Post-processing Validation

Every recommendation in the LLM's output is validated against the catalog:
- URL match → Name match → Fuzzy name search
- If no match found, the recommendation is **dropped** (never sent to the user)
- Test type codes are corrected from canonical catalog data
- Duplicates are removed

This layer makes hallucination **impossible** — even if the LLM invents an assessment name, it never reaches the user.

## 3. Behavioral Design

Patterns derived from analysis of 10 official conversation traces:

| Behavior | Implementation |
|----------|---------------|
| **Conditional clarification** | Only clarify when context is genuinely insufficient (< 15 words, vague signals). Rich queries get immediate recommendations. |
| **Recommendation persistence** | When user confirms, re-emit the full shortlist with `end_of_conversation: true`. |
| **Surgical refinement** | "Add X, drop Y" modifies the shortlist in-place rather than rebuilding. |
| **Grounded comparison** | Product comparisons use catalog metadata. Response has empty recommendations list. |
| **Push-back then defer** | Agent may explain why dropping an assessment is suboptimal, but honors the user's explicit request. |
| **Scope guard** | Refuses legal, regulatory, and off-topic requests with appropriate redirection. |

## 4. Resilience

- **Model fallback chain**: gemini-2.0-flash → gemini-2.0-flash-lite → deterministic fallback
- **Deterministic fallback**: Intent-aware responses using retrieval results directly (no LLM needed)
- **Schema guarantee**: Triple error handling ensures every response matches the required JSON schema
- **Fast failure**: SDK retries disabled; hard 12s timeout per model prevents slow responses

## 5. Technology Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI 0.115 |
| LLM | Google Gemini 2.0 Flash |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Index | FAISS (IndexFlatIP) |
| Keyword Index | BM25 (rank-bm25) |
| Schema Validation | Pydantic v2 |
| Deployment | Render (Docker, free tier) |

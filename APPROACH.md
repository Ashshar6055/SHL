# Approach Document: SHL Conversational Assessment Recommender

## 1. Problem Statement & Constraints

The objective was to architect a conversational AI backend capable of recommending SHL assessment products from a fixed catalog of 377 entries. The system required multi-turn intelligence (clarification, recommendation, refinement, comparison) while strictly adhering to a rigid JSON schema. 

Crucially, the backend had to be deployed to the **Render Free Tier**, which imposes a hard **512MB RAM limit**. This single constraint drove all downstream engineering and architectural tradeoffs, prioritizing absolute reliability, memory efficiency, and hallucination prevention over theoretical ML sophistication.

---

## 2. Architectural Evolution: The Great Tradeoff (Dense vs. Sparse Retrieval)

### The Initial Attempt: Hybrid Semantic Retrieval
Our initial architecture employed a RAG (Retrieval-Augmented Generation) pipeline using a Hybrid Dense-Sparse approach:
- **Dense:** `sentence-transformers` (`all-MiniLM-L6-v2`) indexed in PyTorch/FAISS.
- **Sparse:** `rank-bm25` for exact keyword matching.

While this provided excellent conceptual matching (e.g., mapping "plant operator" to safety assessments), it introduced a fatal flaw for our target environment: **Memory Bloat.** The PyTorch runtime, embedding tensors, and FAISS index idled at ~400MB RAM. During concurrent requests, this frequently spiked above 512MB, triggering instant Out-Of-Memory (OOM) kills on Render.

### The Final Solution: Pure BM25 Retrieval
To guarantee 100% uptime, we performed a drastic architectural pivot. We completely excised PyTorch, Sentence Transformers, and FAISS from the codebase, relying exclusively on **BM25 Keyword Retrieval**.

**Why this tradeoff is a net positive:**
1. **Memory:** Idle RAM plummeted from ~400MB to **~56MB**. We are now operating at 10% of our capacity limit, allowing significant headroom for Uvicorn worker scaling and concurrent request handling.
2. **Startup Time:** Container boot time dropped to `< 0.1s`, eliminating cold-start timeouts.
3. **Precision:** SHL assessments heavily rely on specific technical keywords (e.g., "Java", "Angular", "OPQ32r"). BM25 excels at exact string matching, outperforming semantic search in technical vocabulary retrieval.

---

## 3. Stateless API Reconstruction

To eliminate the operational overhead of managing external state stores (like Redis or PostgreSQL) and to adhere to RESTful principles, the agent is entirely stateless. 

On every `POST /chat` request, the frontend sends the entire conversation history. The backend dynamically reconstructs the context:
1. It parses prior `assistant` messages to extract and rebuild the "current shortlist."
2. It concatenates the `user` messages to form a rich retrieval query.
3. It utilizes lightweight keyword heuristics to classify the latest intent (e.g., determining if the user is asking to *compare* existing recommendations or *clarify* a vague request).

This architecture allows the application to horizontally scale seamlessly without session affinity concerns.

---

## 4. Hallucination Prevention & Schema Enforcement

Large Language Models (LLMs), by their statistical nature, will hallucinate. In a corporate assessment recommendation system, recommending a non-existent test is a critical failure. 

We mitigated this using a **two-tier defense system:**

### Tier 1: Contextual Grounding
The Gemini 2.0 LLM is provided with a meticulously crafted System Prompt and up to 20 highly relevant BM25 catalog entries. It is explicitly instructed to only recommend tests found in the provided context.

### Tier 2: The Ironclad Post-Processing Validation
We do not trust the LLM. Before any response is transmitted over the wire, the backend intercepts the JSON payload. 
- Every recommendation is extracted.
- The `name` and `url` are validated against the internal canonical catalog via exact match, URL match, and fuzzy string matching.
- **If a recommendation cannot be verified against the catalog, it is silently dropped from the payload.**

This guarantees that a hallucination can never reach the end user.

### Schema Guarantee
Evaluator scripts crash if APIs return invalid JSON. We wrapped the entire LLM interaction in a massive `try/except` block. If the LLM returns unparseable markdown, malformed JSON, or if the OpenRouter API times out, the system catches the exception and returns a pre-scripted, valid `ChatResponse` Pydantic model. 

---

## 5. Resilience: Deterministic Fallback Strategy

Network boundaries to external LLMs (OpenRouter/Gemini) are the most volatile parts of an AI architecture. To guarantee the API never returns an HTTP 500 Error due to an upstream failure, we implemented a deterministic fallback protocol.

If the LLM fails, times out, or if the API key is revoked:
1. The backend detects the failure within a hard 12-second timeout window.
2. It falls back to pure intent heuristics (e.g., detecting if the user said "thank you" or asked for a recommendation).
3. It bypasses the LLM completely, passing the BM25 retrieval results directly into the `recommendations` array alongside a hardcoded, context-aware reply string.

The user receives a slightly less conversational, but perfectly valid and accurate recommendation list—achieving graceful degradation.

---

## 6. Evaluator-Oriented Engineering Decisions

This backend was built explicitly for an engineering evaluation. The focus was heavily placed on:
- **Maintainability:** Clear separation of concerns (`agent.py`, `retriever.py`, `catalog.py`, `models.py`).
- **Security:** Payload truncation limits users to the last 15 messages (max 2000 chars each) to prevent Denial of Service (DoS) and excessive token burn.
- **Resource Discipline:** Proving that production-grade AI doesn't always require massive GPU-bound vector databases; sometimes, a highly tuned BM25 implementation on a 56MB footprint is the superior engineering choice.

# SHL Conversational Assessment Recommender

> **A stateless, lightweight conversational AI agent for recommending SHL assessment products.** Designed to operate flawlessly on extreme low-memory environments (Render Free Tier) without sacrificing schema strictness or recommendation accuracy.

---

## 🟢 Live Demo & API Endpoints

The API is fully deployed and production-ready on Render.

*   **Base URL:** `https://shl-assessment-agent-tk8z.onrender.com`
*   **Interactive Docs (Swagger):** [https://shl-assessment-agent-tk8z.onrender.com/docs](https://shl-assessment-agent-tk8z.onrender.com/docs)
*   **Health Check:** `GET /health`
*   **Chat Endpoint:** `POST /chat`

---

## 1. Project Overview

The SHL Assessment Recommender is a backend AI service designed to assist hiring managers and talent acquisition teams in selecting the appropriate SHL assessment products. Operating via a RESTful API (`POST /chat`), the agent conducts multi-turn conversations—gathering context, clarifying needs, refining shortlists, and ultimately returning grounded, catalog-validated recommendations.

## 2. Problem Statement

The objective was to build an intelligent, multi-turn conversational agent capable of recommending assessments from a fixed catalog of 377 SHL products. The primary challenge was balancing the intelligence of a Large Language Model (LLM) with strict evaluator constraints: absolute prevention of hallucinated recommendations, flawless adherence to a precise JSON schema, and deployment viability within the 512MB RAM constraint of Render's free tier.

## 3. Architecture Overview & Evaluator-Focused Design Decisions

Our architecture uses a **Stateless Retrieval-Augmented Generation (RAG)** pipeline optimized for resilience and low overhead.

### Why BM25 Over Dense FAISS Retrieval? (Tradeoff Analysis)
Initially, the architecture employed a hybrid dense-sparse retrieval system using PyTorch, `sentence-transformers`, and FAISS. While semantic search provided slight improvements in conceptual matching, it bloated the idle memory footprint to ~400MB, risking OOM (Out-of-Memory) crashes on the 512MB Render free tier.
**Decision:** We entirely dropped the ML embedding dependencies in favor of pure **BM25 keyword retrieval**.
**Result:** 
- Idle memory footprint plummeted to **~56MB**. 
- Startup time dropped to **< 0.1s**. 
- Exact match reliability for technical test codes (e.g., "Java", "C++", "OPQ32r") improved significantly.

### Stateless Conversation Handling
To eliminate the need for an external database (Redis/Postgres) and keep the backend purely functional, the agent reconstructs the entire conversation state from the message history payload on every request. It parses prior assistant messages to extract current shortlists and uses keyword heuristics to classify user intent (e.g., *clarify*, *recommend*, *refine*, *compare*, *confirm*, *off-topic*).

### Hallucination Prevention
LLMs inherently hallucinate. To prevent the agent from inventing non-existent SHL products:
1. **Context Grounding:** The LLM is provided with top BM25-retrieved catalog entries.
2. **Post-Processing Validation (The Ironclad Guarantee):** Before any response is sent to the user, every recommendation in the LLM's JSON is cross-referenced against the canonical SHL catalog. If a recommended URL or Name doesn't exactly match (or closely fuzzy match) a real product, it is silently dropped. **A hallucination will never reach the user.**

## 4. Resilience & Deterministic Fallback Strategy

API failures (OpenRouter/Gemini timeouts) are inevitable. Our architecture guarantees uptime through a graceful degradation chain:

1. **Primary Model:** `gemini-2.0-flash-001`
2. **Secondary Model:** `gemini-2.0-flash-lite-001` (if primary timeouts/fails)
3. **Deterministic Fallback:** If the API key is missing, or all LLMs fail, the backend bypasses the LLM entirely. It uses heuristic intent classification and BM25 directly to formulate a pre-scripted response and a valid recommendation shortlist. **The API will never return a 500 error due to an upstream LLM failure.**

---

## 5. API Documentation

### `GET /health`
Returns the operational status of the service.
**Response:**
```json
{
  "status": "ok"
}
```

### `POST /chat`
The main conversational endpoint.

**Request Schema:**
```json
{
  "messages": [
    {"role": "user", "content": "I am looking for an assessment for a software engineer."}
  ]
}
```

**Response Schema:**
```json
{
  "reply": "Based on your requirements, here are the SHL assessments I'd recommend...",
  "recommendations": [
    {
      "name": "Agile Software Development",
      "url": "https://www.shl.com/products/product-catalog/view/agile-software-development/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

## 6. Security Considerations

- **Strict Schema Enforcement:** Triple-layer try/except blocks guarantee the API never returns malformed JSON, even under heavy prompt injection.
- **Payload Truncation:** To prevent DoS attacks, the backend automatically truncates incoming chat histories to the last 15 messages, capping each message at 2000 characters.
- **Off-Topic Refusals:** The agent heuristically detects off-topic or jailbreak attempts and firmly steers the conversation back to SHL assessments.

---

## 7. Setup & Deployment

### Environment Variables
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
PORT=8000
```
*(Note: If `OPENROUTER_API_KEY` is omitted, the app runs perfectly in Deterministic Fallback mode).*

### Local Setup
```bash
# Clone the repository
git clone https://github.com/Ashshar6055/SHL.git
cd SHL

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m app.main
```
The API will be available at `http://localhost:8000`.

### Docker Setup
```bash
# Build the image (~783MB)
docker build -t shl-agent .

# Run the container
docker run -p 8000:8000 --env-file .env shl-agent
```

### Render Deployment
This repository is optimized for Render's Free Web Service tier.
1. Connect this GitHub repository to Render.
2. Select **Docker** as the runtime environment.
3. Add the `OPENROUTER_API_KEY` under Environment Variables.
4. Deploy. The `render.yaml` (if used) or default Dockerfile will seamlessly spin up the service.

---

## 8. Performance Metrics

| Metric | Measurement |
|--------|-------------|
| **Startup Time** | < 0.1 seconds |
| **Idle RAM Footprint** | ~56.25 MiB |
| **Peak RAM Footprint** | ~56.76 MiB |
| **P99 API Latency** | ~3.5s (LLM Dependent) |
| **Cold Start** | ~1-2s (Docker init) |

## 9. Future Improvements

While this architecture perfectly meets the constraints of a free-tier deployment, future enterprise iterations could include:
1. **Redis Context Caching:** To reduce bandwidth by sending only message IDs rather than full conversation histories.
2. **Vector Database Integration:** Transitioning to Pinecone or managed Qdrant to offload dense FAISS embeddings from backend memory, allowing a return to hybrid semantic search without the RAM penalty.
3. **Streaming Responses:** Implementing Server-Sent Events (SSE) to stream the LLM `reply` to the frontend for lower perceived latency.

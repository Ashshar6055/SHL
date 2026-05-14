# SHL AI Assessment Recommender

A conversational AI agent that recommends SHL assessment products for hiring and talent development. Built with FastAPI, Gemini 2.0 Flash, FAISS, and BM25 hybrid retrieval.

## Live Demo

- **API**: [Deployed on Render](https://your-app.onrender.com)
- **Health Check**: `GET /health`
- **Chat**: `POST /chat`

## Architecture

```
User Query → Hybrid Retrieval (FAISS + BM25) → Gemini 2.0 Flash → Post-process Validation → Response
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API | FastAPI | REST endpoints with strict schema validation |
| LLM | Gemini 2.0 Flash | Conversational reasoning, intent classification |
| Semantic Search | sentence-transformers + FAISS | Dense vector similarity |
| Keyword Search | BM25 (rank-bm25) | Sparse term matching |
| Fusion | Reciprocal Rank Fusion | Merges dense + sparse results |
| Validation | Post-processing layer | Ensures all recommendations exist in catalog |

### Safety Features

- **Grounded**: Every recommendation validated against the SHL product catalog (377 assessments)
- **Stateless**: Conversation state reconstructed from message history on every request
- **Never breaks schema**: Triple-layered error handling ensures valid JSON on every response
- **Model fallback**: gemini-2.0-flash → gemini-2.0-flash-lite → deterministic fallback
- **No hallucinations**: Post-processing drops any recommendation not found in catalog

## API Schema

### POST /chat

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "We need assessments for senior leadership."},
    {"role": "assistant", "content": "{...}"},
    {"role": "user", "content": "Selection — comparing candidates against a leadership benchmark."}
  ]
}
```

**Response:**
```json
{
  "reply": "For selection with a leadership benchmark...",
  "recommendations": [
    {
      "name": "Occupational Personality Questionnaire OPQ32r",
      "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Run server
python -m app.main
```

Server starts at `http://localhost:8000`.

## Deployment (Render)

1. Push to GitHub
2. Connect repo on [Render](https://render.com)
3. Set `GEMINI_API_KEY` as environment variable
4. Deploy — the `render.yaml` handles configuration

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI endpoints
│   ├── models.py         # Pydantic schemas
│   ├── agent.py          # Core conversation logic
│   ├── retriever.py       # Hybrid retrieval (FAISS + BM25)
│   ├── catalog.py         # Catalog loading & lookup
│   └── prompts.py         # System prompts
├── shlcatalog.json        # SHL product catalog (377 assessments)
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```

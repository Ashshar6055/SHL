"""
FastAPI application for the SHL Conversational Assessment Recommender.
Two endpoints: GET /health and POST /chat.
"""

import os


import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import ChatRequest, ChatResponse, HealthResponse
from app.catalog import catalog
from app.retriever import retriever
from app.agent import get_agent


# --- Startup / Shutdown ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize catalog, retriever, and agent on startup."""
    print("=" * 60)
    print("  SHL Assessment Recommender - Starting up...")
    print("=" * 60)

    start = time.time()

    # 1. Load catalog
    catalog.load()

    # 2. Build retriever indices
    retriever.build()

    # 3. Initialize agent (triggers LLM client setup)
    get_agent()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Startup complete in {elapsed:.1f}s")
    print(f"  Catalog: {len(catalog.entries)} assessments")
    print(f"  BM25 index: {len(retriever.entries)} documents")
    print(f"{'=' * 60}\n")

    yield

    print("Shutting down...")


# --- App ---

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational AI agent for recommending SHL assessment products",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Receives conversation history and returns a response with optional recommendations.
    """
    try:
        # Validate input
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages list cannot be empty")

        # Check last message is from user
        if request.messages[-1].role != "user":
            raise HTTPException(
                status_code=400,
                detail="Last message must be from user"
            )

        # Process
        agent = get_agent()
        start = time.time()
        response = agent.process(request)
        elapsed = time.time() - start

        print(f"[API] Processed in {elapsed:.2f}s | "
              f"Recs: {len(response.recommendations)} | "
              f"EOC: {response.end_of_conversation}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Unhandled error: {e}")
        traceback.print_exc()
        # NEVER return a broken response - always valid schema
        return ChatResponse(
            reply="I can help you find the right SHL assessments. Could you tell me about the role you're hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )


# --- Run ---

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)

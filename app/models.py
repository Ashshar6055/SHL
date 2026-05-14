"""
Pydantic models for the SHL Conversational Assessment Recommender API.
Strict schema enforcement — the evaluator will reject any deviation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Message(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., pattern="^(user|assistant)$", description="Must be 'user' or 'assistant'")
    content: str = Field(..., min_length=1, description="Message content")


class ChatRequest(BaseModel):
    """Incoming chat request with full conversation history."""
    messages: List[Message] = Field(..., min_length=1, description="Conversation history")

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "I am looking for an assessment for a software engineer."
                    }
                ]
            }
        }
    }


class Recommendation(BaseModel):
    """A single assessment recommendation from the catalog."""
    name: str = Field(..., description="Assessment name from catalog")
    url: str = Field(..., description="Catalog URL")
    test_type: str = Field(..., description="Test type code(s): A, B, C, D, E, K, P, S")


class ChatResponse(BaseModel):
    """
    Response from the chat endpoint.
    - reply: always non-empty string
    - recommendations: empty list [] when gathering context or refusing;
      list of 1-10 items when committing to a shortlist
    - end_of_conversation: true only when the agent considers the task complete
    """
    reply: str = Field(..., min_length=1, description="Agent's text response")
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="Empty when clarifying/refusing, 1-10 items when recommending"
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the conversation complete"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"


# --- Internal models (not part of API schema) ---

class CatalogEntry(BaseModel):
    """Internal representation of a catalog assessment."""
    entity_id: str
    name: str
    link: str
    job_levels: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    duration: str = ""
    remote: str = ""
    adaptive: str = ""
    description: str = ""
    keys: List[str] = Field(default_factory=list)

    @property
    def test_type_code(self) -> str:
        """Convert keys to single-letter test type codes."""
        key_map = {
            "Ability & Aptitude": "A",
            "Biodata & Situational Judgment": "B",
            "Competencies": "C",
            "Development & 360": "D",
            "Assessment Exercises": "E",
            "Knowledge & Skills": "K",
            "Personality & Behavior": "P",
            "Simulations": "S",
        }
        codes = []
        for k in self.keys:
            code = key_map.get(k)
            if code and code not in codes:
                codes.append(code)
        return ",".join(codes) if codes else "K"

    @property
    def search_text(self) -> str:
        """Combined text for embedding and BM25 indexing."""
        parts = [
            self.name,
            self.description,
            ", ".join(self.keys),
            ", ".join(self.job_levels),
        ]
        return " ".join(p for p in parts if p)

    def to_recommendation(self) -> Recommendation:
        """Convert to API recommendation format."""
        return Recommendation(
            name=self.name,
            url=self.link,
            test_type=self.test_type_code,
        )

    def to_context_string(self) -> str:
        """Format for LLM context injection."""
        lang_str = ", ".join(self.languages[:3])
        if len(self.languages) > 3:
            lang_str += f" (+{len(self.languages) - 3} more)"
        levels_str = ", ".join(self.job_levels[:4])
        if len(self.job_levels) > 4:
            levels_str += f" (+{len(self.job_levels) - 4} more)"

        return (
            f"[{self.entity_id}] {self.name}\n"
            f"  Type: {self.test_type_code} ({', '.join(self.keys)})\n"
            f"  Description: {self.description[:200]}\n"
            f"  Duration: {self.duration or 'N/A'}\n"
            f"  Job Levels: {levels_str or 'N/A'}\n"
            f"  Languages: {lang_str or 'N/A'}\n"
            f"  Remote: {self.remote} | Adaptive: {self.adaptive}\n"
            f"  URL: {self.link}"
        )


class ConversationState(BaseModel):
    """
    Reconstructed state from stateless conversation history.
    Built fresh on every request.
    """
    # Accumulated requirements
    role_description: str = ""
    seniority_level: str = ""
    skills_needed: List[str] = Field(default_factory=list)
    industry: str = ""
    language_requirements: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)

    # Current shortlist (parsed from prior assistant messages)
    current_recommendations: List[str] = Field(
        default_factory=list,
        description="Names of currently recommended assessments"
    )

    # Intent of the latest message
    latest_intent: str = "unknown"  # clarify, recommend, refine, compare, off_topic, confirm

    # Full context for retrieval query
    retrieval_query: str = ""

    # Turn count
    turn_count: int = 0

    # Whether recommendations have been made previously
    has_prior_recommendations: bool = False

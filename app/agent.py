"""
Core agent logic for the SHL Conversational Assessment Recommender.
Handles conversation state reconstruction, intent classification,
retrieval orchestration, LLM interaction, and response post-processing.
"""

import json
import re
import os
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple

import google.generativeai as genai
from google.api_core import retry as api_retry
from dotenv import load_dotenv

from app.models import (
    Message, ChatRequest, ChatResponse, Recommendation,
    CatalogEntry, ConversationState,
)
from app.catalog import catalog
from app.retriever import retriever
from app.prompts import SYSTEM_PROMPT, RETRIEVAL_CONTEXT_TEMPLATE, REFINEMENT_CONTEXT_TEMPLATE

load_dotenv()

# Models to try in order (fallback chain)
MODEL_CHAIN = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


class Agent:
    """
    Stateless conversational agent.
    Reconstructs context from message history on every call.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        genai.configure(api_key=api_key)

        # Initialize models in fallback order
        self.models = []
        for model_name in MODEL_CHAIN:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config={
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_output_tokens": 4096,
                        "response_mime_type": "application/json",
                    },
                )
                self.models.append((model_name, model))
                print(f"[Agent] Initialized model: {model_name}")
            except Exception as e:
                print(f"[Agent] Could not initialize {model_name}: {e}")

        if not self.models:
            print("[Agent] WARNING: No LLM models available — will use deterministic fallback only")

        print(f"[Agent] Ready with {len(self.models)} model(s)")

    def process(self, request: ChatRequest) -> ChatResponse:
        """
        Main entry point. Process a chat request and return a response.
        Never raises — always returns a valid ChatResponse.
        """
        try:
            messages = request.messages

            # 1. Reconstruct conversation state
            state = self._reconstruct_state(messages)

            # 2. Build retrieval query from accumulated context
            retrieval_query = self._build_retrieval_query(messages, state)

            # 3. Retrieve relevant catalog entries
            retrieved = retriever.search_hybrid(retrieval_query, top_k=20)
            retrieved_entries = [entry for entry, _ in retrieved]

            # 4. If refinement, also include current shortlist entries
            if state.has_prior_recommendations:
                for name in state.current_recommendations:
                    entry = catalog.get_by_name(name)
                    if entry and entry not in retrieved_entries:
                        retrieved_entries.insert(0, entry)

            # 5. Call LLM with fallback
            response = self._call_llm_with_fallback(messages, retrieved_entries, state)

            # 6. Post-process: validate all recommendations against catalog
            response = self._postprocess(response)

            return response

        except Exception as e:
            print(f"[Agent] Error: {e}")
            traceback.print_exc()
            return self._fallback_response(str(e))

    def _reconstruct_state(self, messages: List[Message]) -> ConversationState:
        """
        Reconstruct conversation state from the full message history.
        This is called on every request since the API is stateless.
        """
        state = ConversationState()
        state.turn_count = sum(1 for m in messages if m.role == "user")

        # Parse prior assistant messages for existing recommendations
        for msg in messages:
            if msg.role == "assistant":
                try:
                    parsed = json.loads(msg.content)
                    recs = parsed.get("recommendations", [])
                    if recs:
                        state.has_prior_recommendations = True
                        state.current_recommendations = [r["name"] for r in recs]
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

        # Classify latest intent via heuristics
        if messages:
            last_msg = messages[-1].content.lower().strip()
            state.latest_intent = self._classify_intent_heuristic(last_msg, state)

        return state

    def _classify_intent_heuristic(self, msg: str, state: ConversationState) -> str:
        """Lightweight intent classification using keyword heuristics."""
        msg_lower = msg.lower()

        # Off-topic detection
        off_topic_signals = [
            "weather", "recipe", "joke", "sing", "poem", "story",
            "who are you", "what are you", "your name",
            "play a game", "tell me about yourself",
        ]
        if any(sig in msg_lower for sig in off_topic_signals) and \
           not any(w in msg_lower for w in ["assess", "test", "hire", "candidate", "shl"]):
            return "off_topic"

        # Confirmation signals
        confirm_signals = [
            "perfect", "that works", "confirmed", "locking it in",
            "locked in", "that's it", "that covers it", "good",
            "thanks", "thank you", "that's what we need",
            "keep the shortlist", "keep it", "final list",
        ]
        if any(sig in msg_lower for sig in confirm_signals) and state.has_prior_recommendations:
            return "confirm"

        # Comparison signals
        compare_signals = [
            "difference between", "what's the difference",
            "how does", "compare", "vs", "versus",
            "different from", "distinguish",
        ]
        if any(sig in msg_lower for sig in compare_signals):
            return "compare"

        # Refinement signals
        refine_signals = [
            "add ", "drop ", "remove ", "replace ", "swap ",
            "also include", "can you also", "instead of",
            "in that case", "actually", "change ",
        ]
        if any(sig in msg_lower for sig in refine_signals) and state.has_prior_recommendations:
            return "refine"

        # Vague queries needing clarification
        vague_signals = [
            "we need a solution", "what do you recommend",
            "help us", "help me", "what should we use",
        ]
        vague_count = sum(1 for sig in vague_signals if sig in msg_lower)
        word_count = len(msg.split())

        # Short vague messages need clarification
        if vague_count > 0 and word_count < 15 and not state.has_prior_recommendations:
            return "clarify"

        return "recommend"

    def _build_retrieval_query(self, messages: List[Message], state: ConversationState) -> str:
        """
        Build a comprehensive retrieval query from the full conversation.
        Concatenates all user messages to capture accumulated requirements.
        """
        user_messages = [m.content for m in messages if m.role == "user"]

        # Weight recent messages more by repeating
        if len(user_messages) >= 2:
            query = " ".join(user_messages[:-1]) + " " + user_messages[-1] + " " + user_messages[-1]
        else:
            query = " ".join(user_messages)

        return query

    def _call_llm_with_fallback(
        self,
        messages: List[Message],
        retrieved_entries: List[CatalogEntry],
        state: ConversationState,
    ) -> ChatResponse:
        """Try each model in the fallback chain."""
        last_error = None

        for model_name, model in self.models:
            try:
                return self._call_llm(model, model_name, messages, retrieved_entries, state)
            except Exception as e:
                last_error = e
                print(f"[Agent] Model {model_name} failed: {e}")
                continue

        # All models failed — use deterministic fallback
        print(f"[Agent] All LLM models failed, using deterministic fallback")
        return self._deterministic_fallback(messages, retrieved_entries, state)

    def _call_llm(
        self,
        model,
        model_name: str,
        messages: List[Message],
        retrieved_entries: List[CatalogEntry],
        state: ConversationState,
    ) -> ChatResponse:
        """Call a specific Gemini model."""
        # Format retrieved catalog entries for context
        catalog_context = "\n\n".join(
            entry.to_context_string() for entry in retrieved_entries[:20]
        )

        # Format conversation history
        conv_history = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in messages
        )

        # Choose template based on state
        if state.has_prior_recommendations and state.current_recommendations:
            current_shortlist = "\n".join(
                f"- {name}" for name in state.current_recommendations
            )
            prompt = REFINEMENT_CONTEXT_TEMPLATE.format(
                current_shortlist=current_shortlist,
                additional_entries=catalog_context,
                conversation_history=conv_history,
            )
        else:
            prompt = RETRIEVAL_CONTEXT_TEMPLATE.format(
                catalog_entries=catalog_context,
                conversation_history=conv_history,
            )

        # Call Gemini with hard timeout — bypass SDK retry completely
        import concurrent.futures

        def _do_call():
            return model.generate_content(
                [
                    {"role": "user", "parts": [SYSTEM_PROMPT]},
                    {"role": "model", "parts": ["Understood. I will act as SHL's AI Assessment Consultant, strictly following the catalog and output format rules."]},
                    {"role": "user", "parts": [prompt]},
                ],
                request_options={"timeout": 10, "retry": api_retry.Retry(deadline=0.1)},
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_call)
            try:
                response = future.result(timeout=12)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"{model_name} timed out after 12s")

        print(f"[Agent] {model_name} responded successfully")

        # Parse JSON response
        raw_text = response.text.strip()
        return self._parse_llm_response(raw_text)

    def _parse_llm_response(self, raw_text: str) -> ChatResponse:
        """Parse and validate the LLM's JSON response."""
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                brace_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if brace_match:
                    data = json.loads(brace_match.group(0))
                else:
                    raise ValueError(f"Could not parse LLM response as JSON: {raw_text[:200]}")

        reply = data.get("reply", "I can help you find the right SHL assessments.")
        recommendations = data.get("recommendations", [])
        end_of_conversation = data.get("end_of_conversation", False)

        rec_objects = []
        for rec in recommendations:
            if isinstance(rec, dict) and "name" in rec and "url" in rec:
                rec_objects.append(Recommendation(
                    name=rec["name"],
                    url=rec["url"],
                    test_type=rec.get("test_type", "K"),
                ))

        return ChatResponse(
            reply=str(reply),
            recommendations=rec_objects,
            end_of_conversation=bool(end_of_conversation),
        )

    def _postprocess(self, response: ChatResponse) -> ChatResponse:
        """
        Post-process response to ensure:
        1. All recommendations exist in the catalog
        2. Names and URLs are exact matches
        3. Test types are correct
        4. No duplicates
        """
        if not response.recommendations:
            return response

        validated_recs = []
        seen_names = set()

        for rec in response.recommendations:
            entry = catalog.validate_recommendation(rec.name, rec.url)

            if entry:
                canonical = entry.to_recommendation()
                if canonical.name.lower() not in seen_names:
                    validated_recs.append(canonical)
                    seen_names.add(canonical.name.lower())
            else:
                matches = catalog.search_by_name(rec.name)
                if matches:
                    best = matches[0]
                    canonical = best.to_recommendation()
                    if canonical.name.lower() not in seen_names:
                        validated_recs.append(canonical)
                        seen_names.add(canonical.name.lower())
                else:
                    print(f"[PostProcess] Dropping hallucinated recommendation: {rec.name}")

        validated_recs = validated_recs[:10]

        return ChatResponse(
            reply=response.reply,
            recommendations=validated_recs,
            end_of_conversation=response.end_of_conversation,
        )

    def _deterministic_fallback(
        self,
        messages: List[Message],
        retrieved_entries: List[CatalogEntry],
        state: ConversationState,
    ) -> ChatResponse:
        """
        Intelligent deterministic fallback when all LLM models fail.
        Uses intent classification and retrieval results to produce
        contextually appropriate responses.
        """
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_msg = msg.content
                break

        intent = state.latest_intent

        # --- OFF-TOPIC ---
        if intent == "off_topic":
            return ChatResponse(
                reply="I'm SHL's AI Assessment Consultant. I can help you find the right SHL assessment products for hiring and talent development. Could you describe the role you're hiring for?",
                recommendations=[],
                end_of_conversation=False,
            )

        # --- CONFIRM ---
        if intent == "confirm" and state.has_prior_recommendations:
            recs = []
            for name in state.current_recommendations:
                entry = catalog.get_by_name(name)
                if entry:
                    recs.append(entry.to_recommendation())
            return ChatResponse(
                reply="Confirmed. Here's your final shortlist.",
                recommendations=recs[:10],
                end_of_conversation=True,
            )

        # --- COMPARE ---
        if intent == "compare":
            return ChatResponse(
                reply="Those are distinct products designed for different purposes. Could you specify which aspects you'd like me to compare? I can detail the differences in scope, duration, and target audience.",
                recommendations=[],
                end_of_conversation=False,
            )

        # --- CLARIFY ---
        if intent == "clarify" or not retrieved_entries:
            return ChatResponse(
                reply="I'd like to recommend the best assessments for your needs. Could you tell me more about the role (job title, seniority level), the purpose (hiring, development, or restructuring), and any specific skills or constraints (language, time limits)?",
                recommendations=[],
                end_of_conversation=False,
            )

        # --- RECOMMEND / REFINE ---
        # Use top retrieved entries, with some intelligence about diversity
        recs = self._select_diverse_recommendations(retrieved_entries, state)

        reply = f"Based on your requirements, here are the SHL assessments I'd recommend for this role:"

        return ChatResponse(
            reply=reply,
            recommendations=recs,
            end_of_conversation=False,
        )

    def _select_diverse_recommendations(
        self,
        entries: List[CatalogEntry],
        state: ConversationState,
        max_items: int = 7,
    ) -> List[Recommendation]:
        """
        Select a diverse set of recommendations from retrieved entries.
        Ensures type diversity (K, P, A, S, etc.) and avoids duplicative reports.
        """
        selected = []
        seen_types = set()
        seen_names = set()

        # Priority pass: pick one of each type
        for entry in entries:
            type_code = entry.test_type_code
            primary_type = type_code.split(",")[0] if type_code else "K"

            if primary_type not in seen_types and entry.name.lower() not in seen_names:
                selected.append(entry.to_recommendation())
                seen_types.add(primary_type)
                seen_names.add(entry.name.lower())

            if len(selected) >= max_items:
                break

        # Fill remaining slots
        if len(selected) < max_items:
            for entry in entries:
                if entry.name.lower() not in seen_names:
                    selected.append(entry.to_recommendation())
                    seen_names.add(entry.name.lower())
                if len(selected) >= max_items:
                    break

        return selected

    def _fallback_response(self, error: str = "") -> ChatResponse:
        """Absolute last-resort fallback — never breaks schema."""
        return ChatResponse(
            reply="I can help you find the right SHL assessments. Could you describe the role you're hiring for, the seniority level, and any specific requirements?",
            recommendations=[],
            end_of_conversation=False,
        )


# Lazy-initialized global agent instance
_agent: Optional[Agent] = None

def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent

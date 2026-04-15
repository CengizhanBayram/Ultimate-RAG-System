import json
import logging
from typing import List

import google.generativeai as genai

from .models import QueryExpansion
from .config import Settings

logger = logging.getLogger(__name__)

EXPANSION_SYSTEM_PROMPT = """
You are a query expansion assistant for a Turkish customer support RAG system.
Given a user question, generate {n} alternative phrasings that:
1. Use different vocabulary (synonyms, related legal/business terms in Turkish)
2. Are more specific (e.g., mention article numbers if implied)
3. Cover different aspects of the question

Respond ONLY with a JSON array of strings. No explanation, no markdown, no code blocks.
Example input: "Pro paketi iptal edebilir miyim?"
Example output: ["Pro paket iptali nasıl yapılır", "Pro abonelik fesih koşulları",
                 "Pro paket sözleşme iptali Madde 4", "Pro ücret iadesi iptal durumunda"]
"""


class QueryExpander:
    """
    Uses Claude to generate semantically diverse query variants before retrieval.

    Architectural decision:
    Query expansion dramatically improves recall in RAG systems because users often
    phrase questions differently from how information is stored in source documents.
    For example, a user asking "param ne zaman gelir?" maps to the legal term
    "iade süresi" in the contract. Running 4 variants in parallel costs ~300ms
    but can double recall for edge-case queries.
    """

    def __init__(self, settings: Settings):
        genai.configure(api_key=settings.google_api_key)
        self._model_name = settings.gemini_model
        self.enabled = settings.query_expansion_enabled

    def expand(self, query: str, n: int = 4) -> QueryExpansion:
        """Generate n query variants using Gemini."""
        if not self.enabled:
            return QueryExpansion(
                original=query, variants=[], all_queries=[query]
            )

        try:
            model = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=400,
                    temperature=0.4,
                    response_mime_type="application/json",
                ),
            )
            prompt = EXPANSION_SYSTEM_PROMPT.format(n=n) + f"\n\nInput: {query}"
            response = model.generate_content(prompt)
            raw = response.text.strip()
            # Strip markdown code blocks if model wraps with them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            # Extract JSON array if embedded in extra text
            import re as _re
            arr_match = _re.search(r"\[.*?\]", raw, _re.DOTALL)
            if arr_match:
                raw = arr_match.group(0)
            variants: List[str] = json.loads(raw)
            if not isinstance(variants, list):
                raise ValueError("Expected JSON array")
            variants = [str(v) for v in variants[:n]]
            logger.info(f"Query expanded into {len(variants)} variants.")
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Using original query only.")
            variants = []

        return QueryExpansion(
            original=query,
            variants=variants,
            all_queries=[query] + variants,
        )

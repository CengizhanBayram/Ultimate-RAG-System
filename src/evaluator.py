import json
import logging
from typing import List

import google.generativeai as genai

from .models import ScoredDocument, EvaluationResult
from .config import Settings

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """
You are an evaluation assistant. Your job is to check if an AI answer is fully grounded
in the provided context chunks, with NO information from outside the context.

CONTEXT CHUNKS:
{context}

AI ANSWER TO EVALUATE:
{answer}

Task: For each factual claim in the answer, determine if it is directly supported by a context chunk.

Respond ONLY with JSON in this exact format (no markdown, no code blocks, just JSON):
{{
  "faithfulness_score": 0.0,
  "relevance_score": 0.0,
  "unfaithful_claims": [],
  "explanation": ""
}}

faithfulness_score = (supported claims) / (total claims), range 0.0 to 1.0
relevance_score = proportion of retrieved chunks that actually contributed to the answer, range 0.0 to 1.0
unfaithful_claims = list of specific claim strings that have NO support in the context
explanation = brief explanation in 1-2 sentences
"""


class RAGEvaluator:
    """
    Lightweight faithfulness and relevance scorer using Claude-as-judge.

    Architectural decision — why LLM-as-judge instead of RAGAS:
    RAGAS requires additional LLM calls and a separate pipeline. Using Claude
    directly as the judge is simpler, faster, and allows the same model that
    generated the answer to evaluate it — which works well because Claude can
    identify when a claim goes beyond the provided context.

    Failure mode: The evaluator fails open (passes) on API errors to avoid
    blocking user responses due to evaluation infrastructure failures.
    """

    def __init__(self, settings: Settings):
        genai.configure(api_key=settings.google_api_key)
        self._model_name = settings.gemini_model
        self.faithfulness_threshold = settings.faithfulness_threshold
        self.relevance_threshold = settings.relevance_threshold

    def evaluate(
        self,
        query: str,
        answer: str,
        chunks: List[ScoredDocument],
    ) -> EvaluationResult:
        """Evaluate faithfulness and relevance of the generated answer."""
        context_text = "\n---\n".join(
            [sd.document.content for sd in chunks]
        )

        try:
            model = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=600,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            prompt = (
                "Respond ONLY with valid JSON, no markdown, no code blocks.\n\n"
                + FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)
            )
            response = model.generate_content(prompt)
            raw = response.text.strip()
            # Strip any markdown wrapping
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            data = json.loads(raw)

            faithfulness = float(data.get("faithfulness_score", 0.0))
            relevance = float(data.get("relevance_score", 0.0))

            return EvaluationResult(
                faithfulness_score=faithfulness,
                relevance_score=relevance,
                passes_guardrail=(
                    faithfulness >= self.faithfulness_threshold
                    and relevance >= self.relevance_threshold
                ),
                unfaithful_claims=data.get("unfaithful_claims", []),
                explanation=data.get("explanation", ""),
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}. Failing open.")
            # Fail open: do not block response on evaluation error
            return EvaluationResult(
                faithfulness_score=1.0,
                relevance_score=1.0,
                passes_guardrail=True,
                unfaithful_claims=[],
                explanation=f"Evaluation skipped due to error: {e}",
            )

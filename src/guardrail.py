import logging
from typing import Tuple

from .models import EvaluationResult

logger = logging.getLogger(__name__)

REFUSAL_ANSWER = (
    "⚠️ Bu soruyu güvenilir şekilde yanıtlayamıyorum çünkü bilgi tabanımdaki verilerle "
    "tam uyumlu bir cevap üretemiyorum. Lütfen destek ekibimize başvurun veya "
    "soruyu farklı şekilde sormayı deneyin."
)


class HallucinationGuardrail:
    """
    Post-generation faithfulness gate.

    Architectural decision:
    The guardrail sits between generation and delivery. If the faithfulness score
    (fraction of claims grounded in retrieved context) falls below 0.70, the
    system refuses to surface the answer. This prevents confident-sounding
    hallucinations from reaching the user.

    Two tiers:
    1. Hard block (faithfulness < threshold OR relevance < threshold):
       Replace answer with refusal message entirely.
    2. Soft warning (passes guardrail but has some unfaithful claims):
       Append a warning note but allow the answer through, letting the user
       know which specific claims could not be verified.

    Blocking is never cached — blocked responses are not stored in QueryCache.
    """

    def check(
        self, answer: str, evaluation: EvaluationResult
    ) -> Tuple[str, bool]:
        """
        Returns (final_answer, was_blocked).
        """
        if not evaluation.passes_guardrail:
            logger.warning(
                f"Response BLOCKED by guardrail. "
                f"Faithfulness={evaluation.faithfulness_score:.2f}, "
                f"Relevance={evaluation.relevance_score:.2f}"
            )
            return REFUSAL_ANSWER, True

        if evaluation.unfaithful_claims:
            claim_list = ", ".join(evaluation.unfaithful_claims[:2])
            warning = (
                f"\n\n⚠️ **Not:** Bu yanıtta bağlamda doğrulanamayan bazı ifadeler "
                f"tespit edildi: {claim_list}"
            )
            logger.info(
                f"Response passed with {len(evaluation.unfaithful_claims)} soft warnings."
            )
            return answer + warning, False

        return answer, False

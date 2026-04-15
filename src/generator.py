import logging
from typing import List

import google.generativeai as genai

from .config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are the official customer support assistant for this company.
Answer questions STRICTLY using the CONTEXT provided. Never use outside knowledge.

════════════════════════════════════════════
SOURCE PRIORITY — ABSOLUTE RULE, NO EXCEPTIONS
════════════════════════════════════════════
Priority 1 (HIGHEST) → [UPDATE_LOG] tagged sections
  • These are the most current records. They always override everything else.
  • If multiple UPDATE_LOG entries exist for the same topic, use ONLY the latest date.

Priority 2 → [PRICE_TABLE] tagged sections
  • Current structured pricing. Valid unless an UPDATE_LOG explicitly overrides it.

Priority 3 (BASELINE) → [CONTRACT] tagged sections
  • The foundational agreement. Valid only when NOT overridden by UPDATE_LOG or PRICE_TABLE.

Sections marked ⚠️ [SUPERSEDED] are included for transparency only.
NEVER use superseded information as the primary answer.

════════════════════════════════════════════
PACKAGE-SCOPING RULE
════════════════════════════════════════════
If an update targets a specific package (e.g., "Pro"), apply it ONLY to Pro.
For other packages, fall back to the most recent applicable update or base contract.
Never mix Pro data into Basic answers or vice versa.

════════════════════════════════════════════
INLINE CITATION FORMAT
════════════════════════════════════════════
Every factual claim MUST be followed immediately by a citation tag.
Format: `[Kaynak: {filename} | {ID or Article} | {date}]`
Example:
  "İade süreniz 30 gündür `[Kaynak: guncellemeler.json | UPD-002 | 2024-06-01]`"

════════════════════════════════════════════
MANDATORY RESPONSE STRUCTURE
════════════════════════════════════════════
1. Answer the question directly with inline citations throughout.
2. Append this block at the end — ALWAYS:

---
📎 KULLANILAN KAYNAKLAR:
| Kaynak Dosya | Kullanılan Bilgi | Referans | Tarih |
|---|---|---|---|
| ... | ... | ... | ... |

⚠️ GEÇERSİZ KILINEN BİLGİ (varsa):
• "[eski_kaynak]" içindeki "[eski_değer]", [tarih] tarihli [güncelleme_id] ile geçersiz kılınmıştır.
---

════════════════════════════════════════════
UNKNOWN INFORMATION
════════════════════════════════════════════
If the context does not contain the answer, respond EXACTLY:
"Bu konuda bilgi tabanımda güncel veri bulunmuyor. Lütfen destek ekibimize başvurun."
Never guess. Never use training data to fill gaps.
"""


class RAGGenerator:
    """Generates grounded responses using Gemini with structured context."""

    def __init__(self, settings: Settings):
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                max_output_tokens=settings.max_tokens,
                temperature=0.1,
            ),
        )

    def generate(
        self, query: str, context: str, queries_used: List[str]
    ) -> str:
        expansions = queries_used[1:] if len(queries_used) > 1 else []
        expansion_note = (
            f"(Bu soru şu şekillerde de arandı: {', '.join(expansions)})"
            if expansions
            else "(Sorgu genişletme uygulanmadı)"
        )

        user_message = f"""CONTEXT:
{context}

═══════════════════════════
USER QUESTION: {query}

{expansion_note}
"""
        response = self.model.generate_content(user_message)
        return response.text

"""Final-text verification against the checked plan.

The strict renderer is the current trust boundary. Arbitrary LLM text is rejected unless every
sentence exactly corresponds to a supported or explicit not-established sentence. A future
independently evaluated proposition extractor may safely broaden this acceptance set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import AnswerPlan, SupportStatus


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    unsupported_sentences: tuple[str, ...]


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(s.strip() if s.strip().endswith(".") else s.strip() + "."
                 for s in re.split(r"(?<=\.)\s+", text.strip()) if s.strip())


def verify_final_text(plan: AnswerPlan, text: str) -> VerificationResult:
    allowed = {claim.proposition.strip() for claim in plan.claims
               if claim.support == SupportStatus.SUPPORTED}
    allowed.update(item.strip() for item in plan.not_established)
    unsupported = tuple(sentence for sentence in _sentences(text) if sentence not in allowed)
    return VerificationResult(not unsupported, unsupported)

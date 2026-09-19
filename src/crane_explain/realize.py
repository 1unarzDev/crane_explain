"""Deterministic realization and bounded fallback."""

from .models import AnswerPlan, SupportStatus


def render_template(plan: AnswerPlan) -> str:
    parts = [claim.proposition for claim in plan.claims if claim.support == SupportStatus.SUPPORTED]
    parts.extend(plan.not_established)
    return " ".join(parts) if parts else "The available evidence is insufficient to answer."


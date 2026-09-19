"""Information-parity-aware A/B/C/D/E benchmark orchestration.

Model and extractor implementations are injected so the harness remains offline and calls can be
cached by a provider-specific adapter. This module never retries a model output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .models import AnswerPlan, EpisodeRecord
from .realize import render_template
from .reasoning import plan_contrast, plan_recovery_count
from .verification import verify_final_text


class Condition(str, Enum):
    A_PROSE_DIRECT = "A"
    B_STRUCTURED_DIRECT = "B"
    C_PROSE_EXTRACT_CHECKED = "C"
    D_NATIVE_CHECKED = "D"
    E_TEMPLATE = "E"


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    episode: EpisodeRecord
    prose: str
    question: str
    question_kind: str
    alternative_id: str | None
    structured_fact_ids: frozenset[str]
    prose_fact_ids: frozenset[str]


@dataclass(frozen=True)
class BenchmarkOutput:
    condition: Condition
    case_id: str
    text: str
    disposition: str
    verification_accepted: bool | None
    used_template_fallback: bool


def audit_information_parity(case: BenchmarkCase) -> None:
    if case.structured_fact_ids != case.prose_fact_ids:
        privileged = sorted(case.structured_fact_ids - case.prose_fact_ids)
        prose_only = sorted(case.prose_fact_ids - case.structured_fact_ids)
        raise ValueError(f"information parity failed: structured_only={privileged}; prose_only={prose_only}")


def _plan(case: BenchmarkCase, episode: EpisodeRecord | None = None) -> AnswerPlan:
    record = episode or case.episode
    if case.question_kind == "contrast":
        if not case.alternative_id:
            raise ValueError("contrast question requires alternative_id")
        return plan_contrast(record, case.alternative_id)
    if case.question_kind == "recovery_count":
        return plan_recovery_count(record)
    raise ValueError(f"unsupported question kind: {case.question_kind}")


def run_condition(
    case: BenchmarkCase,
    condition: Condition,
    *,
    direct_generator: Callable[[str, str], str] | None = None,
    extractor: Callable[[str], EpisodeRecord] | None = None,
    plan_realizer: Callable[[AnswerPlan], str] | None = None,
) -> BenchmarkOutput:
    audit_information_parity(case)
    if condition in {Condition.A_PROSE_DIRECT, Condition.B_STRUCTURED_DIRECT}:
        if direct_generator is None:
            raise ValueError("direct_generator is required for A/B")
        evidence = case.prose if condition == Condition.A_PROSE_DIRECT else json.dumps(
            case.episode.to_dict(), sort_keys=True, separators=(",", ":"))
        return BenchmarkOutput(condition, case.case_id, direct_generator(evidence, case.question),
                               "uncontrolled", None, False)
    if condition == Condition.C_PROSE_EXTRACT_CHECKED:
        if extractor is None:
            raise ValueError("extractor is required for C")
        plan = _plan(case, extractor(case.prose))
    else:
        plan = _plan(case)
    if condition == Condition.E_TEMPLATE:
        text = render_template(plan)
        return BenchmarkOutput(condition, case.case_id, text, plan.disposition, True, False)
    candidate = (plan_realizer or render_template)(plan)
    verification = verify_final_text(plan, candidate)
    if verification.accepted:
        return BenchmarkOutput(condition, case.case_id, candidate, plan.disposition, True, False)
    # No resampling. A provider adapter may implement one predeclared repair before returning its
    # candidate; this common harness performs the mandatory checked fallback.
    return BenchmarkOutput(condition, case.case_id, render_template(plan), plan.disposition,
                           False, True)


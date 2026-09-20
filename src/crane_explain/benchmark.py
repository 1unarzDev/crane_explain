"""Information-parity-aware A--H benchmark orchestration.

Model and extractor implementations are injected so the harness remains offline and calls can be
cached by a provider-specific adapter. This module never retries a model output.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .models import AnswerPlan, EpisodeRecord
from .provenance import ResolvedProvenance
from .realize import render_template
from .reasoning import (
    plan_contrast,
    plan_failure_cause,
    plan_planning_failure,
    plan_recovery_count,
    plan_recovery_mechanism,
    plan_terminal_status,
    plan_unsupported_counterfactual,
)
from .verification import verify_final_text


class Condition(str, Enum):
    A_PROSE_DIRECT = "A"
    B_STRUCTURED_DIRECT = "B"
    C_PROSE_EXTRACT_CHECKED = "C"
    D_NATIVE_CHECKED = "D"
    E_TEMPLATE = "E"
    F_REPOSITORY_AGENT = "F"
    G_PROVENANCE_CHECKED = "G"
    H_UNRESTRICTED_REPOSITORY_AGENT = "H"


class RepositoryAccess(str, Enum):
    READ_ONLY_UNRESTRICTED = "read_only_unrestricted"


@dataclass(frozen=True)
class RepositoryAgentRequest:
    evidence: str
    question: str
    repository: str
    repository_commit: str
    access: RepositoryAccess


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
    premise_count: int | None = None
    structured_presentation: dict | None = None
    raw_robot_visible: str | None = None
    repository: str | None = None
    repository_commit: str | None = None
    provenance: ResolvedProvenance | None = None


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
        raise ValueError(
            f"information parity failed: structured_only={privileged}; prose_only={prose_only}"
        )


def _plan(
    case: BenchmarkCase,
    episode: EpisodeRecord | None = None,
    provenance: ResolvedProvenance | None = None,
) -> AnswerPlan:
    record = episode or case.episode
    if case.question_kind == "contrast":
        if not case.alternative_id:
            raise ValueError("contrast question requires alternative_id")
        return plan_contrast(record, case.alternative_id)
    if case.question_kind == "recovery_count":
        return plan_recovery_count(record, case.premise_count)
    if case.question_kind == "recovery_mechanism":
        return plan_recovery_mechanism(record, provenance)
    if case.question_kind == "failure_cause":
        return plan_failure_cause(record)
    if case.question_kind == "planning_failure":
        return plan_planning_failure(record)
    if case.question_kind == "unsupported_counterfactual":
        return plan_unsupported_counterfactual(record)
    if case.question_kind == "terminal_status":
        return plan_terminal_status(record)
    raise ValueError(f"unsupported question kind: {case.question_kind}")


def run_condition(
    case: BenchmarkCase,
    condition: Condition,
    *,
    direct_generator: Callable[[str, str], str] | None = None,
    extractor: Callable[[str], EpisodeRecord] | None = None,
    plan_realizer: Callable[[AnswerPlan], str] | None = None,
    repository_agent: Callable[[RepositoryAgentRequest], str] | None = None,
) -> BenchmarkOutput:
    audit_information_parity(case)
    if condition in {Condition.A_PROSE_DIRECT, Condition.B_STRUCTURED_DIRECT}:
        if direct_generator is None:
            raise ValueError("direct_generator is required for A/B")
        structured = case.structured_presentation or case.episode.to_dict()
        evidence = (
            case.prose
            if condition == Condition.A_PROSE_DIRECT
            else json.dumps(structured, sort_keys=True, separators=(",", ":"))
        )
        return BenchmarkOutput(
            condition,
            case.case_id,
            direct_generator(evidence, case.question),
            "uncontrolled",
            None,
            False,
        )
    if condition in {
        Condition.F_REPOSITORY_AGENT,
        Condition.H_UNRESTRICTED_REPOSITORY_AGENT,
    }:
        if repository_agent is None:
            raise ValueError("repository_agent is required for F/H")
        if not case.repository or not case.repository_commit:
            raise ValueError("repository and exact repository_commit are required for F/H")
        if condition == Condition.F_REPOSITORY_AGENT:
            if case.raw_robot_visible is None:
                raise ValueError("raw_robot_visible is required for F")
            evidence = case.raw_robot_visible
        else:
            structured = case.structured_presentation or case.episode.to_dict()
            evidence = json.dumps(structured, sort_keys=True, separators=(",", ":"))
        request = RepositoryAgentRequest(
            evidence=evidence,
            question=case.question,
            repository=case.repository,
            repository_commit=case.repository_commit,
            access=RepositoryAccess.READ_ONLY_UNRESTRICTED,
        )
        return BenchmarkOutput(
            condition, case.case_id, repository_agent(request), "uncontrolled", None, False
        )
    if condition == Condition.C_PROSE_EXTRACT_CHECKED:
        if extractor is None:
            raise ValueError("extractor is required for C")
        plan = _plan(case, extractor(case.prose))
    elif condition == Condition.G_PROVENANCE_CHECKED:
        if case.provenance is None:
            raise ValueError("resolved provenance is required for G")
        plan = _plan(case, provenance=case.provenance)
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
    return BenchmarkOutput(
        condition, case.case_id, render_template(plan), plan.disposition, False, True
    )

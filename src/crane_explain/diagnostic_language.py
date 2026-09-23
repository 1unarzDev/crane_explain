"""Bounded verification for natural-language realizations of checked diagnostics.

This module is separate from diagnostic computation and from the frozen legacy verifier.  Its
single interface performs one deterministic evidence-citation repair, checks the four-section
response contract, licenses quantitative claims against the checked result, and applies narrow
mechanism-specific proposition gates.  Anything outside those supported families fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from .diagnostics import DiagnosticDisposition, DiagnosticResult


_SECTION_NAMES = (
    "Diagnosis",
    "Decisive evidence",
    "Failure chain",
    "Limits and next check",
)
_SECTION_LINE = re.compile(
    r"^(?:#{1,6}\s*)?(Diagnosis|Decisive evidence|Failure chain|Limits and next check)"
    r"\s*:?[ \t]*(.*)$",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9])")
_HASH = re.compile(r"\b[a-fA-F0-9]{32,}\b")


@dataclass(frozen=True)
class DiagnosticLanguageVerification:
    accepted: bool
    checked_text: str
    repair_applied: bool
    reasons: tuple[str, ...]
    policy: str = "bounded-diagnostic-language-v1"


def _parse_sections(text: str) -> tuple[dict[str, str], list[str]]:
    sections: dict[str, list[str]] = {}
    order: list[str] = []
    current: str | None = None
    errors: list[str] = []
    canonical = {name.lower(): name for name in _SECTION_NAMES}
    for raw_line in text.strip().splitlines():
        line = raw_line.rstrip()
        match = _SECTION_LINE.match(line.strip())
        if match:
            name = canonical[match.group(1).lower()]
            if name in sections:
                errors.append(f"duplicate section: {name}")
                current = None
                continue
            current = name
            order.append(name)
            sections[name] = []
            if match.group(2).strip():
                sections[name].append(match.group(2).strip())
            continue
        if line.lstrip().startswith("#"):
            errors.append(f"unrecognized heading: {line.strip()}")
            continue
        if current is None:
            if line.strip():
                errors.append("text appears outside the four declared sections")
            continue
        if line.strip():
            sections[current].append(line.strip())
    if order != list(_SECTION_NAMES):
        errors.append("sections must appear exactly once in the required order")
    rendered = {name: " ".join(sections.get(name, ())).strip() for name in _SECTION_NAMES}
    for name, body in rendered.items():
        if not body:
            errors.append(f"empty section: {name}")
    return rendered, errors


def _render_sections(sections: dict[str, str]) -> str:
    return "\n\n".join(f"## {name}\n\n{sections[name]}" for name in _SECTION_NAMES)


def _all_numbers(value: Any) -> list[float]:
    numbers: list[float] = []
    if isinstance(value, bool) or value is None:
        return numbers
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            numbers.append(float(value))
        return numbers
    if isinstance(value, str):
        without_hashes = _HASH.sub("", value)
        numbers.extend(float(match.group()) for match in _NUMBER.finditer(without_hashes))
        return numbers
    if isinstance(value, dict):
        for item in value.values():
            numbers.extend(_all_numbers(item))
        return numbers
    if isinstance(value, (tuple, list)):
        for item in value:
            numbers.extend(_all_numbers(item))
        return numbers
    return numbers


def _allowed_numbers(result: DiagnosticResult) -> list[float]:
    values = _all_numbers(result.to_dict())
    for measurement in result.measurements:
        if measurement.interval_s is not None:
            values.append(measurement.interval_s[1] - measurement.interval_s[0])
    return values


def _numeric_claims(text: str) -> list[tuple[str, float, int]]:
    without_ids = re.sub(r"Evidence IDs:.*?(?:\.|$)", "", text, flags=re.IGNORECASE)
    without_hashes = _HASH.sub("", without_ids)
    claims = []
    for match in _NUMBER.finditer(without_hashes):
        token = match.group()
        decimals = len(token.split(".", 1)[1]) if "." in token else 0
        claims.append((token, float(token), decimals))
    return claims


def _number_is_licensed(value: float, decimals: int, allowed: list[float]) -> bool:
    rounding = 0.5 * (10.0 ** -decimals) if decimals else 0.5
    return any(abs(value - candidate) <= max(rounding + 1e-9, abs(candidate) * 1e-6)
               for candidate in allowed)


def _normalized(text: str) -> str:
    return (
        text.lower()
        .replace("→", "->")
        .replace("–", "-")
        .replace("—", "-")
        .replace("`", "")
        .replace("**", "")
    )


def _contains_any(text: str, alternatives: tuple[str, ...]) -> bool:
    return any(item in text for item in alternatives)


def _mechanism_checks(result: DiagnosticResult, sections: dict[str, str]) -> list[str]:
    diagnosis = _normalized(sections["Diagnosis"])
    evidence = _normalized(sections["Decisive evidence"])
    failure = _normalized(sections["Failure chain"])
    limits = _normalized(sections["Limits and next check"])
    all_text = " ".join((diagnosis, evidence, failure, limits))
    errors: list[str] = []

    def require(where: str, alternatives: tuple[str, ...], description: str) -> None:
        if not _contains_any(where, alternatives):
            errors.append(f"missing required proposition: {description}")

    mechanism = result.mechanism
    disposition = result.disposition
    if mechanism in {"terminal_stopping_margin", "post_return_motion_exceeded_position_margin"}:
        if disposition == DiagnosticDisposition.SUPPORTED:
            require(diagnosis, ("margin",), "terminal margin")
            require(all_text, ("post-result", "after return", "after the action returned"),
                    "post-return motion")
            require(all_text, ("settled", "outside"), "settled task outcome")
            require(limits, ("not uniquely", "does not uniquely", "unresolved", "not a unique"),
                    "unresolved physical source")
        elif disposition == DiagnosticDisposition.INSUFFICIENT:
            require(diagnosis, ("cannot", "insufficient", "missing"),
                    "insufficient diagnostic status")
            require(limits, ("missing", "not establish", "cannot establish"),
                    "missing decisive measurement")
        else:
            require(diagnosis, ("not establish", "did not", "not triggered"),
                    "non-triggered terminal mechanism")
    elif mechanism in {"geometric_route_restriction", "deadline_aligned_abort_with_unresolved_geometry"}:
        if disposition == DiagnosticDisposition.SUPPORTED:
            require(diagnosis, ("direct route",), "requested direct route")
            require(diagnosis, ("non-traversable", "restricted", "restriction"),
                    "route restriction")
            if result.contradictory_evidence:
                require(all_text, ("traversable connection", "connected"),
                        "retained-grid connection conflict")
            require(all_text, ("deadline",), "deadline relationship")
            require(failure, ("abort", "failed"), "terminal action outcome")
            require(limits, ("physical infeasibility", "physical impossibility"),
                    "physical-feasibility limit")
        else:
            require(diagnosis, ("cannot", "unresolved", "not assess"),
                    "unresolved geometry")
            require(all_text, ("deadline",), "retained deadline evidence")
    elif mechanism == "no_failure_observed":
        require(diagnosis, ("succeeded", "success"), "recorded successful action")
        require(failure, ("no terminal failure", "no failure chain"), "false failure premise")
        require(limits, ("does not prove", "not prove", "does not establish"),
                "bounded success interpretation")
    elif mechanism == "planner_failure_eligible_recovery_sequence":
        require(diagnosis, ("at least", "lower bound"), "lower-bound recovery count")
        require(all_text, ("source-qualified",), "source-qualified invocations")
        require(failure, ("computepathtopose",), "planner branch failure")
        require(failure, ("wouldaplannerrecoveryhelp",), "recovery eligibility")
        require(all_text, ("eventually succeeded", "action succeeded", "task success"),
                "eventual task outcome")
        require(limits, ("physical cause",), "unresolved physical cause")
    elif mechanism == "retained_navigation_model_disconnection":
        require(diagnosis, ("no 8-connected route", "no connected route", "no connection"),
                "retained-grid disconnection")
        require(all_text, ("goal cell",), "goal-cell classification")
        require(all_text, ("planner-failure", "planner failure", "failed to create plan"),
                "planner failure correspondence")
        require(failure, ("abort",), "terminal action outcome")
        require(limits, ("physical",), "physical-cause limitation")
    elif mechanism == "command_to_motion_discrepancy":
        if disposition == DiagnosticDisposition.SUPPORTED:
            action_status = next(
                (
                    str(item.value).lower()
                    for item in result.measurements
                    if item.id == "action_status"
                ),
                "",
            )
            response_recovery_recorded = any(
                item.id == "recovered_measured_planar_speed" for item in result.measurements
            )
            require(diagnosis, ("command-to-motion", "command and odometry"),
                    "command-to-motion discrepancy")
            require(all_text, ("command",), "delivered command evidence")
            require(all_text, ("odometry", "measured motion", "motion response"),
                    "independently measured motion evidence")
            require(failure, ("followpath", "follow path"), "controller failure sequence")
            require(failure, ("wait",), "source-qualified recovery sequence")
            if action_status == "succeeded":
                require(failure, ("succeeded", "success"), "successful terminal action outcome")
                if response_recovery_recorded:
                    require(failure, ("recover", "resum", "restor", "later measured response"),
                            "measured-response recovery")
            else:
                require(failure, ("abort", "failed"), "terminal action outcome")
            require(limits, ("actuator acceptance", "accepted by the actuator"),
                    "actuator-acceptance limit")
            require(limits, ("unique physical cause", "does not distinguish", "unresolved"),
                    "unresolved unique cause")
        elif disposition == DiagnosticDisposition.NOT_TRIGGERED:
            require(diagnosis, ("succeeded", "success", "not triggered", "did not contain"),
                    "non-triggered or successful outcome")
            require(failure, ("no supported", "no observed", "no failure"),
                    "absent command-motion failure chain")
            require(limits, ("only", "does not prove", "not prove"),
                    "bounded negative interpretation")
        else:
            require(diagnosis, ("cannot", "insufficient", "missing"),
                    "insufficient diagnostic status")
            require(all_text, ("missing", "prevents", "cannot", "without time-aligned"),
                    "missing command-motion evidence")
            require(limits, ("does not establish", "cannot establish", "unresolved", "insufficient"),
                    "bounded insufficient-evidence interpretation")
    else:
        errors.append(f"unsupported diagnostic mechanism: {mechanism}")
    return errors


def _unsupported_cause_checks(text: str) -> list[str]:
    errors = []
    risky = (
        "wave", "current", "wind", "motor", "collision", "hydrodynamic", "inertia",
        "disturbance", "actuation", "physical obstacle", "berth",
    )
    safe = (
        "not ", "does not", "did not", "cannot", "unresolved", "remain unresolved",
        "no evidence", "not established", "not uniquely", "does not uniquely",
    )
    for sentence in re.split(r"(?<=[.!?;])\s+", _normalized(text)):
        # A bounded next-check request may name a signal to record without asserting that signal's
        # mechanism occurred.  Keep this narrow: the sentence must explicitly identify itself as
        # the next check and ask to measure/record/inspect/test evidence.
        prospective_check = (
            (
                "next check:" in sentence
                or sentence.lstrip().startswith("next,")
                or sentence.lstrip().startswith("next ")
            )
            and _contains_any(sentence, ("record", "measure", "inspect", "test"))
        )
        for term in risky:
            term_present = re.search(r"\b" + re.escape(term) + r"\b", sentence) is not None
            if term_present and not prospective_check and not _contains_any(sentence, safe):
                errors.append(f"unsupported physical-cause wording: {term}")
    return errors


def verify_bounded_diagnostic_text(
    result: DiagnosticResult,
    candidate: str,
) -> DiagnosticLanguageVerification:
    """Verify one candidate and apply at most one evidence-ID-only repair.

    The interface never calls a model and never changes substantive wording.  The sole repair
    appends the complete checked evidence-ID list when the candidate omitted citations entirely.
    Partial or altered citation sets fail closed.
    """

    sections, errors = _parse_sections(candidate)
    repair_applied = False
    evidence_ids = tuple(result.supporting_evidence)
    present_ids = {item for item in evidence_ids if item in candidate}
    if evidence_ids and not present_ids and "evidence ids" not in candidate.lower():
        sections["Decisive evidence"] = (
            sections["Decisive evidence"].rstrip(". ")
            + f". Evidence IDs: {', '.join(evidence_ids)}."
        )
        repair_applied = True
    elif len(present_ids) != len(evidence_ids):
        errors.append("partial or altered evidence-ID set")

    checked_text = _render_sections(sections)
    allowed = _allowed_numbers(result)
    for token, value, decimals in _numeric_claims(checked_text):
        if not _number_is_licensed(value, decimals, allowed):
            errors.append(f"unlicensed numeric claim: {token}")

    errors.extend(_mechanism_checks(result, sections))
    errors.extend(_unsupported_cause_checks(checked_text))
    if evidence_ids and not all(item in checked_text for item in evidence_ids):
        errors.append("complete evidence-ID set is absent after bounded repair")
    return DiagnosticLanguageVerification(
        accepted=not errors,
        checked_text=checked_text,
        repair_applied=repair_applied,
        reasons=tuple(dict.fromkeys(errors)),
    )

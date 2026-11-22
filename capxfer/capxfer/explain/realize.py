"""Deterministic realization of a claim list into text, and verification that a text realizes exactly that list.

Verification reuses crane_explain (imported read-only via sys.path):
  * `models.AnswerPlan` / `Claim` / `SupportStatus` carry the realized plan;
  * `verification.verify_final_text` rejects any sentence not licensed by the plan;
  * `diagnostic_language._numeric_claims` / `_number_is_licensed` reject any unlicensed number.
On top of that we check completeness (every planned sentence appears exactly once), claim vocabulary, decision
consistency, and that `withheld` equals the realized withheld sentences.
"""
from __future__ import annotations

from collections import Counter
import sys

_CRANE_SRC = "../packages/astro_dock/src/crane_explain/src"
if _CRANE_SRC not in sys.path:
    sys.path.insert(0, _CRANE_SRC)

from crane_explain.models import AnswerPlan, Claim as CraneClaim, ClaimClass, EvidenceLevel, SupportStatus  # noqa: E402
from crane_explain.verification import _sentences, verify_final_text  # noqa: E402
from crane_explain.diagnostic_language import _number_is_licensed, _numeric_claims  # noqa: E402

CLAIM_TYPES = ("capability", "wheel_mode", "cause", "ood")
CAP_VALUES = ("reliable", "unreliable")
_ORDER = {"capability": 0, "wheel_mode": 1, "cause": 2, "ood": 3}


def join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _cap_template(C, claim):
    basis = claim.get("basis", "calibrated")
    if basis == "point":
        return C.baseline_t[f"point_cap_{claim['value']}"]
    if basis == "uncalibrated":
        return C.baseline_t[f"uncalibrated_cap_{claim['value']}"]
    return C.claims[claim["contract_id"]]["proposition"]


def _cap_fragment(C, claim):
    t = _cap_template(C, claim)
    prefix = t.split("{direction}")[0]
    frag = "{direction}" + t.split("{direction}", 1)[1].rstrip(".")
    nums = {k: C.fmt(v) for k, v in claim.get("numbers", {}).items()}
    return prefix, frag.format(direction=C.direction(claim["subject"]), **nums)


def _claim_sentence(C, claim, n_wheels):
    t, ct = claim["type"], C.claims.get(claim["contract_id"], {})
    if t == "wheel_mode":
        if claim["value"] == "ok":
            ws = list(claim["subject"])
            if len(ws) == n_wheels and n_wheels > 1:
                return ct["proposition_all"].format(n=n_wheels)
            if len(ws) == 1:
                return ct["proposition_one"].format(wheels=ws[0])
            return ct["proposition"].format(wheels=join(ws))
        if claim["value"] == "not_ok":
            return ct["proposition"].format(wheel=claim["subject"],
                                            modes=join(m.replace("_", "-") for m in claim["extra"]["modes"]))
        return ct["proposition"].format(wheel=claim["subject"], mode=C.mode_phrase(claim["value"]))
    if t == "cause":
        dirs = join(C.direction(k) for k in claim["extra"]["deficit"])
        if claim["contract_id"] == "claim-cause-wheel":
            return ct["proposition"].format(directions=dirs, wheel=claim["subject"], mode=C.mode_phrase(claim["value"]))
        return ct["proposition"].format(directions=dirs)
    if t == "ood":
        return ct["proposition"]
    raise ValueError(f"no sentence template for claim type {t}")


def _withheld_sentence(C, claim, tau):
    key = claim["contract_id"]
    if key == "capability_undetermined" and claim.get("basis") == "uncalibrated":
        tmpl = C.baseline_t["uncalibrated_capability_undetermined"]
    else:
        tmpl = C.withheld_t[key]
    ex = claim.get("extra", {})
    nums = {k: C.fmt(v) for k, v in claim.get("numbers", {}).items()}
    return tmpl.format(direction=C.direction(claim["subject"]) if key == "capability_undetermined" else "",
                       tau=C.fmt(tau), wheels=join(ex.get("wheels", [])), wheel=ex.get("wheel", ""),
                       directions=join(C.direction(k) for k in ex.get("deficit", [])), **nums)


def realize_sentences(C, expl):
    """Return (supported [(sentence, claim_ids)], withheld [sentence], actions [sentence]) for an explanation core."""
    tau = expl.get("tau", C.tau)
    n_w = len(expl.get("wheels", ()))
    sup = [c for c in expl["claims"] if c["support"] == "supported"]
    wh = [c for c in expl["claims"] if c["support"] == "withheld"]
    out = []
    # capability: group per (template prefix) to keep the text short; order fixed by target key
    caps = sorted([c for c in sup if c["type"] == "capability"], key=lambda c: (c["value"] != "reliable",
                                                                                 C.keys.index(c["subject"])))
    groups: dict = {}
    for c in caps:
        prefix, frag = _cap_fragment(C, c)
        groups.setdefault(prefix, []).append((frag, c["id"]))
    for prefix, items in groups.items():
        out.append((prefix + join(f for f, _ in items) + ".", tuple(i for _, i in items)))
    for c in sorted([c for c in sup if c["type"] != "capability"], key=lambda c: _ORDER[c["type"]]):
        out.append((_claim_sentence(C, c, n_w), (c["id"],)))
    withheld = [_withheld_sentence(C, c, tau) for c in wh]
    actions = []
    d = expl.get("decision")
    if d:
        actions.append(C.action_t[f"decision_{d['outcome']}"].format(task=d["task"]))
    p = expl.get("next_probe")
    if p:
        actions.append(C.action_t["next_probe"].format(probe=C.params["probe_phrases"][p["pid"]],
                                                       direction=C.direction(p["key"])))
    return out, withheld, actions


def to_answer_plan(C, expl, question="What can the robot still do?"):
    sup, withheld, actions = realize_sentences(C, expl)
    claims = tuple(
        CraneClaim(id="+".join(ids), proposition=s, support=SupportStatus.SUPPORTED, evidence_ids=ids,
                   derivation="wheel-claim-contract-v1", temporal_scope="trial", evidence_level=EvidenceLevel.MODEL_CAUSAL
                   if any(i.startswith("cause") for i in ids) else EvidenceLevel.RECORDED_SEQUENCE,
                   claim_class=ClaimClass.DERIVED)
        for s, ids in sup)
    disposition = expl.get("disposition", "partial")
    return AnswerPlan(question=question, disposition=disposition, claims=claims,
                      not_established=tuple(withheld) + tuple(actions))


def render(C, expl):
    sup, withheld, actions = realize_sentences(C, expl)
    text = " ".join([s for s, _ in sup] + withheld + actions)
    return text if text else "The available evidence is insufficient to answer."


def allowed_numbers(C, expl):
    vals = [expl.get("tau", C.tau), float(len(expl.get("wheels", ())))]
    for c in expl["claims"]:
        vals.extend(float(v) for v in c.get("numbers", {}).values())
    d = expl.get("decision")
    if d:
        vals.extend(float(x) for _, x, _ in _numeric_claims(d["task"]))
    return vals


def _vocab_errors(C, expl):
    errs = []
    keys, modes = set(C.keys), set(C.modes)
    for c in expl["claims"]:
        t = c.get("type")
        if t not in CLAIM_TYPES:
            errs.append(f"unknown claim type {t}")
            continue
        if c.get("support") not in ("supported", "withheld"):
            errs.append(f"unknown support {c.get('support')}")
        if c["support"] == "withheld":
            if c["contract_id"] not in C.withheld_t:
                errs.append(f"unknown withheld template {c['contract_id']}")
            continue
        if c["contract_id"] not in C.claims or C.claims[c["contract_id"]]["claim_type"] != t:
            errs.append(f"claim {c.get('id')} has contract {c.get('contract_id')} not of type {t}")
        if t == "capability" and (c["subject"] not in keys or c["value"] not in CAP_VALUES):
            errs.append(f"bad capability claim {c['subject']}={c['value']}")
        if t == "wheel_mode" and c["value"] not in modes | {"not_ok"}:
            errs.append(f"bad wheel mode {c['value']}")
        if t == "cause" and c["contract_id"] == "claim-cause-wheel" and c["value"] not in modes - {"ok"}:
            errs.append(f"bad cause mode {c['value']}")
    return errs


def _decision_errors(C, expl):
    d = expl.get("decision")
    if not d:
        return []
    cap = {c["subject"]: c["value"] for c in expl["claims"] if c["type"] == "capability" and c["support"] == "supported"}
    req = d["required"]
    if any(cap.get(k) == "unreliable" for k in req):
        want = "no_go"
    elif all(cap.get(k) == "reliable" for k in req):
        want = "go"
    else:
        want = "probe"
    return [] if d["outcome"] == want else [f"decision {d['outcome']} inconsistent with capability claims ({want})"]


def verify(C, expl):
    """Check that expl['text'] realizes exactly expl['claims'] (+ withheld/decision/probe). Returns dict."""
    reasons = _vocab_errors(C, expl)
    if reasons:
        return {"accepted": False, "reasons": reasons}
    plan = to_answer_plan(C, expl)
    text = expl["text"]
    vr = verify_final_text(plan, text)
    reasons += [f"unlicensed sentence: {s}" for s in vr.unsupported_sentences]
    expected = Counter([c.proposition for c in plan.claims] + list(plan.not_established))
    got = Counter(_sentences(text)) if plan.claims or plan.not_established else Counter()
    for s, k in expected.items():
        if got.get(s, 0) != k:
            reasons.append(f"planned sentence realized {got.get(s, 0)}x (expected {k}): {s}")
    allowed = allowed_numbers(C, expl)
    for tok, val, dec in _numeric_claims(text):
        if not _number_is_licensed(val, dec, allowed):
            reasons.append(f"unlicensed number: {tok}")
    _, withheld, _ = realize_sentences(C, expl)
    if list(expl.get("withheld", [])) != withheld:
        reasons.append("withheld list does not match the withheld claims")
    reasons += _decision_errors(C, expl)
    return {"accepted": not reasons, "reasons": list(dict.fromkeys(reasons))}

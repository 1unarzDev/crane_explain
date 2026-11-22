import os, pathlib
os.environ["CAPXFER_STUDY_CONSTANTS"] = str(pathlib.Path(__file__).parent / "fixtures" / "study_constants_e.yaml")
"""WS-E scoring: automatic, truth-only, and not gameable by wording, padding or forged claims."""
import copy

from capxfer import explain as X
from capxfer.explain._stub import SIDEWAYS_DOCK, case


def setup(name="singleton_roller", dec=None):
    c = case(name)
    e = X.explain(c["pred"], c["intervals"], c["wheel_sets"], c["ood_flag"], c["asm"], c["rung"], decision=dec)
    return c, e


def test_rewording_fails_closed():
    c, e = setup()
    base = X.score(e, c["labels"], c["y_true"])
    for mutate in (
        lambda t: t + " The floor is slippery.",                          # extra unlicensed cause sentence
        lambda t: t.replace("roller-locked", "worn out", 1),             # paraphrase
        lambda t: t.replace("0.49", "0.69", 1),                          # changed number
        lambda t: t.split(". ", 1)[1],                                   # dropped sentence
        lambda t: t.replace("M1:fl being", "a slippery floor and M1:fl being"),
    ):
        bad = copy.deepcopy(e)
        bad[0]["text"] = mutate(bad[0]["text"])
        assert not X.verify(bad[0])["accepted"]
        s = X.score(bad, c["labels"], c["y_true"])
        assert s["verification_failure_rate"] == 1.0 and s["correct_claim_rate"] == 0.0
        assert s["correct_claim_rate"] < base["correct_claim_rate"]


def test_sentence_order_does_not_matter():
    c, e = setup()
    re = copy.deepcopy(e)
    sents = re[0]["text"].split(". ")
    re[0]["text"] = ". ".join(reversed([s.rstrip(".") for s in sents])) + "."
    assert X.verify(re[0])["accepted"]
    a, b = X.score(e, c["labels"], c["y_true"]), X.score(re, c["labels"], c["y_true"])
    import json
    strip = lambda d: json.dumps({k: v for k, v in d.items() if k != "per_explanation"}, sort_keys=True)  # noqa: E731
    assert strip(a) == strip(b)


def test_duplicate_claims_and_forged_vocabulary():
    c, e = setup()
    base = X.score(e, c["labels"], c["y_true"])
    dup = copy.deepcopy(e)
    dup[0]["claims"] = dup[0]["claims"] + copy.deepcopy(dup[0]["claims"])
    from capxfer.explain.realize import render
    from capxfer.explain import load_contracts
    dup[0]["text"] = render(load_contracts(), dup[0])
    s = X.score(dup, c["labels"], c["y_true"])
    assert s["correct_claim_rate"] == base["correct_claim_rate"]
    forged = copy.deepcopy(e)
    forged[0]["claims"].append(dict(id="x", contract_id="claim-ood", type="cause", subject="M1:fl", value="weak",
                                    support="supported", numbers={}, extra={"deficit": ["P2+"]}, subject_idx=0))
    assert not X.verify(forged[0])["accepted"]


def test_inconsistent_decision_rejected():
    c, e = setup(dec=SIDEWAYS_DOCK)
    bad = copy.deepcopy(e)
    bad[0]["decision"]["outcome"] = "go"
    from capxfer.explain.realize import render
    from capxfer.explain import load_contracts
    bad[0]["text"] = render(load_contracts(), bad[0])
    assert not X.verify(bad[0])["accepted"]


def test_blanket_abstention_and_blanket_assertion_each_lose():
    c, e = setup()
    from capxfer.explain.realize import render
    from capxfer.explain import load_contracts
    C = load_contracts()
    ab = copy.deepcopy(e)
    ab[0]["claims"] = []
    ab[0]["withheld"] = []
    ab[0]["text"] = render(C, ab[0])
    s_ab = X.score(ab, c["labels"], c["y_true"])
    s = X.score(e, c["labels"], c["y_true"])
    assert s_ab["unsupported_cause_rate"] == 0.0 and s_ab["informativeness"] == 0.0 and s_ab["abstention_rate"] == 1.0
    assert s["correct_informativeness"] > s_ab["correct_informativeness"]
    # the always-assert point estimate on an OOD trial is informative but misattributes
    co = case("ood_low_mu")
    pt = X.point_estimate(co["pred"], co["asm"], co["rung"])
    s_pt = X.score(pt, co["labels"], co["y_true"])
    assert s_pt["informativeness"] == 1.0 and s_pt["ood_misattribution"] == 1.0


def test_scorer_threshold_comes_from_scorer_not_explanation():
    c, e = setup()
    forged = copy.deepcopy(e)
    forged[0]["tau"] = 0.3       # the explanation cannot move the scoring threshold
    from capxfer.explain.realize import render
    from capxfer.explain import load_contracts
    forged[0]["text"] = render(load_contracts(), forged[0])
    s = X.score(forged, c["labels"], c["y_true"])
    assert s["correct_claim_rate"] == X.score(e, c["labels"], c["y_true"])["correct_claim_rate"]


def test_mixed_batch_by_ood_kind():
    parts = [case(n) for n in ("healthy", "ood_payload", "ood_low_mu", "singleton_roller")]
    import numpy as np
    expl, pts = [], []
    for p in parts:
        expl += X.explain(p["pred"], p["intervals"], p["wheel_sets"], p["ood_flag"], p["asm"], p["rung"])
        pts += X.point_estimate(p["pred"], p["asm"], p["rung"])
    labels = dict(mode=np.concatenate([p["labels"]["mode"] for p in parts]),
                  ood=[p["labels"]["ood"] for p in parts])
    y = np.concatenate([p["y_true"] for p in parts])
    s, sp = X.score(expl, labels, y), X.score(pts, labels, y)
    assert s["n_ood"] == 2 and s["ood_misattribution_by_kind"] == {"low_mu": 0.0, "payload": 0.0}
    assert sp["ood_misattribution_by_kind"] == {"low_mu": 1.0, "payload": 1.0}
    assert s["unsupported_cause_rate"] == 0.0 and sp["unsupported_cause_rate"] == 0.5

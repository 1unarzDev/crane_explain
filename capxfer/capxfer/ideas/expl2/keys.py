"""EXPL2 -> SIMNUM key values for WS-FIG (reads only ~/.cache/capxfer/EXPL2/expl2_results.json).

  python3 -m capxfer.ideas.expl2.keys        (no torch needed)
Writes ~/.cache/capxfer/EXPL2/expl2_keys.json: {key: {tex, value, file, path, sha256, meaning}}.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

OUT = pathlib.Path.home() / ".cache/capxfer/EXPL2"
SRC = OUT / "expl2_results.json"


def get(d, path):
    for p in path.split("/"):
        d = d[p]
    return d


def main():
    raw = SRC.read_bytes()
    d = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()
    K = {}

    def add(key, path, meaning, fmt="{:.3f}"):
        v = get(d, path)
        K[key] = dict(tex=fmt.format(v), value=v, file=str(SRC), path=path, sha256=sha, meaning=meaning)

    op = "id/{t}/operating_point/{a}/{m}"
    # ---- the 4 requested keys (CERT, alpha = 0.10, ID test modules; wheel claims do not depend on tau)
    add("expl_wheel_named_rate", op.format(t="0.5", a="CERT", m="wheel_named_rate_faulted"),
        "truly faulted post-split modules on which the certificate explanation names >= 1 wheel as faulted")
    add("expl_named_wheel_faulty", op.format(t="0.5", a="CERT", m="named_wheel_precision"),
        "precision of named-faulted wheels (named wheel truly faulted)")
    add("expl_mode_in_set", op.format(t="0.5", a="CERT", m="mode_in_set_named_faulty"),
        "named, truly faulted wheels whose stated mode set contains the injected mode")
    for t, s in (("0.5", "_t05"), ("0.8", "_t08")):
        add("expl_binding_correct" + s, f"id/{t}/binding_CERT/notcert_unsafe/binding_hits_faulted_wheel",
            "modules not certified and truly unsafe: binding hypothesis faults a truly faulted wheel")
        add("expl_binding_faultset_exact" + s, f"id/{t}/binding_CERT/notcert_unsafe/binding_fault_set_exact",
            "same rows: binding hypothesis' faulted-wheel set equals the true one")
    K["expl_binding_correct"] = dict(K["expl_binding_correct_t08"], meaning=K["expl_binding_correct_t08"]["meaning"]
                                     + " (tau = 0.8; the _t05 value is also given)")
    # ---- better-defined replacements / companions
    add("expl_mode_set_cov", op.format(t="0.5", a="CERT", m="set_cov_faulted_wheels"),
        "all truly faulted wheels: projected mode set contains the injected mode")
    add("expl_pe_named_wheel_faulty", op.format(t="0.5", a="PE", m="named_wheel_precision"), "PE baseline precision")
    add("expl_pe_mode_in_set", op.format(t="0.5", a="PE", m="mode_in_set_named_faulty"), "PE: argmax mode = truth")
    add("expl_uncal_named_wheel_faulty", op.format(t="0.5", a="UNCAL", m="named_wheel_precision"), "UNCAL precision")
    for t, s in (("0.5", "_t05"), ("0.8", "_t08")):
        for a, lo in (("CERT", "cert"), ("UNCAL", "uncal"), ("PE", "pe"), ("PEC", "pec")):
            add(f"expl_{lo}_inf{s}", op.format(t=t, a=a, m="informativeness"),
                "post-split modules with a CORRECT definite can/cannot-strafe claim")
            add(f"expl_{lo}_err{s}", op.format(t=t, a=a, m="decision_claim_error"),
                "wrong definite can/cannot-strafe claims / definite claims", fmt="{:.4f}")
            add(f"expl_{lo}_ccr{s}", op.format(t=t, a=a, m="correct_claim_rate_micro"), "correct-claim rate (micro)")
        m = f"id/{t}/matched/decision"
        for a, lo in (("UNCAL", "uncal"), ("PE", "pe"), ("PEC", "pec")):
            for y, fmt in (("decision_claim_error", "{:.4f}"), ("false_go", "{:.3f}")):
                try:
                    v = get(d, f"{m}/{a}/at_cert_definite_rate/{y}")
                except KeyError:
                    continue
                if v == v:
                    add(f"expl_matched_{lo}_{'err' if y.startswith('dec') else 'fg'}{s}",
                        f"{m}/{a}/at_cert_definite_rate/{y}",
                        f"{a} swept to CERT's definite-claim rate: {y}", fmt=fmt)
        add(f"expl_cert_fg{s}", f"{m}/CERT/at_cert_definite_rate/false_go", "CERT false-go (= SPLIT T1)")
        add(f"expl_cert_err{s}_matched", f"{m}/CERT/at_cert_definite_rate/decision_claim_error", "CERT", fmt="{:.4f}")
        add(f"expl_cert_definite{s}", f"{m}/CERT/at_cert_definite_rate/x", "CERT definite-claim rate")
    add("expl_matched_pe_wheel_prec", "id/0.5/matched/wheels/PE/at_cert_wheel_informativeness/named_wheel_precision",
        "PE wheel naming swept (p_conf) to CERT's wheel informativeness: precision")
    add("expl_matched_uncal_wheel_prec",
        "id/0.5/matched/wheels/UNCAL/at_cert_wheel_informativeness/named_wheel_precision", "UNCAL, same")
    add("expl_wheel_inf_cert", "id/0.5/matched/wheels/CERT/at_cert_wheel_informativeness/x",
        "faulted modules with >= 1 correctly named faulted wheel (CERT)")
    # ---- OOD (consequential modules, tau = 0.8 and 0.5)
    for kind, kk in (("payload", "pay"), ("low_mu", "lowmu")):
        for t, s in (("0.5", "_t05"), ("0.8", "_t08")):
            base = f"ood/{kind}/by_tau/{t}/consequential/operating_point"
            for a, lo in (("CERT", "cert"), ("PE", "pe"), ("UNCAL", "uncal")):
                add(f"expl_ood_{kk}_{lo}_cause{s}", f"{base}/{a}/cause_claim_rate",
                    "OOD-consequential modules with a (necessarily wrong) wheel-cause claim")
                add(f"expl_ood_{kk}_{lo}_fg{s}", f"{base}/{a}/false_go", "false-go on OOD-consequential modules")
        add(f"expl_ood_{kk}_n_cons", f"ood/{kind}/n_consequential", "n modules", fmt="{:d}")
    add("expl_ood_lowmu_healthy_blame_cert",
        "ood/low_mu/by_tau/0.8/consequential_healthy_wheels/operating_point/CERT/wheel_named_rate_healthy",
        "low-mu consequential modules with all wheels healthy: any wheel named faulted (CERT)")
    add("expl_ood_lowmu_healthy_blame_pe",
        "ood/low_mu/by_tau/0.8/consequential_healthy_wheels/operating_point/PE/wheel_named_rate_healthy", "same, PE")
    add("expl_n_modules", "id/0.5/operating_point/CERT/n", "ID test modules", fmt="{:d}")
    # ---- design properties (NOT results)
    for t, s in (("0.5", "_t05"), ("0.8", "_t08")):
        add(f"expl_design_verif_fail{s}", f"validation_id/{t}/CERT/wse_verification_failure_rate",
            "DESIGN PROPERTY: WS-E verifier rejects 0 realised texts (by construction)")
    (OUT / "expl2_keys.json").write_text(json.dumps(K, indent=1))
    for k, v in K.items():
        print(k, v["tex"])


if __name__ == "__main__":
    main()

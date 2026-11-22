"""Compact key-row tables from results_i3_v4.json (I3.md numbers come from here)."""
import json
import sys

from .seq import ROOT

ROWS = ["fixed/P1P5", "fixed/P2", "fixed/all", "C1_strafe_first/raw", "C1_strafe_first/naive", "C1_strafe_first/exact",
        "C4_P2_first/naive", "C4_P2_first/std", "C6_prior_P2half_P2/naive", "C6_prior_P2half_P2/bonf",
        "C6_prior_P2half_P2/std"]
K = ("false_go_given_inf", "certify_given_feas", "refute_given_inf", "abstain", "decision_error", "time_mean")


def v(x):
    return x["mean"] if isinstance(x, dict) and "mean" in x else x


def show(title, res, group=None):
    print(f"\n#### {title}" + ("" if group is None else f" [group {group}]"))
    print("| policy | FG|inf | cert|feas | refute|inf | abstain | dec.err | time s |\n|---|---|---|---|---|---|---|")
    for r in ROWS:
        x = res.get(r)
        if x is None:
            continue
        if group is not None:
            x = x.get("by_group", {}).get(str(group)) or x.get("by_group", {}).get(group)
            if x is None:
                continue
        print(f"| {r} | " + " | ".join(f"{v(x[k]):.3f}" if k != "time_mean" else f"{v(x[k]):.1f}" for k in K) + " |")


def main():
    R = json.loads((ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results_i3_v4.json")).read_text())
    for tau, o in R.items():
        for g in (None, 1):
            show(f"{tau} assembly-level exchangeable", o["exchangeable"], g)
        show(f"{tau} trial-level exchangeable", o["trial_level"], 1)
        for ty in ("single", "pair"):
            show(f"{tau} held-out {ty} (n_inf={o['loao'][ty]['fixed/P2']['n_inf']})", o["loao"][ty])
        for nm in ("dev_ood_lowmu_v3", "dev_ood_payload_v3"):
            for k in ("all", "ood_consequential"):
                if nm in o and k in o[nm]:
                    show(f"{tau} {nm}/{k} (n_inf={o[nm][k]['fixed/P2']['n_inf']}, n={o[nm][k]['fixed/P2']['n']})",
                         o[nm][k])
        if "shift" in o:
            for k, r in o["shift"].items():
                show(f"{tau} shift: A score fn, {k}", r)
                show(f"{tau} shift: A score fn, {k}", r, 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""docking_runs.csv (+ the sealed arm key, opened only after collection) -> the docking CSV run_frozen reads.

  docking_to_analysis.py manual/docking_runs.csv --arm-key SEALED/dock_arm_key.csv --out derived/docking.csv
                         [--dock-runsheet RUN_SHEETS/dock_runsheet.csv] [--auto auto_outcomes.csv]

Rules (DOCKING_PROTOCOL §5, §7): a technically invalid run (invalid_reason != none) is dropped when a valid repeat of it
exists (repeat_of), else it stays as a failure flagged `invalid_unrepeated`; an abort caused by a person in the arena is
replaced by its repeat; every other abort is a failure. Only complete blocks enter (the session may stop at a block
boundary). Success comes from the automatic tag-based outcome (--auto: run_global, outcome, final_pose_error_m) when it is
given (WS-F scorer), else from the operator's call (success_source = operator). Output columns: arm, run, block,
start_pose_id, outcome, success_source (analysis file); the .audit.csv
next to it keeps every row with its exclusion reason.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import packet_lib as PL  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs")
    ap.add_argument("--arm-key", required=True)
    ap.add_argument("--dock-runsheet")
    ap.add_argument("--auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    hdr, rows = PL.read_csv(a.runs)
    if hdr != PL.header("docking_runs"):
        sys.exit("docking_runs.csv header differs from the template")
    runs = [dict(zip(hdr, r)) for r in rows if r and any(x.strip() for x in r)]
    key = {r[0]: r[1] for r in PL.read_csv(a.arm_key)[1] if r}
    auto = {}
    if a.auto:
        ah, ar = PL.read_csv(a.auto)
        auto = {int(r[ah.index("run_global")]): dict(zip(ah, r)) for r in ar if r}
    blocks_n = defaultdict(int)
    if a.dock_runsheet:
        dh, dr = PL.read_csv(a.dock_runsheet)
        for r in dr:
            d = dict(zip(dh, r))
            if d["what"] == "DOCK":
                blocks_n[int(d["block"])] += 1
    repeated = {int(r["repeat_of"]) for r in runs if r["repeat_of"].strip()}
    out = []
    for r in runs:
        rg = int(r["run_global"])
        excl = ""
        if r["invalid_reason"] != "none" or (r["operator_outcome"] == "abort" and r["invalid_reason"] == "person_in_arena"):
            excl = "replaced_by_repeat" if rg in repeated else "invalid_unrepeated"
        au = auto.get(rg)
        outcome = au["outcome"] if au else r["operator_outcome"]
        out.append(dict(arm=key.get(r["arm_code"], f"UNKNOWN_{r['arm_code']}"), run=rg, block=int(r["block"]),
                        start_pose_id=r["start_pose_id"], success=int(outcome == "success"), outcome=outcome,
                        success_source="auto" if au else "operator",
                        final_pose_error=(au or {}).get("final_pose_error_m", ""), excluded=excl))
    # complete blocks only
    have = defaultdict(set)
    for o, r in zip(out, runs):
        if o["excluded"] != "replaced_by_repeat":
            have[o["block"]].add(int(r["run_in_block"]))
    for o in out:
        need = blocks_n.get(o["block"])
        if need and len(have[o["block"]]) < need and not o["excluded"]:
            o["excluded"] = "incomplete_block"
    keep = [o for o in out if o["excluded"] in ("", "invalid_unrepeated")]
    for o in keep:
        if o["excluded"] == "invalid_unrepeated":
            o["success"] = 0
    audit = str(Path(a.out).with_suffix("")) + ".audit.csv"
    with open(audit, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    cols = ["arm", "run", "success", "final_pose_error", "block", "start_pose_id", "outcome", "success_source"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(keep)
    by = defaultdict(lambda: [0, 0])
    for o in keep:
        by[o["arm"]][0] += o["success"]
        by[o["arm"]][1] += 1
    print("  ".join(f"{k}: {s}/{n}" for k, (s, n) in sorted(by.items())), f"(analysis rows: {len(keep)} in {a.out}; every row with its exclusion in {audit})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""One-command processing of a received binned CSV session (WS-KIT2): arrival check -> adapter -> feature/target caches.

  capxfer/field/process_session.sh SESSION_DIR [OUT_ROOT] [--seed-file SEED] [--bundle BUNDLE] [--force]
  python -m capxfer.field.process_session SESSION_DIR [--out OUT_ROOT] ...

Writes OUT_ROOT/<session>/ (default ~/capxfer_derived/<session>/; never inside the session folder):
  arrival_check.txt / .json   capxfer.field.arrival_check report (stops here on FAIL unless --force)
  validate.json               csv_adapter.verify_binned_session verdict
  qc.csv                      per trial: block, layout, declared fault, schedule_match (+ reasons), max gap, min truth
                              coverage, adapter QC notes
  raw_<k>.pt                  the RawTrial batches (torch.save; what builders / predict / choose_module read)
  feat_<k>.pt                 featurize(raw, rung, pids of the frozen bundle, with_targets=True): the cached Feat
  targets.csv                 trial_id, block, layout, fault, rho_<14 keys>, z (evaluator-only truth targets)
  SHA256SUMS                  of every file written
With --seed-file (the sealed seed, analysis side only) the labels of record are joined from the sealed schedule.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from ..probes import TARGET_KEYS
from . import arrival_check as AC
from . import csv_adapter as CA

DEFAULT_OUT = Path.home() / "capxfer_derived"
DEFAULT_BUNDLE = Path.home() / ".cache" / "capxfer" / "I3F" / "bundle_W_provisional"
STRAFE = (TARGET_KEYS.index("P2+"), TARGET_KEYS.index("P2-"))


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def process(session, out_root=DEFAULT_OUT, seed_file=None, bundle=DEFAULT_BUNDLE, force=False, log=print):
    import torch
    import yaml
    from ..featurize import featurize
    session = Path(session).resolve()
    out = Path(out_root).expanduser().resolve() / session.name
    if out == session or session in out.parents:
        raise SystemExit("refusing to write inside the session folder")
    out.mkdir(parents=True, exist_ok=True)
    r = AC.check_session(session)
    (out / "arrival_check.txt").write_text(AC.report_text(r) + "\n")
    (out / "arrival_check.json").write_text(json.dumps(r, indent=1, default=str))
    log(AC.report_text(r))
    if r["verdict"] == "FAIL" and not force:
        log(f"arrival check FAIL: stopped (send the fixes above to the team; --force to process anyway). {out}")
        return dict(out=out, verdict="FAIL", processed=False)
    v = CA.verify_binned_session(session)
    (out / "validate.json").write_text(json.dumps(v, indent=1))
    sched = CA.schedule_from_seed(seed_file) if seed_file else None
    raws = CA.load_binned_session(session, sched, allow_unverified=True, verdict=v)
    rung, pids = "E1", ("P1", "P5", "P2")
    if bundle and (Path(bundle) / "bundle.yaml").is_file():
        by = yaml.safe_load((Path(bundle) / "bundle.yaml").read_text())
        rung, pids = by.get("rung", rung), tuple(by.get("pids", pids))
    qrows, trows = [], []
    qc = {}
    for raw in raws:
        for tid, note in raw["meta"]["qc"]:
            qc.setdefault(tid, []).append(note)
    for k, raw in enumerate(raws):
        torch.save(raw, out / f"raw_{k}.pt")
        F = featurize(raw, rung=rung, pids=pids, with_targets=True)
        torch.save(F, out / f"feat_{k}.pt")
        lab = raw["meta"]["labels"]
        y = F["y"].cpu().numpy()
        for i, tid in enumerate(raw["meta"]["trial_ids"]):
            stem = tid.rsplit(":", 2)[1]
            gap, cover = raw["meta"]["csv_checks"][tid]
            tr = raw["meta"]["trials_rows"][stem]
            fault = ";".join(f"{c}:{m}" for c, m in zip(raw["meta"]["assembly"].wheel_corner,
                                                        lab["mode_echo"][i].tolist()) if m) or "ok"
            qrows.append([tid, tr["block"], CA.layout_str(raw["meta"]["assembly"].modules), fault,
                          int(lab["schedule_match"][i]), "; ".join(lab["schedule_mismatch"][i]), f"{gap:.3f}",
                          f"{min(cover.values()):.3f}", "; ".join(qc.get(stem, []))])
            trows.append([tid, tr["block"], CA.layout_str(raw["meta"]["assembly"].modules), fault]
                         + [("" if not math.isfinite(v_) else f"{v_:.6f}") for v_ in y[i]]
                         + [f"{float(np.min(y[i][list(STRAFE)])):.6f}"])
    with open(out / "qc.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["trial_id", "block", "layout", "fault_declared_codes", "schedule_match", "mismatch", "max_gap_s",
                    "min_truth_coverage", "adapter_qc"])
        w.writerows(qrows)
    with open(out / "targets.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["trial_id", "block", "layout", "fault_declared_codes"] + [f"rho_{k}" for k in TARGET_KEYS] + ["z"])
        w.writerows(trows)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{_sha(p)}  {p.name}\n" for p in files))
    n = sum(len(x["meta"]["trial_ids"]) for x in raws)
    log(f"processed {n} probe trials in {len(raws)} group(s) -> {out}")
    return dict(out=out, verdict=r["verdict"], processed=True, n_trials=n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed-file")
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    r = process(a.session, a.out, a.seed_file, a.bundle, a.force)
    return 0 if r["processed"] else 2


if __name__ == "__main__":
    sys.exit(main())

"""SIMULATION equivalence check of the minimal 20 Hz binned field format (WS-KIT2; capxfer/field_kit_v2/CSV_FORMAT.md).

Same simulated trials, same sensor and camera realisation, two paths:
  FULL   the in-memory 100 Hz RawTrial (capxfer.sensors.to_raw; camera frames held from receipt, truth_latency 0.15 s)
  BINNED the team's pre-processing (csv_adapter.binned_rows: 50 ms count differences, bin-mean duty / IMU, camera poses
         shifted by the team's latency estimate and interpolated to the bin centres) -> binned.csv on disk ->
         csv_adapter.load_binned_session.
Both go through capxfer.featurize (frozen bundle rung E1, P1+P5+P2), the frozen W models (structured_ensemble,
ensemble) and the SPLIT (a) certificate (bundle wheel sets + bands). Reported deltas: features, targets rho / z and the
tau decisions, diagnosis recall / false alarms, black-box coverage / width, certificate scores and go / no-go.

  PYTHONPATH=. .venv/bin/python -m capxfer.field.binned_validation [--n 48] [--out ~/.cache/capxfer/KIT2/validation.json]

SIMULATION ONLY. No field data are read or written.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .. import assembly as A
from ..featurize import featurize
from ..probes import TARGET_KEYS
from . import choose_module as CM
from . import csv_adapter as CA

BUNDLE = Path.home() / ".cache" / "capxfer" / "I3F" / "bundle_W_provisional"
OUT = Path.home() / ".cache" / "capxfer" / "KIT2" / "validation.json"
TYPES = {"single": [("M1", 0, 0, 0)], "pair": list(A.PHYSICAL_TYPES["pair"]), "L3": list(A.PHYSICAL_TYPES["L3"]),
         "square4": list(A.PHYSICAL_TYPES["square4"])}
TAU = 0.8
STRAFE = (TARGET_KEYS.index("P2+"), TARGET_KEYS.index("P2-"))
VARIANTS = {"baseline": dict(lat_est_err=0.0, have_acc=True),
            "latency_error_+30ms": dict(lat_est_err=0.03, have_acc=True),
            "latency_error_-30ms": dict(lat_est_err=-0.03, have_acc=True),
            "no_ax_ay": dict(lat_est_err=0.0, have_acc=False)}


def faults(B, n, rng):
    """25 % healthy, 50 % one passive (coast/brake), 25 % one weak wheel (g 0.3-0.9)."""
    m = np.zeros((B, n), np.int64)
    g = np.ones((B, n), np.float32)
    for b in range(B):
        u = rng.random()
        if u < 0.25:
            continue
        w = int(rng.integers(n))
        if u < 0.75:
            m[b, w] = int(rng.choice([2, 3]))
        else:
            m[b, w], g[b, w] = 1, float(rng.uniform(0.3, 0.9))
    return m, g


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    return dict(max_abs=float(np.abs(d).max()), rms=float(np.sqrt((d ** 2).mean())),
                rel_rms=float(np.sqrt((d ** 2).mean()) / max(np.sqrt((a ** 2).mean()), 1e-12)))


def _to_binned_raw(binned, spec, tmp, name):
    trials = [dict(trial_id=f"T{b:03d}", seq=0, block="SYSID", layout=CA.layout_str(spec), fault="ok",
                   imu_module=spec[0][0], truth_tag=spec[0][0], aborted=0) for b in range(len(binned))]
    s = CA.write_binned_session(tmp, f"SIMULATION_{name}", trials, {f"T{b:03d}": binned[b] for b in range(len(binned))})
    raws = CA.load_binned_session(s, None, allow_unverified=True)
    assert len(raws) == 1 and len(raws[0]["meta"]["trial_ids"]) == len(binned)
    return raws[0]


def evaluate(raw, b, models, bb):
    """Features, targets, model outputs and certificate scores of one RawTrial batch."""
    F = featurize(raw, rung=b["rung"], pids=tuple(b["pids"]), with_targets=True)
    P, zs = models.predict(raw)
    ys = bb.predict(featurize(raw, rung=b["rung"], pids=tuple(b["pids"]), with_targets=False))["y_samples"]
    ys = ys.detach().cpu().numpy().astype(float)                                       # (B, S, 14)
    asm = raw["meta"]["assembly"]
    mids = [m for m, *_ in asm.modules]
    s = np.array([[CM.certificate_bounds(P[k], asm.wheel_module, mids, b["_ws"], b["_bands"])[m] for m in mids]
                  for k in range(P.shape[0])])
    return dict(F=F, y=F["y"].cpu().numpy().astype(float), P=P, zs=zs, ys=ys, s=s)


def compare(full, binr, modes):
    out = {}
    Ff, Fb = full["F"], binr["F"]
    for c, name in enumerate(("ref_speed", "enc_speed", "duty")):
        out[f"wheel.{name}"] = _rel(Ff["wheel_x"][..., c].numpy(), Fb["wheel_x"][..., c].numpy())
    for c, name in enumerate(("cmd_vx", "cmd_vy", "cmd_wz", "gyro_z", "acc_x", "acc_y")):
        out[f"body.{name}"] = _rel(Ff["body_x"][..., c].numpy(), Fb["body_x"][..., c].numpy())
    yf, yb = full["y"], binr["y"]
    d = yb - yf
    out["rho"] = dict(rms=float(np.sqrt(np.nanmean(d ** 2))), bias=float(np.nanmean(d)),
                      max_abs=float(np.nanmax(np.abs(d))),
                      per_key_rms={k: float(np.sqrt(np.nanmean(d[:, j] ** 2))) for j, k in enumerate(TARGET_KEYS)})
    h = ~(modes != 0).any(1)                     # v4 (D17): relative capability, each path's own healthy median
    rf = yf / np.nanmedian(yf[h], 0) if h.any() else yf
    rb = yb / np.nanmedian(yb[h], 0) if h.any() else yb
    zf, zb = np.min(rf[:, STRAFE], 1), np.min(rb[:, STRAFE], 1)
    za_f, za_b = np.min(yf[:, STRAFE], 1), np.min(yb[:, STRAFE], 1)
    out["z_absolute"] = dict(decision_agree_tau08=float(np.mean((za_f >= TAU) == (za_b >= TAU))),
                             healthy_median=float(np.median(za_f[h])) if h.any() else math.nan)
    out["z"] = dict(rms=float(np.sqrt(np.mean((zb - zf) ** 2))), max_abs=float(np.max(np.abs(zb - zf))),
                    decision_agree_tau08=float(np.mean((zf >= TAU) == (zb >= TAU))),
                    decision_agree_tau05=float(np.mean((zf >= 0.5) == (zb >= 0.5))))
    for tag, r in (("full", full), ("binned", binr)):
        am = r["P"].argmax(-1)
        fw = modes != 0
        out[f"diag.{tag}"] = dict(recall_exact=float((am[fw] == modes[fw]).mean()) if fw.any() else math.nan,
                                  recall_any_fault=float((am[fw] != 0).mean()) if fw.any() else math.nan,
                                  false_alarm=float((am[~fw] != 0).mean()))
        lo, hi = np.quantile(r["ys"], 0.05, 1), np.quantile(r["ys"], 0.95, 1)
        out[f"bb.{tag}"] = dict(coverage90_vs_full_truth=float(np.nanmean((yf >= lo) & (yf <= hi))),
                                coverage90_vs_own_truth=float(np.nanmean((r["y"] >= lo) & (r["y"] <= hi))),
                                width90=float(np.nanmean(hi - lo)))
    out["diag.argmax_agree"] = float((full["P"].argmax(-1) == binr["P"].argmax(-1)).mean())
    out["diag.prob_max_abs"] = float(np.abs(full["P"] - binr["P"]).max())
    out["bb.q05z_rms"] = float(np.sqrt(np.mean((np.quantile(full["zs"], 0.05, 1) - np.quantile(binr["zs"], 0.05, 1)) ** 2)))
    sf, sb = full["s"], binr["s"]
    fin = np.isfinite(sf) & np.isfinite(sb)
    out["cert"] = dict(score_rms=float(np.sqrt(np.mean((sf[fin] - sb[fin]) ** 2))) if fin.any() else math.nan,
                       go_agree_tau08=float(np.mean((sf >= TAU) == (sb >= TAU))),
                       go_full=float(np.mean(sf >= TAU)), go_binned=float(np.mean(sb >= TAU)), n_modules=int(sf.size))
    return out


def run(n=48, seed=20261121, bundle=BUNDLE, out=OUT, variants=None, log=print):
    torch.set_num_threads(4)
    b = CM.load_bundle(bundle)
    models = CM.Models(b, "cpu")
    bb = models._bb
    res = dict(note="SIMULATION ONLY: same simulated trials, FULL in-memory 100 Hz path vs the BINNED 20 Hz CSV path",
               bundle=str(bundle), bundle_sha256=b["_sha"], n_per_type=n, seed=seed, rung=b["rung"], pids=b["pids"],
               variants={})
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="kit2val_"))
    for vname, kw in (variants or VARIANTS).items():
        per = {}
        for k, (tname, spec) in enumerate(TYPES.items()):
            asm = A.build(CA.asm_name(spec), spec, A.load_params(CA.PARAMS_PATH))
            rng = np.random.default_rng([seed, k])
            modes, gs = faults(n, asm.n, rng)
            raw, binned, asm = CA.simulate_trials(spec, modes, gs, seed=seed + 17 * k, **kw)
            rb = _to_binned_raw(binned, spec, tmp, f"{vname}_{tname}")
            full = evaluate(raw, b, models, bb)
            binr = evaluate(rb, b, models, bb)
            per[tname] = compare(full, binr, modes)
            log(f"{vname} {tname}: rho rms {per[tname]['rho']['rms']:.4f}, z agree {per[tname]['z']['decision_agree_tau08']:.3f}, "
                f"cert agree {per[tname]['cert']['go_agree_tau08']:.3f} ({time.time() - t0:.0f} s)")
        res["variants"][vname] = per
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--bundle", default=str(BUNDLE))
    a = ap.parse_args(argv)
    run(a.n, bundle=Path(a.bundle), out=Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

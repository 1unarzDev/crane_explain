"""Arrival check of a received binned CSV session (WS-KIT2; capxfer/field_kit_v2/CSV_FORMAT.md). Read-only, < 1 min.

  python -m capxfer.field.arrival_check SESSION_DIR [--json OUT.json] [--params params/module_measured.yaml]

The team does its own pre-processing and does not share its scripts, so every likely processing error is detected from
the data alone. Each check reports PASS / WARN / FAIL (or SKIP) with the measured value and, for every problem, the
likely fix in plain words to send to the team:
  (a) schema: files, columns, 50 ms bins from t = 0, bins per trial, value ranges, blank cells
  (b) command timeline vs the fixed probe routine: t = 0 offset (cross-correlation), swapped / inverted / mis-scaled
      command axes, twist reference point (P5 must rotate about the assembly centre = mean of the module centres)
  (c) healthy trials: wheel speeds vs the AprilTag body twist through the mecanum IK, per wheel: order, sign, units,
      counts-per-rev (scale factor estimate)
  (d) duty vs speed co-movement (sign, wrong column, constant duty)
  (e) yaw continuity (unwrapped) and gz vs the truth yaw rate (sign, units)
  (f) residual truth latency: cross-correlation of the truth body speed with the wheel-odometry body speed
  (g) healthy tracking: absolute truth/command ratio in the holds; WARN below 0.75 (capability is relative, D17)
Overall verdict = the worst check. Thresholds are in THRESH.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from .. import assembly as A
from ..probes import Schedule
from . import csv_adapter as CA

THRESH = dict(cmd_err=0.02, lag_bins=20, speed_scale=(0.85, 1.20), ik_corr=0.8, duty_corr=0.3, yaw_jump=0.5,
              gz_scale=(0.85, 1.15), lat_warn=0.03, lat_fail=0.08, blank_warn=0.02, min_bins_frac=0.98,
              track_warn=0.75)                 # v4 (D17): absolute healthy tracking floor (was 0.94 in the spec)
UNITS = {"rpm (multiply by 2 pi / 60 to get rad/s)": 60 / (2 * math.pi),
         "deg/s (multiply by pi / 180)": 180 / math.pi,
         "counts per second (divide by counts_per_rev and multiply by 2 pi)": None,          # filled with cpr / 2 pi
         "counts per bin (divide by counts_per_rev, multiply by 2 pi, divide by 0.05 s)": None,
         "rad per bin (divide by 0.05 s)": 0.05}
RANK = {"PASS": 0, "SKIP": 0, "WARN": 1, "FAIL": 2}


def _res(name, status, value=None, message="", fix=""):
    return dict(check=name, status=status, value=value, message=message, fix=fix)


def _unit_guess(s, cpr):
    cands = dict(UNITS)
    cands["counts per second (divide by counts_per_rev and multiply by 2 pi)"] = cpr / (2 * math.pi)
    cands["counts per bin (divide by counts_per_rev, multiply by 2 pi, divide by 0.05 s)"] = cpr / (2 * math.pi) * 0.05
    best = min(cands.items(), key=lambda kv: abs(math.log(abs(s) / kv[1])))
    return best[0] if abs(math.log(abs(s) / best[1])) < 0.15 else None


def _truth_twist(b, asm, tag):
    """Body twist (K,3) about the assembly origin from the bin-centre truth poses (central differences)."""
    P = CA.tag_to_origin(np.stack([b["x"], b["y"], np.unwrap(np.nan_to_num(b["yaw"]))], 1), asm, tag)
    P[~np.isfinite(b["x"]) | ~np.isfinite(b["yaw"])] = np.nan
    K = len(P)
    tw = np.full((K, 3), np.nan)
    if K < 3:
        return tw
    dp = (P[2:] - P[:-2]) / (2 * CA.BIN)
    th = P[1:-1, 2]
    c, s = np.cos(th), np.sin(th)
    tw[1:-1] = np.stack([c * dp[:, 0] + s * dp[:, 1], -s * dp[:, 0] + c * dp[:, 1], dp[:, 2]], 1)
    return tw


def check_session(d, params=None):
    d = Path(d)
    params = params or A.load_params(CA.PARAMS_PATH)
    cpr = float(params["motor"]["counts_per_output_rev"])
    radius = float(params["wheel"]["radius_m"])
    out = []
    # ------------------------------------------------------------------ (a) schema
    v = CA.verify_binned_session(d)
    if v["verdict"] == "INVALID":
        out.append(_res("a_schema", "FAIL", v["errors"][:6], "the session cannot be read",
                        "Send trials.csv and binned.csv with exactly the columns of CSV_FORMAT.md (one column per wheel "
                        "and quantity, e.g. M2_fl_speed, M2_fl_duty) and every non-aborted trial present in binned.csv."))
        return dict(session=d.name, verdict="FAIL", checks=out)
    trials = [t for t in CA._read_rows(d / "trials.csv") if str(t["aborted"]) in ("0", "")]
    B = CA.read_binned(d)
    probs, blank = [], {}
    tr = []
    for t in trials:
        spec = CA.parse_layout(t["layout"])
        asm = A.build(CA.asm_name(spec), spec, params)
        sch = Schedule(asm.char_length())
        b = B[t["trial_id"]]
        tb = b["t"]
        K = int(round(sch.T / CA.BIN))
        if abs(tb[0]) > 1e-3 or (len(tb) > 1 and np.abs(np.diff(tb) - CA.BIN).max() > 1e-3):
            dtm = float(np.median(np.diff(tb))) if len(tb) > 1 else math.nan
            probs.append(f"{t['trial_id']}: bins start at {tb[0]:.3f} s with spacing {dtm:.3f} s")
        if len(tb) < THRESH["min_bins_frac"] * K:
            probs.append(f"{t['trial_id']}: {len(tb)} bins, the routine needs {K}")
        for c in CA.wheel_cols(spec) + ["gz", "x", "y", "yaw"]:
            blank.setdefault(c, []).append(float(np.mean(~np.isfinite(b[c]))))
        sp = [b[f"{m}_{c}_speed"] for m, *_ in spec for c in CA.CORNERS]
        du = [b[f"{m}_{c}_duty"] for m, *_ in spec for c in CA.CORNERS]
        if np.nanmax(np.abs(du)) > 1.05:
            probs.append(f"{t['trial_id']}: |duty| up to {np.nanmax(np.abs(du)):.2f} (must be -1..1)")
        tr.append((t, spec, asm, sch, b, np.array(sp), np.array(du)))
    bl = {c: float(np.mean(v_)) for c, v_ in blank.items()}
    worst = max(bl.items(), key=lambda kv: kv[1]) if bl else ("", 0.0)
    if probs:
        out.append(_res("a_schema", "FAIL", probs[:6], "bins are not 50 ms from t = 0, or values out of range",
                        "Bin k must cover [k*0.05, (k+1)*0.05) s with t = k*0.05 (t = 0 at the first command of the "
                        "routine), about 590 bins per trial; duty must be the controller duty in -1..1."))
    elif worst[1] > THRESH["blank_warn"]:
        out.append(_res("a_schema", "WARN", {"column": worst[0], "blank_fraction": worst[1]},
                        f"{worst[1]:.1%} blank cells in {worst[0]}", "Check that sensor logging keeps up (no dropped "
                        "messages); blanks are fine only for real dropouts."))
    else:
        out.append(_res("a_schema", "PASS", {"trials": len(tr), "max_blank_fraction": worst[1]}))
    # ------------------------------------------------------------------ (b) command timeline
    lags, errs, axis_msgs, rc = [], [], [], []
    for t, spec, asm, sch, b, sp, du in tr:
        tb = b["t"]
        Lg = np.stack([b["cmd_vx"], b["cmd_vy"], b["cmd_wz"]], 1)
        best = (math.inf, 0)
        for lag in range(-THRESH["lag_bins"], THRESH["lag_bins"] + 1):
            ref = sch.twist(tb - lag * CA.BIN)
            e = float(np.nanmean(np.abs(Lg - ref)))
            best = min(best, (e, lag))
        errs.append(best[0])
        lags.append(best[1])
        ref = sch.twist(tb - best[1] * CA.BIN)
        for j, ax in enumerate(("cmd_vx", "cmd_vy", "cmd_wz")):
            ok = np.isfinite(Lg[:, j])
            cc = [np.corrcoef(Lg[ok, j], ref[ok, i])[0, 1] if np.std(ref[ok, i]) > 0 and np.std(Lg[ok, j]) > 0 else 0
                  for i in range(3)]
            i = int(np.argmax(np.abs(cc)))
            if abs(cc[i]) > 0.8 and (i != j or cc[i] < 0):
                axis_msgs.append(f"{ax} looks like {'-' if cc[i] < 0 else ''}{('vx', 'vy', 'wz')[i]}")
            elif i == j and abs(cc[i]) > 0.8:
                s = float(np.nansum(Lg[ok, j] * ref[ok, j]) / max(np.nansum(ref[ok, j] ** 2), 1e-12))
                if not 0.9 < s < 1.1:
                    axis_msgs.append(f"{ax} is {s:.3g} x the routine")
        # rotation centre in the P5 holds (truth)
        if len(spec) > 1:
            tw = _truth_twist(b, asm, t.get("truth_tag") or spec[0][0])
            for pid, leg, ta, tb_, twc in sch.windows():
                if pid == "P5":
                    sel = (tb + CA.BIN / 2 >= ta) & (tb + CA.BIN / 2 <= tb_)
                    v_ = np.nanmedian(np.hypot(tw[sel, 0], tw[sel, 1]))
                    rc.append(float(v_ / max(abs(twc[2]), 1e-9)))
    lag = int(np.median(lags)) if lags else 0
    if lags and lag != 0:
        out.append(_res("b_timeline", "FAIL", {"t0_offset_s": lag * CA.BIN, "per_trial_lags": sorted(set(lags))[:8]},
                        f"the logged command matches the routine {lag * CA.BIN:+.2f} s later than t = 0",
                        f"Set t = 0 at the first command of the probe routine (the start of the 0.5 s still period); "
                        f"your t = 0 is {abs(lag) * CA.BIN:.2f} s {'too early' if lag > 0 else 'too late'}."))
    elif axis_msgs:
        out.append(_res("b_timeline", "FAIL", sorted(set(axis_msgs))[:6], "command axes do not match the routine",
                        "Log cmd_vx, cmd_vy, cmd_wz as the assembly twist actually sent, in m/s, m/s, rad/s, x forward, "
                        "y left, yaw counter-clockwise; check axis order, signs and units."))
    elif errs and max(errs) > THRESH["cmd_err"]:
        out.append(_res("b_timeline", "WARN", {"mean_abs_err_max": max(errs)}, "the command differs from the routine",
                        "Check the ramp/hold/rest timing of your probe routine against RUN_PLAN.md."))
    elif rc and np.median(rc) > 0.08:
        out.append(_res("b_timeline", "FAIL", {"p5_centre_offset_m": float(np.median(rc))},
                        "in P5 the assembly turns about a point away from its centre",
                        "The P5 yaw (and every twist) must be about the assembly centre = the mean of the module "
                        "centres, not about the first module; compute the wheel IK about that point."))
    else:
        out.append(_res("b_timeline", "PASS", {"t0_offset_s": 0.0, "mean_abs_err_max": max(errs) if errs else None,
                                               "p5_centre_offset_m": float(np.median(rc)) if rc else None}))
    # ------------------------------------------------------------------ (c) wheel speeds vs truth IK; (d) duty; (e) yaw
    pred, meas, duty, gzs, wzt = {}, {}, {}, [], []
    lat_t, lat_w = [], []
    jumps, track = [], []
    for t, spec, asm, sch, b, sp, du in tr:
        yw = b["yaw"]
        ok = np.isfinite(yw)
        if ok.sum() > 2:
            jumps.append(float(np.max(np.abs(np.diff(yw[ok])))))
        healthy = CA.parse_fault(t["fault"]) == {} and t["block"] != "HOOK"
        tw = _truth_twist(b, asm, t.get("truth_tag") or spec[0][0])
        gzs.append(b["gz"])
        wzt.append(tw[:, 2])
        if not healthy:
            continue
        Lc2 = asm.char_length() ** 2
        for pid, leg, ta, tb_, twc in sch.windows():                # (g) absolute healthy tracking in every hold
            sel = (b["t"] + CA.BIN / 2 >= ta) & (b["t"] + CA.BIN / 2 <= tb_)
            xi = np.nanmean(tw[sel], 0) if sel.any() else np.full(3, np.nan)
            Wd = np.array([1.0, 1.0, Lc2])
            if np.isfinite(xi).all():
                track.append(float((xi * twc * Wd).sum() / (twc * twc * Wd).sum()))
        P = asm.ik(np.nan_to_num(tw), radius)                       # (K, n) wheel speeds the truth twist implies
        P[~np.isfinite(tw).all(1)] = np.nan
        names = [f"{m}_{c}" for m, *_ in spec for c in CA.CORNERS]
        for w, nm in enumerate(names):
            pred.setdefault(nm, []).append(P[:, w])
            meas.setdefault(nm, []).append(sp[w])
            duty.setdefault(nm, []).append(du[w])
        # body speed from the wheel odometry (least squares FK) for the latency check
        J = np.stack([asm.ik(e, radius) for e in np.eye(3)], 1)    # (n, 3)
        fk = np.linalg.lstsq(J, np.nan_to_num(sp), rcond=None)[0].T  # (K, 3)
        lat_t.append(np.hypot(tw[:, 0], tw[:, 1]))
        lat_w.append(np.hypot(fk[:, 0], fk[:, 1]))
    if not pred:
        out.append(_res("c_wheel_ik", "SKIP", None, "no healthy trial in this session"))
        out.append(_res("d_duty_speed", "SKIP", None, "no healthy trial in this session"))
    else:
        names = sorted(pred)
        Pm = {k: np.concatenate(v) for k, v in pred.items()}
        Mm = {k: np.concatenate(v) for k, v in meas.items()}
        Dm = {k: np.concatenate(v) for k, v in duty.items()}
        scale, order, sign = {}, [], []
        for mod in sorted({k.split("_")[0] for k in names}):
            cols = [k for k in names if k.startswith(mod + "_")]
            for j in cols:
                cc = {}
                for i in cols:
                    ok = np.isfinite(Pm[i]) & np.isfinite(Mm[j]) & (np.abs(Pm[i]) > 0.5)
                    cc[i] = float(np.corrcoef(Pm[i][ok], Mm[j][ok])[0, 1]) if ok.sum() > 20 else 0.0
                best = max(cc, key=lambda i: abs(cc[i]))
                if best != j and abs(cc[best]) > abs(cc[j]) + 0.1:
                    order.append(f"your {j} column behaves like our {best}")
                elif cc[j] < -THRESH["ik_corr"]:
                    sign.append(j)
                ok = np.isfinite(Pm[j]) & np.isfinite(Mm[j]) & (np.abs(Pm[j]) > 0.5)
                if ok.sum() > 20:
                    scale[j] = float(np.sum(Mm[j][ok] * Pm[j][ok]) / np.sum(Pm[j][ok] ** 2))
        s_med = float(np.median([abs(v) for v in scale.values()])) if scale else math.nan
        lo, hi = THRESH["speed_scale"]
        if order:
            out.append(_res("c_wheel_ik", "FAIL", order[:6], "wheel columns are in the wrong order",
                            "Map your motor channels to the corner names fl, fr, rl, rr of each module (front-left, "
                            "front-right, rear-left, rear-right, module front up): " + "; ".join(order[:4]) + "."))
        elif sign:
            out.append(_res("c_wheel_ik", "FAIL", {"inverted": sign}, "wheel speed sign inverted",
                            f"Flip the sign of {', '.join(sign)}: a wheel speed is positive when that wheel alone "
                            "would drive its module forward (+x)."))
        elif not lo <= s_med <= hi:
            u = _unit_guess(s_med, cpr)
            fix = (f"Wheel speeds look like {u}; send rad/s." if u else
                   f"Wheel speeds are {s_med:.3g} x what the AprilTag motion implies: check counts_per_rev "
                   f"(we use {cpr:.0f}; yours looks like {cpr / s_med:.0f}) and the 0.05 s bin divisor.")
            out.append(_res("c_wheel_ik", "FAIL", {"scale": s_med, "per_wheel": scale}, "wheel speed scale is wrong",
                            fix))
        else:
            out.append(_res("c_wheel_ik", "PASS", {"scale_median": s_med, "per_wheel": scale}))
        cors = {}
        for k in names:
            ok = np.isfinite(Dm[k]) & np.isfinite(Mm[k])
            cors[k] = float(np.corrcoef(Dm[k][ok], Mm[k][ok])[0, 1]) if ok.sum() > 20 and np.std(Dm[k][ok]) > 0 else 0.0
        neg = [k for k, c in cors.items() if c < -THRESH["duty_corr"]]
        flat = [k for k, c in cors.items() if abs(c) <= THRESH["duty_corr"]]
        if neg:
            out.append(_res("d_duty_speed", "FAIL", {"corr": cors}, "duty moves against the wheel speed",
                            f"Flip the sign of the duty of {', '.join(neg)} (duty must have the sign of the commanded "
                            "wheel direction), or check that duty and speed columns belong to the same wheel."))
        elif flat:
            out.append(_res("d_duty_speed", "FAIL", {"corr": cors}, "duty does not follow the wheel speed",
                            f"The duty of {', '.join(flat)} does not follow its wheel: log the velocity controller's "
                            "duty for THAT wheel, before any fault scaling, averaged over the bin."))
        else:
            out.append(_res("d_duty_speed", "PASS", {"corr_min": min(cors.values())}))
    # (e) yaw continuity + gz vs truth yaw rate
    jm = max(jumps) if jumps else 0.0
    g = np.concatenate(gzs) if gzs else np.zeros(0)
    w = np.concatenate(wzt) if wzt else np.zeros(0)
    ok = np.isfinite(g) & np.isfinite(w) & (np.abs(w) > 0.2)
    sg = float(np.sum(g[ok] * w[ok]) / np.sum(w[ok] ** 2)) if ok.sum() > 20 else math.nan
    glo, ghi = THRESH["gz_scale"]
    if jm > THRESH["yaw_jump"]:
        out.append(_res("e_yaw", "FAIL", {"max_jump_rad": jm}, "yaw jumps between bins (wrapped at +-pi?)",
                        "Unwrap yaw before interpolating to the bin centres (no +-pi jumps); yaw in radians."))
    elif math.isfinite(sg) and not glo <= sg <= ghi:
        u = _unit_guess(sg, cpr) if sg > 0 else None
        fix = ("gz has the wrong sign: yaw rate is positive counter-clockwise seen from above." if sg < 0 else
               (f"gz looks like {u}; send rad/s." if u else f"gz is {sg:.3g} x the AprilTag yaw rate: check units."))
        out.append(_res("e_yaw", "FAIL", {"gz_scale": sg}, "gz does not match the truth yaw rate", fix))
    else:
        out.append(_res("e_yaw", "PASS", {"max_jump_rad": jm, "gz_scale": sg}))
    # (f) residual latency
    if lat_t:
        a = np.concatenate([np.r_[x, np.full(20, np.nan)] for x in lat_w])
        bb = np.concatenate([np.r_[x, np.full(20, np.nan)] for x in lat_t])
        L = 10
        cc = []
        for lag in range(-L, L + 1):                           # truth(k + lag) vs wheels(k)
            x, y = a[max(0, -lag): len(a) - max(0, lag)], bb[max(0, lag): len(bb) - max(0, -lag)]
            ok = np.isfinite(x) & np.isfinite(y)
            cc.append(np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 50 else -1)
        cc = np.array(cc)
        i = int(np.argmax(cc))
        off = 0.0
        if 0 < i < len(cc) - 1:
            den = cc[i - 1] - 2 * cc[i] + cc[i + 1]
            off = 0.5 * (cc[i - 1] - cc[i + 1]) / den if den != 0 else 0.0
        lat = float((i - L + off) * CA.BIN)
        st = "FAIL" if abs(lat) > THRESH["lat_fail"] else ("WARN" if abs(lat) > THRESH["lat_warn"] else "PASS")
        out.append(_res("f_truth_latency", st, {"residual_latency_s": lat},
                        "" if st == "PASS" else f"the AprilTag pose lags the wheel motion by {lat * 1000:+.0f} ms",
                        "" if st == "PASS" else
                        f"Shift the AprilTag poses {abs(lat) * 1000:.0f} ms {'earlier' if lat > 0 else 'later'} than now "
                        "(your camera-latency correction is off by that much), then re-interpolate to the bin centres."))
    else:
        out.append(_res("f_truth_latency", "SKIP", None, "no healthy trial in this session"))
    # (g) healthy tracking (absolute ratio of the truth twist to the command in the holds). v4 (SPINE D17): capability is
    # analysed RELATIVE to the healthy baseline, so a controller tracking ~0.84-0.90 is expected; warn only below 0.75
    if track:
        med = float(np.median(track))
        if med < THRESH["track_warn"]:
            out.append(_res("g_healthy_tracking", "WARN", {"median_abs_ratio": med, "n_legs": len(track)},
                            f"healthy trials reach only {med:.2f} of the commanded twist",
                            "Healthy modules should reach at least ~0.75 of the probe command: check the velocity loop "
                            "(feed-forward reference vs the real no-load speed from free_speed_full_duty_<wheel>), the "
                            "battery voltage and the wheel radius; tell us the controller settings."))
        else:
            out.append(_res("g_healthy_tracking", "PASS", {"median_abs_ratio": med, "n_legs": len(track)}))
    else:
        out.append(_res("g_healthy_tracking", "SKIP", None, "no healthy trial in this session"))
    verdict = max((c["status"] for c in out), key=lambda s: RANK[s])
    return dict(session=d.name, verdict=("PASS" if verdict == "SKIP" else verdict), checks=out)


def report_text(r):
    L = [f"ARRIVAL CHECK {r['session']}: {r['verdict']}"]
    for c in r["checks"]:
        L.append(f"  [{c['status']}] {c['check']}: {c['message'] or 'ok'}"
                 + (f" (value: {json.dumps(c['value'], default=str)[:160]})" if c["value"] is not None else ""))
        if c["fix"]:
            L.append(f"         -> tell the team: {c['fix']}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--json")
    ap.add_argument("--params")
    a = ap.parse_args(argv)
    r = check_session(a.session, A.load_params(a.params) if a.params else None)
    print(report_text(r))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(r, indent=1, default=str))
    return {"PASS": 0, "WARN": 0, "FAIL": 2}[r["verdict"]]


if __name__ == "__main__":
    sys.exit(main())

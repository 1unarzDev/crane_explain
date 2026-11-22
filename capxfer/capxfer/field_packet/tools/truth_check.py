#!/usr/bin/env python3
"""AprilTag ground-truth checks (card 05 part C). READ-ONLY on the sessions.

  truth_check.py static SESSION_DIR --reference manual/apriltag_reference.csv [--tag-offset dx,dy,dyaw_deg] [--csv out.csv]
  truth_check.py motion SESSION_DIR [SESSION_DIR ...] [--tag-offset ...] [--csv out.csv]

static: for every reference pose (one FieldLog trial each, robot untouched for >= 10 s), the tag pose of the assembly
        origin: mean bias vs the tape/laser reference (mm, deg) and the standard deviation over the middle 10 s (mm, mrad).
motion: for every probe trial, the tag latency relative to the encoders (lag maximising the correlation of tag speed
        and encoder-odometry speed, 0-400 ms) and the tag/encoder speed ratio in the P1 holds (slip + scale check).
Targets (TEAM-CONFIRM after the first run): static std <= 2 mm / 4 mrad, |bias| <= 10 mm (reference-limited),
latency 0.05-0.30 s and stable within 0.03 s between trials, P1 speed ratio 0.95-1.02 on the main floor.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import packet_lib as PL  # noqa: E402
import validate_csv_session as V  # noqa: E402

from capxfer import assembly as A  # noqa: E402
from capxfer.probes import Schedule  # noqa: E402
from capxfer.field import fieldlog_config as FC  # noqa: E402
from capxfer.field.schedule import trial_schedule  # noqa: E402


def tag_pose(st, off):
    tm, d = st.get(FC.SRC["TRUTH"], (np.zeros(0, np.int64), {}))
    if not len(tm):
        return tm, np.zeros((0, 3))
    th = np.unwrap(2 * np.arctan2(d["gt_qz"], d["gt_qw"])) - off[2]
    x = d["gt_x_m"] - (np.cos(th) * off[0] - np.sin(th) * off[1])
    y = d["gt_y_m"] - (np.sin(th) * off[0] + np.cos(th) * off[1])
    o = np.argsort(tm)
    return tm[o], np.stack([x, y, th], 1)[o]


def parse_off(s):
    v = [float(x) for x in (s or "0,0,0").split(",")]
    return (v[0], v[1], np.radians(v[2]))


def static(a):
    d = Path(a.sessions[0])
    _, rows = PL.read_csv(a.reference)
    hdr = PL.header("apriltag_reference")
    refs = {r[hdr.index("trial_id")]: dict(zip(hdr, r)) for r in rows if r and r[0] == d.name}
    out = []
    rep = V.Report()
    for p in sorted(d.glob("gnd_capxprobe_*.csv")):
        if p.stem not in refs:
            continue
        st, _, _ = V.read_trial(p, rep, p.stem)
        tm, P = tag_pose(st, parse_off(a.tag_offset))
        if len(tm) < 30:
            out.append(dict(trial=p.stem, n=len(tm), note="too few tag samples"))
            continue
        mid = (tm >= tm[0] + int(2.5e9)) & (tm <= tm[0] + int(12.5e9))
        Q = P[mid] if mid.sum() > 30 else P
        r = refs[p.stem]
        bx = (Q[:, 0].mean() - float(r["x_m"])) * 1000
        by = (Q[:, 1].mean() - float(r["y_m"])) * 1000
        byaw = np.degrees(np.angle(np.exp(1j * (Q[:, 2].mean() - np.radians(float(r["yaw_deg"]))))))
        out.append(dict(trial=p.stem, pose_id=r["pose_id"], n=len(Q), std_x_mm=Q[:, 0].std() * 1000, std_y_mm=Q[:, 1].std() * 1000,
                        std_yaw_mrad=Q[:, 2].std() * 1000, bias_x_mm=bx, bias_y_mm=by, bias_yaw_deg=byaw,
                        rate_hz=len(tm) / max((tm[-1] - tm[0]) / 1e9, 1e-9)))
    return out


def motion(a):
    params = A.load_params(PL.REPO / "params" / "module_measured.yaml")
    r = float(params["wheel"]["radius_m"])
    out = []
    rep = V.Report()
    for sd in a.sessions:
        for p in sorted(Path(sd).glob("gnd_capxprobe_*.csv")):
            st, _, _ = V.read_trial(p, rep, p.stem)
            ev = V.markers(st)
            s0 = [e for e in ev if e[2] == 1]
            e1 = [e for e in ev if e[2] == 2]
            if not s0 or not e1:
                continue
            t0, num, _, code = s0[0]
            t1 = e1[-1][0]
            try:
                asm, spec, _ = V.build_assembly(V.window(st, t0 - int(0.2e9), t1), code, params)
            except Exception:
                continue
            grid = t0 + (np.arange(0, (t1 - t0) / 1e9, 0.01) * 1e9).astype(np.int64)
            om = np.zeros((len(grid), asm.n))
            for w, name in enumerate(asm.wheel_corner):
                m, c = name.split(":")
                tm, dd = st.get(FC.SRC[f"WHL_{m}"], (np.zeros(0, np.int64), {}))
                if not len(tm):
                    break
                o = np.argsort(tm)
                om[:, w] = np.interp(grid, tm[o], dd[f"wspd_{c}_{FC.mk(m)}"][o])
            J = np.stack([asm.r_hat[:, 0], asm.r_hat[:, 1], -asm.wheel_pos[:, 1] * asm.r_hat[:, 0]
                          + asm.wheel_pos[:, 0] * asm.r_hat[:, 1]], 1) / (r * (asm.w_hat * asm.r_hat).sum(-1))[:, None]
            xi_enc = np.linalg.lstsq(J, om.T, rcond=None)[0].T                   # body twist from the encoders
            v_enc = np.hypot(xi_enc[:, 0], xi_enc[:, 1])
            tm, P = tag_pose(st, parse_off(a.tag_offset))
            sel = (tm >= t0 - int(0.5e9)) & (tm <= t1 + int(0.8e9))
            tm, P = tm[sel], P[sel]
            if len(tm) < 50:
                continue
            gx = np.interp(grid, tm, P[:, 0])
            gy = np.interp(grid, tm, P[:, 1])
            k = 5
            v_tag = np.zeros(len(grid))
            v_tag[k:-k] = np.hypot(gx[2 * k:] - gx[:-2 * k], gy[2 * k:] - gy[:-2 * k]) / (2 * k * 0.01)
            best, lag = -2, 0
            for L in range(0, 41):
                a_, b_ = v_enc[: len(v_enc) - L], v_tag[L:]
                c = np.corrcoef(a_, b_)[0, 1] if a_.std() > 0 and b_.std() > 0 else -2
                if c > best:
                    best, lag = c, L
            sch = trial_schedule(asm, None, duration_s=(t1 - t0) / 1e9)    # full library or P1+P5+P2 (WS-P2)
            ratios = []
            for pid, leg, ta, tb, _ in sch.windows():
                if pid != "P1":
                    continue
                i = (grid >= t0 + int(ta * 1e9)) & (grid <= t0 + int(tb * 1e9))
                j = np.roll(i, lag)
                if v_enc[i].mean() > 0.05:
                    ratios.append(v_tag[j].mean() / v_enc[i].mean())
            out.append(dict(session=Path(sd).name, trial=p.stem, schedule_trial=num, assembly=asm.name,
                            latency_s=lag * 0.01, corr=best, p1_speed_ratio=float(np.mean(ratios)) if ratios else np.nan))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["static", "motion"])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--reference")
    ap.add_argument("--tag-offset", default="0,0,0", help="tag minus assembly origin: dx_m,dy_m,dyaw_deg (session_index.csv)")
    ap.add_argument("--csv")
    a = ap.parse_args(argv)
    if a.csv and any(PL.inside(a.csv, s) for s in a.sessions):
        sys.exit("refusing to write inside a session folder")
    if a.mode == "static" and not a.reference:
        sys.exit("static needs --reference manual/apriltag_reference.csv")
    rows = static(a) if a.mode == "static" else motion(a)
    if not rows:
        sys.exit("no usable trials")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    for r in rows:
        print("  " + "  ".join(f"{k}={r[k]:.4g}" if isinstance(r.get(k), float) else f"{k}={r.get(k, '')}" for k in keys))
    if a.mode == "motion":
        lat = np.array([r["latency_s"] for r in rows])
        print(f"latency median {np.median(lat):.3f} s, IQR {np.percentile(lat, 25):.3f}-{np.percentile(lat, 75):.3f} s "
              f"over {len(rows)} trials (params truth.apriltag_latency_s = "
              f"{A.load_params(PL.REPO / 'params' / 'module_measured.yaml')['truth']['apriltag_latency_s']})")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

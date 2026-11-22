#!/usr/bin/env python3
"""Firmware fault-hook acceptance test (FIRMWARE_HOOKS.md §3) from the logged CSVs. READ-ONLY on the session.

  check_hook_test.py SESSION_DIR --sheets HOOKTEST_SHEETS_DIR [--setup lifted|floor] [--csv results.csv]

The session is a hook-test session (make_run_sheets.py extra hooktest --module Mk): one module on a stand, wheels in the
air, trials F0 / weak 0.8 / weak 0.4 / coast / brake per wheel, then an e-stop test. Per wheel and hold window (P1+,
P1-, P2+, P2-) it compares encoder speed and duty with the firmware IK reference and with the preceding healthy trial:

  IK      F0: every wheel turns the way the IK says (sign) at |omega/omega_ref - 1| < 0.15   (wheel order, roller
          pattern, cmd_vel -> wheel IK, encoder sign)
  CPR     F0: counts advanced over the P1+ hold / (integral of wheel_state velocity) * 2 pi within 10 % of the CPR
  WEAK08  g = 0.8: speed still reached (|omega/omega_ok - 1| < 0.15); duty/duty_ok in [1.10, 1.45] (expected 1.25)
  WEAK04  g = 0.4: duty saturates (|duty| >= 0.9); omega / (0.4 omega_ok / |duty_ok|) in [0.75, 1.25]
  PASSIVE coast/brake: |omega| < 0.1 |omega_ref| (lifted); duty does not wind up (|change over the hold| < 0.10 and
          |duty| < 0.9): integrator frozen, P-term still reported
  OTHERS  the other three wheels are unaffected by a fault (|omega/omega_ok - 1| < 0.10)
  ECHO    every fault_cmd echoed on fault_state with the same wheel/mode/g/trial before motion, and 'ok' after the trial
  ESTOP   after the abort marker all wheels below 0.5 rad/s within 0.5 s
With --setup floor (no stand) only IK sign, ECHO, WEAK duty direction and PASSIVE no-windup are checked.
Exit 0 if every check passes, 2 otherwise. Coast vs brake cannot be told apart from these logs with the wheels in the
air: the manual electrical checks (hook_test_manual.csv) do that.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import packet_lib as PL  # noqa: E402
import validate_csv_session as V  # noqa: E402

from capxfer import assembly as A  # noqa: E402
from capxfer.probes import Schedule  # noqa: E402
from capxfer.field import fieldlog_config as FC  # noqa: E402

WINDOWS = ("P1+", "P1-", "P2+", "P2-")


def omega_ref(asm, twist, params):
    """Firmware IK at nominal geometry (as capxfer.sim): omega_ref per wheel for an assembly twist."""
    r = float(params["wheel"]["radius_m"])
    pos, rh, wh = asm.wheel_pos, asm.r_hat, asm.w_hat
    vx = twist[0] - twist[2] * pos[:, 1]
    vy = twist[1] + twist[2] * pos[:, 0]
    return (vx * rh[:, 0] + vy * rh[:, 1]) / (r * (wh * rh).sum(-1))


def per_window(streams, t0, asm, spec, params, cpr, trial=None):
    from capxfer.field.schedule import trial_schedule
    sch = trial_schedule(asm, trial)                 # WS-P2: hook-test trials run P1+P5+P2 (only P1/P2 are checked)
    out = {}
    for pid, leg, ta, tb, tw in sch.windows():
        key = f"{pid}{'+' if leg > 0 else '-'}"
        if key not in WINDOWS:
            continue
        ref = omega_ref(asm, tw, params)
        a, b = t0 + int(ta * 1e9), t0 + int(tb * 1e9)
        rec = {}
        for w, name in enumerate(asm.wheel_corner):
            m, c = name.split(":")
            s = FC.SRC[f"WHL_{m}"]
            tm, d = streams.get(s, (np.zeros(0, np.int64), {}))
            sel = (tm >= a) & (tm <= b)
            om = d.get(f"wspd_{c}_{FC.mk(m)}", np.zeros(0))[sel]
            du = d.get(f"duty_{c}_{FC.mk(m)}", np.zeros(0))[sel]
            ct = d.get(f"cnt_{c}_{FC.mk(m)}", np.zeros(0))[sel]
            tt = tm[sel] / 1e9
            n = len(om)
            q = max(1, n // 5)
            integ = float(np.trapezoid(om, tt)) if n > 1 else np.nan
            rec[name] = dict(ref=float(ref[w]), om=float(np.nanmedian(om)) if n else np.nan,
                             duty=float(np.nanmedian(du)) if n else np.nan,
                             dslope=float(np.nanmean(du[-q:]) - np.nanmean(du[:q])) if n > 2 else np.nan,
                             cpr=float((ct[-1] - ct[0]) / integ * 2 * np.pi) if n > 1 and abs(integ) > 1e-3 else np.nan)
        out[key] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--sheets", help="hooktest run-sheet folder (schedule.json inside)")
    ap.add_argument("--schedule")
    ap.add_argument("--setup", choices=["lifted", "floor"], default="lifted")
    ap.add_argument("--csv")
    a = ap.parse_args(argv)
    d = Path(a.session)
    if a.csv and PL.inside(a.csv, d):
        sys.exit("refusing to write inside the session folder")
    sp = Path(a.schedule) if a.schedule else Path(a.sheets) / "schedule.json"
    sched = json.loads(sp.read_text())
    sidx = V.sched_index(sched)
    params = A.load_params(PL.REPO / "params" / "module_measured.yaml")
    cpr_nom = float(params["motor"]["counts_per_output_rev"])
    rows, fails = [], 0
    ok_ref = {}

    def res(trial, wheel, mode, check, window, value, passed, note=""):
        nonlocal fails
        fails += 0 if passed else 1
        rows.append(dict(trial=trial, wheel=wheel, mode=mode, check=check, window=window,
                         value="" if value is None else (f"{value:.4g}" if isinstance(value, float) else value),
                         result="PASS" if passed else "FAIL", note=note))

    rep = V.Report()
    for p in sorted(d.glob("gnd_capxprobe_*.csv")):
        st, _, _ = V.read_trial(p, rep, p.stem)
        ev = V.markers(st)
        starts = [e for e in ev if e[2] == 1]
        if not starts:
            continue
        t0, num, _, code = starts[0]
        endv = [e for e in ev if e[2] in (2, 3)]
        t1 = endv[-1][0] if endv else t0 + int(30e9)
        asm, spec, _ = V.build_assembly(V.window(st, t0 - int(0.2e9), t1), code, params)
        t = sidx.get((asm.name, num))
        if t is None:
            res(num, "-", "-", "SCHEDULE", "-", None, False, "trial not in the hook-test schedule")
            continue
        faults = [f for f in t["faults"] if ":" in f["wheel"]]
        mode = faults[0]["mode"] if faults else "ok"
        g = faults[0].get("g") if faults else None
        target = faults[0]["wheel"] if faults else None
        if t.get("condition") == "ESTOP_TEST":
            ab = [e for e in ev if e[2] == 3]
            if not ab:
                res(num, "all", "estop", "ESTOP", "-", None, False, "no abort marker: press Ctrl-C in the runner during P2")
                continue
            ta = ab[0][0]
            mx = 0.0
            for m, *_ in spec:
                tm, dd = st.get(FC.SRC[f"WHL_{m}"], (np.zeros(0, np.int64), {}))
                sel = tm > ta + int(0.5e9)
                for c in PL.CORNERS:
                    v = dd.get(f"wspd_{c}_{FC.mk(m)}", np.zeros(0))[sel]
                    if len(v):
                        mx = max(mx, float(np.nanmax(np.abs(v))))
            res(num, "all", "estop", "ESTOP", "after+0.5s", mx, mx < 0.5, "max |omega| rad/s")
            continue
        # ECHO
        ctl = st.get(FC.SRC["CTL"], (np.zeros(0, np.int64), {}))
        mv = [ctl[0][k] for k in range(len(ctl[0])) if abs(np.nan_to_num(ctl[1]["cmd_vx"][k])) + abs(np.nan_to_num(ctl[1]["cmd_vy"][k])) > 1e-6]
        tmo = int(min(mv)) if mv else t1
        ech = V.echoes(st, t0 - int(0.2e9), tmo)
        after = V.echoes(st, tmo, t1 + int(0.5e9))
        for f in faults:
            m, c = f["wheel"].split(":")
            hit = [e for e in ech if e[1] == m and np.isfinite(e[2]) and int(e[2]) == PL.CORNERS.index(c)
                   and int(e[3]) == V.MODE_CODE[f["mode"]] and int(e[5]) == num
                   and (f["mode"] != "weak" or abs(e[4] - f["g"]) <= 1e-3)]
            back = [e for e in after if e[1] == m and np.isfinite(e[2]) and int(e[2]) == PL.CORNERS.index(c) and int(e[3]) == 0]
            res(num, f["wheel"], f["mode"], "ECHO", "-", len(hit), bool(hit) and bool(back),
                "fault echo before motion and ok echo after" if hit and back else f"echo {len(hit)}, ok-echo {len(back)}")
        win = per_window(st, t0, asm, spec, params, cpr_nom, t)
        if mode == "ok":
            for wname in asm.wheel_corner:
                for k in WINDOWS:
                    r = win[k][wname]
                    ratio = r["om"] / r["ref"] if abs(r["ref"]) > 1e-6 else np.nan
                    good = np.sign(r["om"]) == np.sign(r["ref"]) and (a.setup == "floor" or abs(ratio - 1) < 0.15)
                    res(num, wname, "ok", "IK", k, float(ratio), bool(good), f"omega {r['om']:.2f} vs ref {r['ref']:.2f} rad/s")
                    ok_ref[(wname, k)] = r
                if a.setup == "lifted":
                    cp = win["P1+"][wname]["cpr"]
                    res(num, wname, "ok", "CPR", "P1+", float(cp), bool(np.isfinite(cp) and abs(cp / cpr_nom - 1) < 0.10),
                        f"params CPR {cpr_nom:.0f}")
            continue
        for wname in asm.wheel_corner:
            for k in WINDOWS:
                r, o = win[k][wname], ok_ref.get((wname, k))
                if o is None:
                    res(num, wname, mode, "REF", k, None, False, "no preceding healthy trial")
                    continue
                if wname != target:
                    if a.setup == "lifted":
                        q = r["om"] / o["om"] if abs(o["om"]) > 1e-6 else np.nan
                        res(num, wname, mode, "OTHERS", k, float(q), bool(abs(q - 1) < 0.10))
                    continue
                if mode == "weak" and g is not None and g >= 0.6:
                    q = r["duty"] / o["duty"] if abs(o["duty"]) > 1e-6 else np.nan
                    if a.setup == "lifted":
                        so = r["om"] / o["om"] if abs(o["om"]) > 1e-6 else np.nan
                        res(num, wname, "weak", "WEAK08", k, float(so), bool(abs(so - 1) < 0.15), "omega / omega_ok")
                        res(num, wname, "weak", "WEAK08", k, float(q), bool(1.10 <= q <= 1.45), "duty / duty_ok (expect 1/g)")
                    else:
                        res(num, wname, "weak", "WEAK08", k, float(q), bool(q > 1.0), "duty / duty_ok > 1")
                elif mode == "weak":
                    if a.setup == "lifted":
                        res(num, wname, "weak", "WEAK04", k, abs(r["duty"]), bool(abs(r["duty"]) >= 0.9), "|duty| saturated")
                        exp = g * o["om"] / max(abs(o["duty"]), 1e-3)
                        q = r["om"] / exp if abs(exp) > 1e-6 else np.nan
                        res(num, wname, "weak", "WEAK04", k, float(q), bool(0.75 <= q <= 1.25), "omega / (g omega_ok/|duty_ok|)")
                    else:
                        q = abs(r["duty"]) / max(abs(o["duty"]), 1e-3)
                        res(num, wname, "weak", "WEAK04", k, float(q), bool(q > 1.0), "|duty| / |duty_ok| > 1")
                else:
                    if a.setup == "lifted":
                        q = abs(r["om"]) / max(abs(r["ref"]), 1e-6)
                        res(num, wname, mode, "PASSIVE", k, float(q), bool(q < 0.10), "|omega| / |omega_ref|")
                    res(num, wname, mode, "PASSIVE", k, float(abs(r["dslope"])),
                        bool(abs(r["dslope"]) < 0.10 and abs(r["duty"]) < 0.9), f"duty change over the hold (duty {r['duty']:.2f})")
    if not rows:
        sys.exit("no hook-test trials found")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault(r["check"], [0, 0])[0 if r["result"] == "PASS" else 1] += 1
    print(f"hook test {d.name} ({a.setup}): " + ", ".join(f"{k} {v[0]} pass / {v[1]} fail" for k, v in sorted(by.items())))
    for r in rows:
        if r["result"] == "FAIL":
            print(f"  FAIL trial {r['trial']} {r['wheel']} {r['mode']} {r['check']} {r['window']}: {r['value']} {r['note']}")
    print("VERDICT:", "PASS" if fails == 0 else "FAIL")
    return 0 if fails == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

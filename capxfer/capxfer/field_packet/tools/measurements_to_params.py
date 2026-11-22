#!/usr/bin/env python3
"""Measurements CSV -> proposed params values (for the integrator). Never edits params/.

  measurements_to_params.py manual/measurements.csv [--out derived/params_from_measurements.yaml]

Converts the MEASURED rows of MEASUREMENTS_SHEET.csv into the units of params/module_measured.yaml (roll distance ->
radius, tilt angle -> mu, counts for 10 turns -> CPR, spring-scale force -> torque, per-module averages for shared
geometry, roller signs -> '/' or '\\') and prints a table: key, current params value, proposed value, change. The YAML
is a PROPOSAL; the integrator edits params/ (CONTRACTS rule 1). Rows that are TODO or NOT_MEASURED are listed as gaps.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import packet_lib as PL  # noqa: E402


def get(d, key):
    for k in key.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("measurements")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.out and PL.inside(a.out, PL.REPO / "params"):
        sys.exit("refusing to write into params/ (the integrator edits params)")
    params = yaml.safe_load((PL.REPO / "params" / "module_measured.yaml").read_text())
    hdr, rows = PL.read_csv(a.measurements)
    R = {r[0]: dict(zip(hdr, r)) for r in rows if r}
    val = {k: float(r["value"]) for k, r in R.items() if r["status"] == "MEASURED" and r["value"].strip()}
    gaps = sorted(k for k, r in R.items() if r["status"] in ("TODO", "NOT_MEASURED"))
    prop, prov = {}, {}

    def put(key, v, src):
        prop[key], prov[key] = v, src
    mods = PL.MODULES
    for m in mods:
        if f"MASS_{m}" in val:
            put(f"module.mass_kg.{m}", round(val[f"MASS_{m}"], 3), f"MASS_{m}")
    for flag, key in (("HASLIDAR", "module.lidar_modules"), ("HASLIFTER", "module.lifter_modules")):
        got = [m for m in mods if f"{flag}_{m}" in val]
        if len(got) == 4:
            put(key, [m for m in mods if val[f"{flag}_{m}"] >= 0.5], f"{flag}_M1..M4")

    def mean(prefix, f=lambda x: x):
        v = [f(val[f"{prefix}_{m}"]) for m in mods if f"{prefix}_{m}" in val]
        return (sum(v) / len(v), max(v) - min(v), len(v)) if v else None
    for prefix, key, f in (("SIDEX", "module.side_x_m", lambda x: x), ("SIDEY", "module.side_y_m", lambda x: x),
                           ("WHEELBASE", "module.half_wheelbase_m", lambda x: x / 2), ("TRACK", "module.half_track_m", lambda x: x / 2),
                           ("ROLL5", "wheel.radius_m", lambda d: d / (10 * math.pi)), ("CGH", "module.cg_height_m", lambda x: x),
                           ("CPR10", "motor.counts_per_output_rev", lambda c: c / 10), ("NROLLERS", "wheel.rollers", lambda x: x)):
        r = mean(prefix, f)
        if r:
            put(key, round(r[0], 5), f"mean of {r[2]} module(s), spread {r[1]:.4g}")
    pats = {}
    for c in PL.CORNERS:
        s = {val[f"ROLLER_{c.upper()}_{m}"] for m in mods if f"ROLLER_{c.upper()}_{m}" in val}
        if len(s) == 1:
            pats[c] = "/" if s.pop() > 0 else "\\"
        elif len(s) > 1:
            pats[c] = "MODULES DISAGREE"
    if pats:
        put("wheel.roller_pattern", pats, "ROLLER_* (all modules must agree)")
    r = float(prop.get("wheel.radius_m", params["wheel"]["radius_m"]))
    bd = mean("BACKDRIVE", lambda F: F * r)
    if bd:
        put("motor.backdrive_torque_nm", [round(bd[0] * 0.7, 3), round(bd[0] * 1.3, 3)], "BACKDRIVE * radius, +-30 % range")
    for iid, key, f in (("FW_VELHZ", "firmware.vel_loop_hz", float), ("FW_KP", "firmware.kp_duty_per_radps", float),
                        ("FW_KI", "firmware.ki_duty_per_rad", float), ("FW_ENCHZ", "firmware.encoder_rate_hz", float),
                        ("FW_CMDLAT", "firmware.actuation_latency_s", float), ("CS_NOISE", "sensors.current_noise_a", float),
                        ("TAG_RATE", "truth.apriltag_rate_hz", float), ("TAG_LAT", "truth.apriltag_latency_s", float),
                        ("TAG_NOISE_POS", "truth.apriltag_pos_noise_m", lambda x: x / 1000),
                        ("TAG_NOISE_YAW", "truth.apriltag_yaw_noise_rad", lambda x: x / 1000)):
        if iid in val:
            put(key, round(f(val[iid]), 6), iid)
    if "FLOOR_A_MU" in val:
        mu = math.tan(math.radians(val["FLOOR_A_MU"]))
        put("floor.mu", [round(mu * 0.9, 3), round(mu * 1.1, 3)], "tan(FLOOR_A_MU) +-10 % (sys-ID narrows it)")
    sh = [math.tan(math.radians(val[k])) for k in ("SHEET1_MU", "SHEET2_MU") if k in val]
    if sh:
        put("floor.low_mu_sheet", [round(min(sh), 3), round(max(sh), 3)], "tan(SHEET*_MU)")
        if max(sh) > 0.40:
            print("WARNING: a sheet has mu > 0.40: not a useful F7 sheet (A.md Task 4)")
    if "PAYLOAD_6" in val and val["PAYLOAD_6"] > 6.0:
        print("WARNING: the 6 kg payload set weighs more than 6.00 kg: remove mass (prereg safety limit)")
    print(f"{'key':38s} {'params now':>22s} {'proposed':>22s}  source")
    for k in sorted(prop):
        print(f"{k:38s} {str(get(params, k)):>22s} {str(prop[k]):>22s}  {prov[k]}")
    print(f"\n{len(prop)} proposed value(s); {len(gaps)} item(s) not measured: {gaps[:12]}{' ...' if len(gaps) > 12 else ''}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("# PROPOSAL from measurements (capxfer/field_packet/tools/measurements_to_params.py); "
                               "the integrator decides what enters params/\n"
                               + yaml.safe_dump({"proposed": prop, "provenance": prov, "not_measured": gaps}, sort_keys=True))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

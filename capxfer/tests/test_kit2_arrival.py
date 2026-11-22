"""WS-KIT2 arrival check: a clean SIMULATION binned session passes; each injected processing error is detected by the
right check with a plain-words fix; process_session writes the caches outside the session folder."""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from capxfer import assembly as A
from capxfer.field import arrival_check as AC
from capxfer.field import csv_adapter as CA

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    trials, binned = [], {}
    for spec, B in (([("M1", 0, 0, 0)], 3), (list(A.PHYSICAL_TYPES["pair"]), 2)):
        n = 4 * len(spec)
        _, bn, _ = CA.simulate_trials(spec, np.zeros((B, n), np.int64), np.ones((B, n), np.float32), seed=len(trials) + 1)
        for b in range(B):
            tid = f"T{len(trials):03d}"
            trials.append(dict(trial_id=tid, seq=0, block="SYSID", layout=CA.layout_str(spec), fault="ok",
                               imu_module=spec[0][0], truth_tag=spec[0][0], aborted=0))
            binned[tid] = bn[b]
    return tmp_path_factory.mktemp("arr"), trials, binned


def _run(clean, name, f=None):
    d, trials, binned = clean
    bb = copy.deepcopy(binned)
    for k in bb:
        if f:
            f(bb[k])
    r = AC.check_session(CA.write_binned_session(d, name, trials, bb))
    return r, {c["check"]: c for c in r["checks"]}


def _speeds(b, fn):
    for k in list(b):
        if k.endswith("_speed"):
            b[k] = fn(b[k])


def test_clean_passes(clean):
    r, c = _run(clean, "SIMULATION_clean")
    assert r["verdict"] == "PASS", AC.report_text(r)
    assert 0.9 < c["c_wheel_ik"]["value"]["scale_median"] < 1.1
    assert c["g_healthy_tracking"]["status"] == "PASS"
    assert abs(c["f_truth_latency"]["value"]["residual_latency_s"]) < 0.02


def test_slow_controller_warns_only(clean):
    """v4 (D17): a healthy robot tracking 0.70 of command (wheels, IMU and truth all consistent) only WARNs (g)."""
    def slow(b):
        for k in list(b):
            if k.endswith("_speed") or k in ("gz", "ax", "ay"):
                b[k] = b[k] * 0.70
        for k in ("x", "y", "yaw"):
            b[k] = b[k][0] + 0.70 * (b[k] - b[k][0])
    r, c = _run(clean, "SIMULATION_slow", slow)
    assert c["g_healthy_tracking"]["status"] == "WARN" and r["verdict"] == "WARN", AC.report_text(r)
    r, c = _run(clean, "SIMULATION_tracks085", lambda b: [b.__setitem__(k, b[k] * 0.85) for k in list(b)
                                                          if k.endswith("_speed") or k in ("gz", "ax", "ay")]
                or [b.__setitem__(k, b[k][0] + 0.85 * (b[k] - b[k][0])) for k in ("x", "y", "yaw")])
    assert c["g_healthy_tracking"]["status"] == "PASS"            # 0.85 is the expected D17 tracking, no warning


def _shift(b, n=4):
    for k in b:
        if k != "t":
            b[k] = np.r_[b[k][n:], np.full(n, np.nan)]


def _swap(b):
    b["M1_fl_speed"], b["M1_fr_speed"] = b["M1_fr_speed"].copy(), b["M1_fl_speed"].copy()


def _wrap(b):
    b["yaw"] = b["yaw"].copy()
    b["yaw"][300:] += 2 * np.pi                                          # a +-pi wrap artefact, heading unchanged


def _lat(b):
    for k in ("x", "y", "yaw"):
        b[k] = np.r_[np.full(2, np.nan), b[k][:-2]]


def _bins(b):
    b["t"] = b["t"] * 2                                                  # 100 ms bins


CASES = {
    "t0_offset": (_shift, "b_timeline", "t = 0"),
    "wheel_order": (_swap, "c_wheel_ik", "corner names"),
    "wheel_sign": (lambda b: b.__setitem__("M1_rl_speed", -b["M1_rl_speed"]), "c_wheel_ik", "Flip the sign of M1_rl"),
    "units_rpm": (lambda b: _speeds(b, lambda v: v * 60 / (2 * np.pi)), "c_wheel_ik", "rpm"),
    "counts_per_rev": (lambda b: _speeds(b, lambda v: v * 0.5), "c_wheel_ik", "counts_per_rev"),
    "duty_sign": (lambda b: b.__setitem__("M1_fr_duty", -b["M1_fr_duty"]), "d_duty_speed", "duty of M1_fr"),
    "duty_constant": (lambda b: b.__setitem__("M1_rl_duty", np.full_like(b["M1_rl_duty"], 0.2)), "d_duty_speed",
                      "does not follow"),
    "yaw_wrapped": (_wrap, "e_yaw", "Unwrap yaw"),
    "gz_sign": (lambda b: b.__setitem__("gz", -b["gz"]), "e_yaw", "wrong sign"),
    "truth_latency": (_lat, "f_truth_latency", "earlier"),
    "bin_spacing": (_bins, "a_schema", "Bin k must cover"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_injected_error_detected(clean, case):
    f, check, words = CASES[case]
    r, c = _run(clean, f"SIMULATION_{case}", f)
    assert r["verdict"] == "FAIL" and c[check]["status"] == "FAIL", AC.report_text(r)
    assert words in c[check]["fix"], c[check]["fix"]
    if case == "t0_offset":
        assert abs(c[check]["value"]["t0_offset_s"] + 0.2) < 1e-9
    if case == "truth_latency":
        assert 0.08 < c[check]["value"]["residual_latency_s"] < 0.12
    if case == "counts_per_rev":
        assert 0.45 < c[check]["value"]["scale"] < 0.55


def test_process_session_one_command(clean, tmp_path):
    d, trials, binned = clean
    s = CA.write_binned_session(d, "SIMULATION_proc", trials, binned)
    r = subprocess.run([str(REPO / "capxfer" / "field" / "process_session.sh"), str(s), str(tmp_path / "derived")],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:] + r.stdout[-2000:]
    o = tmp_path / "derived" / "SIMULATION_proc"
    for f in ("arrival_check.txt", "validate.json", "qc.csv", "targets.csv", "raw_0.pt", "feat_0.pt", "SHA256SUMS"):
        assert (o / f).is_file(), f
    assert not any(p.name.startswith(("raw_", "feat_")) for p in s.iterdir())     # nothing written into the session
    bad = CA.write_binned_session(d, "SIMULATION_procbad", trials, {k: dict(v, gz=-v["gz"]) for k, v in binned.items()})
    r = subprocess.run([str(REPO / "capxfer" / "field" / "process_session.sh"), str(bad), str(tmp_path / "derived")],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 2 and "tell the team" in r.stdout

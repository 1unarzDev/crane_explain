"""Pre-processed 20 Hz binned CSV sessions (`capxfer/field_kit_v2/CSV_FORMAT.md`) -> RawTrial (CONTRACTS C1) (WS-KIT2).

The robot team runs its own robot / logging / binning code and returns, per session folder: trials.csv, ONE binned.csv
(50 ms bins from t = 0 = the first command of the probe routine), docks.csv (DEMO) and an optional notes.csv. This
module is the only reader of that format. It never writes inside a session folder.

  python -m capxfer.field.csv_adapter validate SESSION_DIR [--seed-file SEALED_SEED]

Reconstruction (exact for every feature the frozen pipeline computes; `capxfer.featurize` pools 100 Hz samples
(5 per 50 ms bin) and takes encoder count differences over 50 ms):
* grid t = 0, 0.01, ..., < probes.Schedule.T (100 Hz). Bin k = [0.05 k, 0.05 (k + 1)).
* duty / gz / ax / ay: grid sample i >= 1 carries bin floor((i - 1) / 5), so every featurize bin mean is the team's
  bin mean. Blank cells stay NaN (featurize zero-fills them, as for a sensor dropout).
* counts: cumulative counts C_k at the bin edges from the binned speed (C_{k+1} = C_k + speed_k * 0.05 * cpr / 2 pi),
  held on the grid, so featurize's 50 ms count difference returns the team's speed exactly. A blank speed adds 0.
* cmd: the fixed probe routine (probes.Schedule.twist on the grid), exactly as in simulation; the logged cmd_* columns
  are only checked against it (meta["qc"], and capxfer.field.arrival_check).
* truth: the bin-centre poses (already latency-corrected by the team) linearly interpolated onto the grid
  (truth_latency_s = 0), converted from the `truth_tag` module's pose to the assembly origin (mean of the module
  centres; `tag_offsets` {module: (dx, dy, dyaw)} if a tag is not at its module centre). Evaluator-only.
* lidar odometry and motor current do not exist in this format (NaN): rungs E2 / E3 are not available; the frozen
  bundle uses E1.
* labels: the declared `fault` joined with the sealed schedule (realv2_design.expand_schedule(seed)) by (block, seq)
  -> schedule_match (PROTOCOL_V2 §7 (b)); episode-link labels for builders; meta["csv_checks"] (max blank run in the
  P1/P5/P2 holds -> gap s, truth coverage per leg) and meta["trials_rows"] for builders.
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

from .. import assembly as A
from ..probes import PIDS, Schedule
from . import loader as L

FORMAT_DOC = "capxfer/field_kit_v2/CSV_FORMAT.md"
REPO = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO / "params" / "module_measured.yaml"
MODULES = ("M1", "M2", "M3", "M4")
CORNERS = ("fl", "fr", "rl", "rr")
MODE_CODE = {"ok": 0, "weak": 1, "coast": 2, "brake": 3}
BLOCKS = ("SYSID", "MASK", "DEMO_PROBE", "DEMO_SOLO", "SIMVAL", "HOOK")
DEMO_BLOCKS = ("DEMO_PROBE", "DEMO_SOLO")
SEALED_BLOCKS = ("MASK", "SIMVAL") + DEMO_BLOCKS
BIN = 0.05
GRID_HZ = 100.0
PER = 5                                                    # grid samples per bin
ROBOT_PIDS = ("P1", "P5", "P2")
TRIALS_REQ = ("trial_id", "seq", "block", "layout", "fault", "imu_module", "truth_tag", "aborted")
BINNED_REQ = ("trial_id", "t", "cmd_vx", "cmd_vy", "cmd_wz", "gz", "x", "y", "yaw")
BINNED_OPT = ("ax", "ay")
DOCKS_REQ = ("episode", "module", "dock_order", "success", "wall_contact", "aborted")


class SessionRefused(PermissionError):
    pass


# ---------------------------------------------------------------------------------------------------- helpers
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_rows(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]


def _header(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return [c.strip() for c in next(csv.reader(fh), [])]


def parse_layout(s):
    """'M2@0,0,0;M3@1,0,0' -> [(module, i, j, yaw_k)] (first listed first)."""
    out = []
    for part in [x for x in str(s).replace(" ", "").split(";") if x]:
        m, pos = part.split("@")
        i, j, k = (int(v) for v in pos.split(","))
        if m not in MODULES:
            raise ValueError(f"unknown module {m}")
        out.append((m, i, j, k % 4))
    if not out:
        raise ValueError("empty layout")
    return out


def layout_str(spec):
    return ";".join(f"{m}@{i},{j},{k}" for m, i, j, k in spec)


def asm_name(spec):
    mods = sorted(m for m, *_ in spec)
    for name, lay in A.PHYSICAL_TYPES.items():
        if name != "single" and sorted(m for m, *_ in lay) == mods:
            return name
    return "single" if len(spec) == 1 else "custom"


def parse_fault(s):
    """'M3:fl:coast;M1:rr:weak:0.8' | 'ok' -> {(module, wheel): (mode, g)}."""
    out = {}
    for part in [x for x in str(s or "ok").replace(" ", "").split(";") if x and x.lower() != "ok"]:
        f = part.split(":")
        mode = f[2].lower()
        g = float(f[3]) if (mode == "weak" and len(f) > 3) else 1.0
        out[(f[0], f[1])] = (mode, g)
    return out


def fault_arrays(faults, asm):
    mode = np.zeros(asm.n, np.int64)
    g = np.ones(asm.n, np.float32)
    bad = []
    for (m, c), (md, gv) in faults.items():
        name = f"{m}:{c}"
        if name not in asm.wheel_corner or md not in MODE_CODE:
            bad.append(f"fault {m}:{c}:{md} not on this assembly / unknown mode")
            continue
        w = asm.wheel_corner.index(name)
        mode[w], g[w] = MODE_CODE[md], (gv if md == "weak" else 1.0)
    return mode, g, bad


def wheel_cols(spec):
    return [f"{m}_{c}_{q}" for m, *_ in spec for c in CORNERS for q in ("speed", "duty")]


def is_simulation(d):
    """A session is SIMULATION iff its folder name, or any notes.csv / trials.csv text, says SIMULATION."""
    d = Path(d)
    if "SIMULATION" in d.name.upper():
        return True
    for n in ("notes.csv",):
        p = d / n
        if p.is_file() and "SIMULATION" in p.read_text(errors="replace").upper():
            return True
    return False


def read_binned(d):
    """binned.csv -> {trial_id: {column: float array}} (rows in file order; blank -> NaN)."""
    import pandas as pd
    df = pd.read_csv(Path(d) / "binned.csv", dtype={"trial_id": str}, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    out = {}
    for tid, g in df.groupby("trial_id", sort=False):
        out[str(tid).strip()] = {c: g[c].to_numpy(float) for c in g.columns if c != "trial_id"}
    return out


# ---------------------------------------------------------------------------------------------------- sealed schedule
def schedule_from_seed(seed_file_or_bytes):
    from ..analysis import realv2_design as D
    s = seed_file_or_bytes
    seed = bytes(s) if isinstance(s, (bytes, bytearray)) else bytes.fromhex(Path(s).read_text().strip())
    return D.expand_schedule(seed)


def sealed_entry(sched, block, seq):
    """(entry, expected layout spec (None for DEMO_SOLO: from the trial), expected faults, link labels)."""
    from ..analysis import realv2_design as D
    if block == "MASK":
        t = sched["mask_trials"][int(seq)]
        b = t["build"]
        spec = [(b.split("_", 1)[1], 0, 0, 0)] if b.startswith("single_") else list(A.PHYSICAL_TYPES[b])
        f = {(t["module"], t["wheel"]): (t["firmware"], 1.0)} if t["module"] else {}
        link = dict(episode=f"{t['module']}:{t['wheel']}" if t["module"] else f"H_{b}",
                    phase="split" if b.startswith("single_") else "assembly", round=b, rep=t.get("rep", ""),
                    choice_block="", sheet_r3_split=t.get("r3_split") or "NONE", sheet_realv2_seq=t["seq"])
        return t, spec, f, link
    if block == "SIMVAL":
        t = sched["simval_day2"][int(seq)]
        f = {("M1", t["wheel"]): (t["firmware"], 1.0)} if t["wheel"] else {}
        link = dict(episode=f"M1:{t['wheel']}" if t["wheel"] else "H_single_M1", phase="split", round="single_M1",
                    rep="", choice_block="", sheet_r3_split="NONE", sheet_realv2_seq="")
        return t, [("M1", 0, 0, 0)], f, link
    e = sched["demo"][int(seq)]
    fw = {(m, w): ("coast", 1.0) for m, w in e["faulted_wheels"].items()}
    link = dict(episode=f"D{int(e['episode']) + 1:02d}", phase="assembly" if block == "DEMO_PROBE" else "split",
                round="", rep=1, choice_block=int(e["episode"]) + 1, sheet_r3_split="NONE", sheet_realv2_seq="")
    return e, (list(A.PHYSICAL_TYPES[D.DEMO_ASSEMBLY]) if block == "DEMO_PROBE" else None), fw, link


# ---------------------------------------------------------------------------------------------------- verification
def verify_binned_session(d):
    """Structural verdict of one returned session folder (read-only): FIELD_OK | SIMULATION | INVALID."""
    d = Path(d)
    err, info = [], []
    tp, bp = d / "trials.csv", d / "binned.csv"
    for p in (tp, bp):
        if not p.is_file():
            err.append(f"missing {p.name}")
    if err:
        return dict(verdict="INVALID", errors=err, info=info)
    h = _header(tp)
    miss = [c for c in TRIALS_REQ if c not in h]
    if miss:
        err.append(f"trials.csv: missing columns {miss}")
    hb = _header(bp)
    miss = [c for c in BINNED_REQ if c not in hb]
    if miss:
        err.append(f"binned.csv: missing columns {miss}")
    if err:
        return dict(verdict="INVALID", errors=err, info=info)
    trials = _read_rows(tp)
    ids = [t["trial_id"] for t in trials]
    if len(set(ids)) != len(ids) or any(not i or ":" in i for i in ids):
        err.append("trials.csv: trial_id must be unique, non-empty and contain no ':'")
    B = read_binned(d)
    for t in trials:
        try:
            spec = parse_layout(t["layout"])
            parse_fault(t["fault"])
        except (ValueError, IndexError):
            err.append(f"trial {t['trial_id']}: unreadable layout {t['layout']!r} or fault {t['fault']!r}")
            continue
        if str(t["aborted"]) not in ("0", ""):
            continue
        b = B.get(t["trial_id"])
        if b is None:
            err.append(f"trial {t['trial_id']}: no rows in binned.csv")
            continue
        missw = [c for c in wheel_cols(spec) if c not in b]
        if missw:
            err.append(f"trial {t['trial_id']}: binned.csv lacks wheel columns {missw[:4]}{'...' if len(missw) > 4 else ''}")
    if (d / "SHA256SUMS").is_file():
        for line in (d / "SHA256SUMS").read_text().splitlines():
            if line.strip():
                hsh, n = line.split(None, 1)
                n = n.strip().lstrip("*")
                if not (d / n).is_file() or sha256_file(d / n) != hsh:
                    err.append(f"SHA256SUMS: {n} missing or changed")
    if err:
        return dict(verdict="INVALID", errors=err, info=info)
    return dict(verdict="SIMULATION" if is_simulation(d) else "FIELD_OK", errors=[], info=info)


# ---------------------------------------------------------------------------------------------------- reconstruction
def bins_to_grid(T_grid, v):
    """Bin values v (K,) -> grid (T_grid,) with grid sample i >= 1 carrying bin floor((i - 1) / 5) (featurize pools
    samples 5k+1 .. 5k+5 for bin k), sample 0 carrying bin 0; beyond the last bin: NaN."""
    i = np.arange(T_grid)
    k = np.maximum(i - 1, 0) // PER
    out = np.full(T_grid, np.nan)
    ok = k < len(v)
    out[ok] = v[k[ok]]
    return out


def speed_to_counts(T_grid, speed, cpr):
    """Bin speeds (K,) rad/s -> cumulative counts on the grid (edge k at grid 5k, held), so that
    counts[5(k+1)] - counts[5k] = speed_k * 0.05 * cpr / 2 pi. Blank -> 0 increment."""
    inc = np.nan_to_num(np.asarray(speed, float), nan=0.0) * BIN * cpr / (2 * math.pi)
    C = np.concatenate([[0.0], np.cumsum(inc)])
    i = np.arange(T_grid)
    return np.round(C[np.clip(i // PER, 0, len(C) - 1)])


def truth_to_grid(grid, tb, x, y, yaw, max_gap=0.1):
    """Bin-centre poses -> grid by linear interpolation; NaN where the nearest valid centres are > max_gap apart."""
    tc = tb + BIN / 2
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(yaw)
    out = np.full((len(grid), 3), np.nan)
    if ok.sum() < 2:
        return out
    tcv = tc[ok]
    for j, v in enumerate((x[ok], y[ok], np.unwrap(yaw[ok]))):
        out[:, j] = np.interp(grid, tcv, v)
    k = np.searchsorted(tcv, grid)
    lo = tcv[np.clip(k - 1, 0, len(tcv) - 1)]
    hi = tcv[np.clip(k, 0, len(tcv) - 1)]
    bad = ((hi - lo) > max_gap + 1e-9) | (grid < tcv[0] - BIN) | (grid > tcv[-1] + BIN)
    out[bad] = np.nan
    return out


def tag_to_origin(pose_tag, asm, tag_module, tag_offset=(0.0, 0.0, 0.0)):
    """Tag pose (T,3) of `tag_module` (tag at offset in the module frame) -> assembly-origin pose (T,3)."""
    X, Y, th = pose_tag[:, 0], pose_tag[:, 1], pose_tag[:, 2]
    odx, ody, odth = tag_offset
    thm = th - odth
    Xm = X - (np.cos(thm) * odx - np.sin(thm) * ody)
    Ym = Y - (np.sin(thm) * odx + np.cos(thm) * ody)
    mods = [m for m, *_ in asm.modules]
    k = mods.index(tag_module)
    tha = thm - asm.modules[k][3] * math.pi / 2
    px, py = asm.module_pos[k]
    return np.stack([Xm - (np.cos(tha) * px - np.sin(tha) * py), Ym - (np.sin(tha) * px + np.cos(tha) * py), tha], 1)


def _max_blank_gap(vals, k0, k1):
    """Gap (s) of the longest run of blank bins in [k0, k1] over the given columns: (run + 1) * 0.05 (0.05 if none)."""
    run = 0
    best = 0
    for k in range(max(k0, 0), k1 + 1):
        blank = any(k >= len(v) or not np.isfinite(v[k]) for v in vals)
        run = run + 1 if blank else 0
        best = max(best, run)
    return (best + 1) * BIN


def trial_arrays(b, spec, asm, sched, cpr, tag_module, tag_offset=(0.0, 0.0, 0.0), imu_module=None):
    """One trial's binned columns -> RawTrial arrays on the 100 Hz grid + (gap s, {key: truth coverage}, qc list)."""
    T = int(round(sched.T * GRID_HZ))
    grid = np.arange(T) / GRID_HZ
    qc = []
    tb = b["t"]
    K = len(tb)
    if K == 0 or abs(tb[0]) > 1e-6 or (K > 1 and np.abs(np.diff(tb) - BIN).max() > 1e-6):
        qc.append("bins are not t = 0, 0.05, 0.10, ...")
    n = asm.n
    counts, duty = np.zeros((T, n)), np.full((T, n), np.nan)
    for w, name in enumerate(asm.wheel_corner):
        m, c = name.split(":")
        counts[:, w] = speed_to_counts(T, b.get(f"{m}_{c}_speed", np.full(K, np.nan)), cpr)
        duty[:, w] = bins_to_grid(T, b.get(f"{m}_{c}_duty", np.full(K, np.nan)))
    gz = bins_to_grid(T, b["gz"])
    yk = dict((m, kk) for m, i, j, kk in asm.modules).get(imu_module or asm.modules[0][0], 0)
    ax = b.get("ax", np.full(K, np.nan))
    ay = b.get("ay", np.full(K, np.nan))
    cy, sy = math.cos(yk * math.pi / 2), math.sin(yk * math.pi / 2)
    acc = np.stack([bins_to_grid(T, cy * ax - sy * ay), bins_to_grid(T, sy * ax + cy * ay)], 1)
    cmd = sched.twist(grid)
    tp = truth_to_grid(grid, tb, b["x"], b["y"], b["yaw"])
    truth = tag_to_origin(tp, asm, tag_module, tag_offset)
    # checks: logged cmd vs the fixed routine; robot-stream blank runs in the P1/P5/P2 holds; truth coverage per leg
    ref = sched.twist(tb)
    d = np.nanmax(np.abs(np.stack([b["cmd_vx"], b["cmd_vy"], b["cmd_wz"]], 1) - ref)) if K else math.nan
    if not (d <= 0.02):
        qc.append(f"logged cmd differs from the probe routine by up to {d:.3f} (check t = 0 / the routine)")
    robot = [b.get(f"{m}_{c}_{q}", np.full(K, np.nan)) for m, *_ in asm.modules for c in CORNERS
             for q in ("speed", "duty")] + [b["gz"]]
    gap, cover = BIN, {}
    tok = np.isfinite(b["x"]) & np.isfinite(b["y"]) & np.isfinite(b["yaw"])
    for pid, leg, ta, tb_, _ in sched.windows():
        key = f"{pid}{'+' if leg > 0 else '-'}"
        k0, k1 = int(math.floor(ta / BIN)), int(math.ceil(tb_ / BIN)) - 1
        if pid in ROBOT_PIDS:
            gap = max(gap, _max_blank_gap(robot, k0, k1))
        sel = (tb + BIN / 2 >= ta) & (tb + BIN / 2 <= tb_)
        cover[key] = float(tok[sel].mean()) if sel.any() else 0.0
    arr = dict(cmd=cmd, counts=counts, duty=duty, gyro_z=gz, acc=acc, lidar_twist=np.full((T, 3), np.nan),
               current=np.full((T, n), np.nan), truth_pose=truth)
    return grid, arr, gap, cover, qc


# ---------------------------------------------------------------------------------------------------- loading
def load_binned_session(path, schedule=None, params=None, allow_unverified=False, blocks=None, tag_offsets=None,
                        verdict=None):
    """Binned session folder -> list[RawTrial] (one per (layout, probe set)); see the module docstring.
    schedule: realv2_design.expand_schedule(seed) dict (labels of record + sealed links), or None (labels = the declared
    faults, no CAL/TEST split). blocks: keep only these trials.csv blocks."""
    import torch
    d = Path(path)
    rep = verdict or verify_binned_session(d)
    v = rep["verdict"]
    if v == "INVALID" or (v != "FIELD_OK" and not allow_unverified):
        raise SessionRefused(f"{d.name}: verdict {v}: {rep['errors'][:5]}")
    source = "FIELD" if v == "FIELD_OK" else "SIMULATION"
    params = params or A.load_params(PARAMS_PATH)
    cpr = float(params["motor"]["counts_per_output_rev"])
    B = read_binned(d)
    groups, qc = {}, []
    for k, t in enumerate(_read_rows(d / "trials.csv")):
        tid, blk = t["trial_id"], t["block"]
        if blk not in BLOCKS or (blocks is not None and blk not in blocks):
            continue
        if str(t["aborted"]).strip() not in ("0", ""):
            qc.append((tid, "aborted: not loaded (PROTOCOL_V2 §7 (c))"))
            continue
        spec = parse_layout(t["layout"])
        asm = A.build(asm_name(spec), spec, params)
        sched = Schedule(asm.char_length())
        tag = t.get("truth_tag") or spec[0][0]
        if tag not in [m for m, *_ in spec]:
            qc.append((tid, f"truth_tag {tag} is not in the layout: not loaded"))
            continue
        grid, arr, gap, cover, q = trial_arrays(B[tid], spec, asm, sched, cpr, tag, (tag_offsets or {}).get(tag, (0, 0, 0)),
                                                t.get("imu_module") or spec[0][0])
        qc += [(tid, x) for x in q]
        mode_e, g_e, why = fault_arrays(parse_fault(t["fault"]), asm)
        if schedule is not None and blk in SEALED_BLOCKS:
            ent, spec_s, f_s, link = sealed_entry(schedule, blk, t["seq"] or 0)
            if blk == "DEMO_SOLO":
                f_s = {kk: vv for kk, vv in f_s.items() if kk[0] == spec[0][0]}
                if len(spec) != 1 or spec[0][0] not in ent["dock_order"]:
                    why.append(f"DEMO_SOLO layout {t['layout']} is not one lone L3 module")
            elif sorted(m for m, *_ in spec_s) != sorted(m for m, *_ in spec):
                why.append(f"layout {t['layout']} != sealed {layout_str(spec_s)}")
            mode, g, bad_s = fault_arrays(f_s, asm)
            why += bad_s + L.compare_echo(mode, g, "none", mode_e, g_e, "none")
        else:
            mode, g = mode_e, g_e
            link = dict(episode="", phase="", round="", rep="", choice_block="", sheet_r3_split="NONE",
                        sheet_realv2_seq="")
        key = (tuple(asm.modules), PIDS)
        g_ = groups.setdefault(key, dict(asm=asm, sched=sched, grid=grid, arrs=[], ids=[], mode=[], g=[], mode_e=[],
                                         g_e=[], match=[], why=[], nums=[], link={}, checks={}, trows={}))
        full = f"{d.name}:{tid}:{k + 1}"
        for kk, vv in (("arrs", arr), ("ids", full), ("nums", k + 1), ("mode", mode), ("g", g), ("mode_e", mode_e),
                       ("g_e", g_e), ("match", not why), ("why", why)):
            g_[kk].append(vv)
        g_["checks"][full] = (gap, cover)
        g_["trows"][tid] = {"outcome_code": "1", "t_start_wall_ns": None, "block": blk, "seq": t["seq"]}
        for kk, vv in dict(link, block=blk, seq=t["seq"]).items():
            g_["link"].setdefault(kk, []).append(vv)
    out = []
    for key, g_ in groups.items():
        st = {k: np.stack([a[k] for a in g_["arrs"]]) for k in g_["arrs"][0]}
        asm_ = g_["asm"]
        nB = len(g_["ids"])
        raw = {k: torch.tensor(st[k], dtype=torch.float32) for k in ("cmd", "duty", "gyro_z", "acc", "lidar_twist",
                                                                       "current", "truth_pose")}
        raw["t"] = torch.tensor(g_["grid"], dtype=torch.float32)
        raw["counts"] = torch.tensor(st["counts"], dtype=torch.long)
        mset = "+".join(sorted(m for m, *_ in asm_.modules))
        labels = dict(mode=torch.tensor(np.stack(g_["mode"])), g=torch.tensor(np.stack(g_["g"])), ood="none",
                      ood_per_trial=["none"] * nB, schedule_trial=list(g_["nums"]),
                      fault_id=[L.infer_fault_id(m_, "none") for m_ in g_["mode"]],
                      fault_id_source=["schedule" if schedule is not None else "declared"] * nB,
                      n_faulted=[int((np.asarray(m_) != 0).sum()) for m_ in g_["mode"]], module_set=[mset] * nB,
                      assembly_type=asm_.name, mode_echo=torch.tensor(np.stack(g_["mode_e"])),
                      g_echo=torch.tensor(np.stack(g_["g_e"])), ood_echo=["none"] * nB,
                      schedule_match=list(g_["match"]), schedule_mismatch=list(g_["why"]))
        labels.update({k: list(v) for k, v in g_["link"].items()})
        raw["meta"] = dict(assembly=asm_, cpr=cpr, schedule=g_["sched"], source=source, trial_ids=g_["ids"],
                           wheel_radius=float(params["wheel"]["radius_m"]), truth_latency_s=0.0, session=d.name,
                           verdict=v, label=source, pids=key[1], csv_checks=g_["checks"], trials_rows=g_["trows"],
                           qc=qc, format_doc=FORMAT_DOC, fault_schedule_sha256=None, labels=labels)
        out.append(raw)
    return out


def load_binned_sessions(dirs, schedule=None, blocks=None, allow_simulation=False, params=None, tag_offsets=None):
    """[(dir, raw)] over several sessions (the builders' `loaded` structure)."""
    out = []
    for d in dirs:
        for raw in load_binned_session(d, schedule, params=params, allow_unverified=allow_simulation, blocks=blocks,
                                       tag_offsets=tag_offsets):
            out.append((Path(d), raw))
    return out


# ---------------------------------------------------------------------------------------------------- measurements
import re as _re

_FREE = _re.compile(r"^free_speed_(full_duty|half_duty|duty_?0?\.?5|0\.5_duty)_(?:(M[1-4])_)?(fl|fr|rl|rr)$", _re.I)
FF_REF_MPS = 0.65                     # the team's feed-forward reference (SPINE D17)


def read_measurements(path):
    """measurements.csv (`item, module, value, unit, method`) -> list of row dicts with float `value_f` (NaN if not a
    number). Items are free text except the parsed ones below."""
    rows = _read_rows(path)
    for r in rows:
        try:
            r["value_f"] = float(r.get("value", ""))
        except ValueError:
            r["value_f"] = math.nan
    return rows


def free_speeds(rows, radius=None):
    """v4 hook-check measurement (RUN_PLAN: wheels in the air, velocity loop bypassed, 5 s): the steady free wheel
    speed at duty 1.0 (`free_speed_full_duty_<wheel>`) and at duty 0.5 (`free_speed_half_duty_<wheel>`; also accepted:
    free_speed_duty05_..., free_speed_duty_0.5_..., free_speed_0.5_duty_...). <wheel> = fl|fr|rl|rr with the module in
    the `module` column, or M<k>_<corner>. Values in rad/s, or m/s (unit m/s) converted with the wheel radius.
    -> {"wheels": {"M1:fl": {"full": rad/s, "half": rad/s}}, "no_load_surface_mps": median full-duty speed x radius,
        "ff_ref_over_no_load": 0.65 / that, "half_over_full": median ratio (motor linearity), "problems": [...]}."""
    radius = float(radius if radius is not None else A.load_params(PARAMS_PATH)["wheel"]["radius_m"])
    out, probs = {}, []
    for r in rows:
        m = _FREE.match((r.get("item") or "").strip())
        if not m:
            continue
        which = "full" if m.group(1).lower() == "full_duty" else "half"
        mod = m.group(2) or (r.get("module") or "").strip()
        if mod not in MODULES:
            probs.append(f"{r['item']}: module {mod!r} unknown")
            continue
        v = r["value_f"]
        unit = (r.get("unit") or "rad/s").strip().lower().replace(" ", "")
        if unit in ("m/s", "mps"):
            v = v / radius
        elif unit in ("rpm",):
            v = v * 2 * math.pi / 60
        elif unit not in ("rad/s", "radps", "1/s"):
            probs.append(f"{r['item']} ({mod}): unit {unit!r} not rad/s, m/s or rpm")
            continue
        out.setdefault(f"{mod}:{m.group(3).lower()}", {})[which] = abs(v)
    full = [w["full"] for w in out.values() if math.isfinite(w.get("full", math.nan))]
    ratio = [w["half"] / w["full"] for w in out.values() if w.get("full") and math.isfinite(w.get("half", math.nan))]
    nl = float(np.median(full)) * radius if full else math.nan
    return dict(wheels=out, no_load_surface_mps=nl, ff_ref_over_no_load=(FF_REF_MPS / nl if nl else math.nan),
                half_over_full=float(np.median(ratio)) if ratio else math.nan, n_wheels=len(out), problems=probs)


# ---------------------------------------------------------------------------------------------------- docks
def dock_outcomes(docks_csv):
    """docks.csv -> [{episode (0-based, as on the run sheet), module, y, invalid, source}] for builders.build_demo_v3.
    y = 1 iff success == 1 and wall_contact == 0; an aborted dock is invalid (the episode is repeated, PROTOCOL_V2 R4 /
    §7). Several rows for one (episode, module): the last one counts (a repeat is listed after the original)."""
    rows = _read_rows(docks_csv)
    miss = [c for c in DOCKS_REQ if rows and c not in rows[0]]
    if miss:
        raise ValueError(f"{docks_csv}: missing columns {miss}")
    out = []
    for r in rows:
        inv = str(r.get("aborted", "0")) not in ("0", "")
        ok = str(r.get("success")) == "1" and str(r.get("wall_contact", "0")) in ("0", "")
        out.append(dict(episode=int(r["episode"]), module=r["module"], y=(math.nan if inv else float(ok)), invalid=inv,
                        source="docks.csv"))
    return out


# ---------------------------------------------------------------------------------------------------- sim writer
def _num(x, nd=6):
    x = float(x)
    return "" if not math.isfinite(x) else f"{x:.{nd}f}"


def bin_mean(x, K):
    """Team rule: mean over the 100 Hz samples in [0.05 k, 0.05 (k + 1)) = samples 5k .. 5k+4 (NaN-aware)."""
    x = np.asarray(x, float)
    out = np.full(K, np.nan)
    for k in range(K):
        s = x[PER * k: PER * k + PER]
        if len(s) and np.isfinite(s).any():
            out[k] = np.nanmean(s)
    return out


def camera_frames(t, pose, rng, rate=30.0, latency=0.15, pos_noise=0.002, yaw_noise=0.004):
    """Overhead-camera frames of a simulated pose track: capture times (phase-random, `rate` Hz), noisy poses, and
    receipt times (capture + latency). -> (t_capture, frames (k, 3), t_receipt)."""
    ph = rng.uniform(0, 1.0 / rate)
    tc = np.arange(ph, t[-1], 1.0 / rate)
    fr = np.stack([np.interp(tc, t, pose[:, j]) for j in range(3)], 1)
    fr = fr + rng.normal(0, 1, fr.shape) * np.array([pos_noise, pos_noise, yaw_noise])
    return tc, fr, tc + latency


def binned_rows(raw_b, asm, spec, cpr, tc, frames, lat_est_err=0.0, have_acc=True, cmd_origin_first=False, side=0.32):
    """The team's pre-processing of one simulated trial (CSV_FORMAT.md): raw_b = per-trial 100 Hz sensor arrays (cmd,
    counts, duty, gyro_z, acc) and camera frames of the TRUTH TAG MODULE (here: the first module) at capture times tc
    (the team corrects latency, leaving lat_est_err). -> {column: (K,) array}."""
    T = raw_b["duty"].shape[0]
    K = T // PER
    tb = np.arange(K) * BIN
    out = {"t": tb}
    cmd = raw_b["cmd"]
    c0 = cmd[np.clip((tb * GRID_HZ).round().astype(int), 0, T - 1)]
    if cmd_origin_first:
        cen = np.mean([[i * side, j * side] for _, i, j, _ in spec], 0) - np.array([spec[0][1], spec[0][2]]) * side
        c0 = np.stack([c0[:, 0] + c0[:, 2] * cen[1], c0[:, 1] - c0[:, 2] * cen[0], c0[:, 2]], 1)
    out.update(cmd_vx=c0[:, 0], cmd_vy=c0[:, 1], cmd_wz=c0[:, 2])
    cnt = raw_b["counts"]
    for w, name in enumerate(asm.wheel_corner):
        m, c = name.split(":")
        e = cnt[np.clip(PER * (np.arange(K) + 1), 0, T - 1), w] - cnt[PER * np.arange(K), w]
        out[f"{m}_{c}_speed"] = e / cpr * 2 * math.pi / BIN
        out[f"{m}_{c}_duty"] = bin_mean(raw_b["duty"][:, w], K)
    out["gz"] = bin_mean(raw_b["gyro_z"], K)
    out["ax"] = bin_mean(raw_b["acc"][:, 0], K) if have_acc else np.full(K, np.nan)
    out["ay"] = bin_mean(raw_b["acc"][:, 1], K) if have_acc else np.full(K, np.nan)
    tcen = tb + BIN / 2 + lat_est_err
    fy = np.unwrap(frames[:, 2])
    for j, k in enumerate(("x", "y")):
        out[k] = np.interp(tcen, tc, frames[:, j], left=np.nan, right=np.nan)
    out["yaw"] = np.interp(tcen, tc, fy, left=np.nan, right=np.nan)
    return out


def write_binned_session(out_dir, name, trials, binned, docks=None, notes="SIMULATION (capxfer.sim), never field data"):
    """Write a session folder in the CSV_FORMAT.md layout. trials: [dict(trial_id, seq, block, layout, fault,
    imu_module, truth_tag, aborted)]; binned: {trial_id: {column: (K,) array}}; docks: optional rows."""
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "trials.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(TRIALS_REQ)
        for t in trials:
            w.writerow([t.get(c, "") for c in TRIALS_REQ])
    cols = list(BINNED_REQ[:5])
    for t in trials:
        for c in wheel_cols(parse_layout(t["layout"])):
            if c not in cols:
                cols.append(c)
    cols += ["gz", "ax", "ay", "x", "y", "yaw"]
    with open(d / "binned.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for t in trials:
            b = binned.get(t["trial_id"])
            if b is None:
                continue
            for k in range(len(b["t"])):
                w.writerow([t["trial_id"], f"{b['t'][k]:.2f}"] + [_num(b[c][k]) if c in b else "" for c in cols[2:]])
    if docks is not None:
        with open(d / "docks.csv", "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(list(DOCKS_REQ[:3]) + ["success", "wall_contact", "final_x", "final_y", "final_yaw", "aborted"])
            for x in docks:
                w.writerow([x["episode"], x["module"], x["dock_order"], int(x["success"]), int(not x["success"]),
                            "", "", "", 0])
    (d / "notes.csv").write_text(f"text\n{notes}\n")
    return d


def simulate_trials(specs, modes, gs, seed=0, params=None, latency=0.15, lat_est_err=0.0, have_acc=True,
                    cmd_origin_first=False, device="cpu"):
    """Simulate trials of ONE layout and return (raw_full, binned) on the same sensor and camera realisation:
    raw_full = the full-rate in-memory RawTrial (sensors.to_raw, its truth replaced by the camera frames held from
    their receipt time, truth_latency_s = latency) and binned = {i: columns} = the team's pre-processing of exactly
    those samples and frames. specs: one layout spec; modes/gs: (B, n)."""
    import torch
    from .. import sensors as SN, sim as SIM, sysid
    params = params or A.load_params(PARAMS_PATH)
    spec = specs
    asm = A.build(asm_name(spec), spec, params)
    sched = Schedule(asm.char_length())
    B = len(modes)
    dom = sysid.nominal_domain(B, params)
    R = SIM.simulate(asm, params, dom, torch.as_tensor(np.asarray(modes)), torch.as_tensor(np.asarray(gs, np.float32)),
                     schedule=sched, device=device, seed=seed)
    R = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in R.items()}
    raw = SN.to_raw(R, params, dom, np.random.default_rng(seed), current_wired=False)
    rng = np.random.default_rng([seed, 7])
    t = raw["t"].numpy().astype(float)
    pose = R["pose"].numpy().astype(float)
    cpr = float(raw["meta"]["cpr"])
    tp = params["truth"]
    truth = np.full((B, len(t), 3), np.nan)
    binned = {}
    k0 = 0
    for b in range(B):
        # truth tag = the first module: its pose = origin pose composed with its position
        px, py = asm.module_pos[0]
        th = pose[b, :, 2]
        mp = np.stack([pose[b, :, 0] + np.cos(th) * px - np.sin(th) * py,
                       pose[b, :, 1] + np.sin(th) * px + np.cos(th) * py, th], 1)
        tc, fr, trc = camera_frames(t, mp, rng, float(tp.get("apriltag_rate_hz", 30.0)), latency,
                                    float(tp["apriltag_pos_noise_m"]), float(tp["apriltag_yaw_noise_rad"]))
        # full-rate path: the frames held from their receipt time (as the FieldLog loader), origin pose
        idx = np.searchsorted(trc, t, side="right") - 1
        frh = fr[np.clip(idx, 0, len(fr) - 1)]
        frh[idx < 0] = fr[0]
        truth[b] = tag_to_origin(frh, asm, spec[0][0])
        rb = {k: raw[k][b].numpy().astype(float) for k in ("cmd", "counts", "duty", "gyro_z", "acc")}
        binned[b] = binned_rows(rb, asm, spec, cpr, tc, fr, lat_est_err, have_acc, cmd_origin_first,
                                float(params["module"]["side_x_m"]))
        k0 += 1
    raw["truth_pose"] = torch.tensor(truth, dtype=torch.float32)
    raw["lidar_twist"] = torch.full_like(raw["lidar_twist"], float("nan"))
    raw["meta"]["truth_latency_s"] = latency
    raw["meta"]["wheel_radius"] = float(params["wheel"]["radius_m"])
    return raw, binned, asm


# ---------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("validate")
    p.add_argument("session")
    p.add_argument("--seed-file", help="the sealed seed (labels of record); without it the declared faults are used")
    a = ap.parse_args(argv)
    r = verify_binned_session(a.session)
    print(json.dumps(r, indent=1))
    if r["verdict"] == "INVALID":
        return 1
    sched = schedule_from_seed(a.seed_file) if a.seed_file else None
    raws = load_binned_session(a.session, sched, allow_unverified=True)
    print(f"loaded {sum(len(x['meta']['trial_ids']) for x in raws)} probe trials in {len(raws)} group(s)")
    for x in raws:
        lab = x["meta"]["labels"]
        for tid, ok, why in zip(x["meta"]["trial_ids"], lab["schedule_match"], lab["schedule_mismatch"]):
            gap, _ = x["meta"]["csv_checks"][tid]
            if not ok or gap > 0.1:
                print(f"  {tid}: schedule_match={ok} {why} max_gap={gap:.3f}s")
    for q in (raws[0]["meta"]["qc"] if raws else []):
        print("  qc", *q)
    return 0


if __name__ == "__main__":
    sys.exit(main())

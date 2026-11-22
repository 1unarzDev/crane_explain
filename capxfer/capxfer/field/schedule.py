"""Sealed, randomised fault schedule for the main collection (plan: assemblies x faults in randomised blocks).

  python -m capxfer.field.schedule OUT_DIR [--seed-dir SEALED_DIR] [--per-assembly-scale 1.0]

Writes
  OUT_DIR/schedule.json            the schedule (no seed inside); canonical JSON
  OUT_DIR/schedule.sha256          sha256 of schedule.json (hash it into the freeze BEFORE the main collection)
  OUT_DIR/runsheet_<asm>.csv       one FieldLog v2 run sheet per assembly session (portion capxprobe, <= 999 rows)
  OUT_DIR/commit.json              {schedule_sha256, seed_sha256} (publishable commitment)
  SEED_DIR/seed.txt                the 128-bit seed (kept sealed, separately; reveals nothing until opened)
`verify(out_dir, seed_hex)` regenerates the schedule from the seed and checks the hash.

Faults: F0 healthy; F1 weak (g ~ U(0.3, 0.9)); F2 coast; F3 brake; F5 two simultaneous firmware faults (weak/coast/brake)
are firmware-automated and run in randomised automated blocks. F4 roller-locked (zip-tie), F6 payload (lifter places mass on
a neighbour; out-of-model), F7 low-friction sheet (out-of-model) are operator blocks at randomised positions. D3 (pre-freeze
2026-11-20): F6 = 4-6 kg on a lifter deck, CG 8-12 cm toward a corner, single/pair only (other types run those blocks as
F7); F7 = sheet mu 0.12-0.35 under the left (outbound strafe) wheels, edge under the centre line. In every run
of 10 F6/F7 block trials, 5 (shuffled) also carry two passive firmware wheel faults (coast/brake, distinct wheels): F6w/F7w
(trial["condition"]; trial["fault_id"] stays F6/F7, the OOD condition, so codes and run sheets are unchanged). Assembly-type
order is randomised. Every trial runs the full probe library.
"""
from __future__ import annotations

import csv
import hashlib
import json
import secrets
from pathlib import Path

import numpy as np

from .. import assembly as A
from ..probes import PIDS, Schedule
from .fieldlog_config import PORTION_CODE, PORTION_TOKEN

SCHEMA = "capxfer.fault_schedule/v1"
ASSEMBLIES = ("single", "pair", "L3", "square4")
FIRMWARE_FAULTS = ("F0", "F1", "F2", "F3", "F5")
OPERATOR_FAULTS = ("F4", "F6", "F7")
FAULT_CODE = {f"F{k}": k for k in range(8)}
# D3 (pre-freeze 2026-11-20): defaults = the planned per-type sample of PROTOCOL section 3 (F0 50, F1 60, F2 40, F3 40,
# F4 40, F5 60, F6 30, F7 30; operator blocks of 10).
DEFAULT_COUNTS = {"F0": 50, "F1": 60, "F2": 40, "F3": 40, "F5": 60}         # per assembly, automated
DEFAULT_OPERATOR = {"F4": (4, 10), "F6": (3, 10), "F7": (3, 10)}           # per assembly: (blocks, trials per block)
AUTO_BLOCK = 20
G_RANGE = (0.3, 0.9)
FW_MODES = ("weak", "coast", "brake")
OOD_WHEEL_FAULT_IDS = ("F6", "F7")        # operator blocks whose trials may also carry a firmware wheel fault (F6w/F7w)
OOD_WHEEL_FAULT_PER_10 = 5
# D3 consequential OOD field specs (WS-A2 Task 4, docs/coordination/A.md; PROTOCOL 'Pre-freeze changes'):
# F6 payload: rigid 4-6 kg on a lifter deck (lifter stowed), CG 8-12 cm off the module centre toward one corner
#    (diagonal); single and pair only (never consequential on L3/square4 in sim); never > 6 kg, never > 12 cm.
PAYLOAD_KG = (4.0, 6.0)
PAYLOAD_OFFSET_M = (0.08, 0.12)
PAYLOAD_MAX_KG, PAYLOAD_MAX_OFFSET_M = 6.0, 0.12
F6_TYPES = ("single", "pair")             # other types: their F6 blocks are run as F7 blocks (same trial count)
# F7 low-friction sheet: wheel-sheet mu 0.12-0.35, sheet edge under the assembly centre line, covering the wheels on
#    the outbound strafe side (left of the assembly: P2+/P6y+ drive toward it).
SHEET_MU = (0.12, 0.35)
SHEET_SIDE = "left"
# F6w/F7w: TWO passive firmware faults (coast/brake each uniform) on two distinct wheels (uniform). With one fault a
#    decision leg never drops below tau = 0.5 in sim; with two passive faults the single module sits at tau (D3).
OOD_WHEEL_FAULT_MODES = ("coast", "brake")
OOD_WHEEL_FAULT_N = 2
# D3b (owner-approved 2026-11-20): per-type allocation overrides of the default schedule. Single gets 5 F6 + 5 F7
# blocks with 9 of 10 trials F6w/F7w (consequential OOD for H3 lives almost only on single), paid for by 20 single F1
# trials (weak single faults: diagnosis saturated) and +20 trials. Applied only when generate() is called with the
# default counts/operator (explicit counts/operator = no overrides unless type_overrides is passed).
DEFAULT_TYPE_OVERRIDES = {"single": {"counts": {"F1": 40}, "operator": {"F6": [5, 10], "F7": [5, 10]},
                                     "ood_w_per_10": 9}}
# D3c (critic review 2026-11-20, owner/integrator-approved): the main collection is split into N_SESSIONS FieldLog
# sessions and EVERY session holds blocks of ALL assembly types (type is not confounded with session/day). Each type's
# blocks are stratified over the sessions (dealt round-robin per block kind, so every fault cell is spread across
# sessions); inside a session the block order is randomised over types ("block"), with a reassembly whenever two
# consecutive blocks have different types. "segment" = each type's blocks of a session run consecutively, types in a
# random order (fewer reassemblies; sensitivity option).
N_SESSIONS = 12
SESSION_ORDER = "block"


# WS-P2 (2026-11-21): a trial may carry "probes": a subset of the probe library (masking/split and module-choice
# trials run P1+P5+P2 only, 12.7 s). Absent = the full library (every main-collection trial). The loader, the
# validator, the probe runner and the synthetic sessions all take the trial's probe schedule from trial_schedule().
SHORT_PIDS = ("P1", "P2", "P5")
KNOWN_PID_SETS = (PIDS, SHORT_PIDS)


def trial_pids(trial=None):
    """Probe primitives a scheduled trial runs, in library order (the full library when the trial has no "probes")."""
    want = (trial or {}).get("probes")
    if not want:
        return PIDS
    bad = [p for p in want if p not in PIDS]
    if bad:
        raise ValueError(f"unknown probe primitive(s) {bad}")
    return tuple(p for p in PIDS if p in want)


def trial_schedule(asm, trial=None, duration_s=None):
    """probes.Schedule of one trial: from the scheduled trial's "probes" if given, else (no schedule at hand) the known
    probe set whose duration is closest to the logged start..end duration, else the full library."""
    if trial is not None:
        return Schedule(asm.char_length(), pids=trial_pids(trial))
    if duration_s is not None:
        cands = [Schedule(asm.char_length(), pids=p) for p in KNOWN_PID_SETS]
        return min(cands, key=lambda s: abs(s.T - float(duration_s)))
    return Schedule(asm.char_length())


def sensor_modules(asm):
    """(imu_module, lidar_module) used for an assembly session: the first lidar module in assembly order (else the first
    module, and no lidar). Render the FieldLog config per session with IMU_MODULE / LIDAR_MODULE set to these."""
    mods = [m for m, *_ in asm.modules]
    lid = [m for m, has in zip(mods, asm.lidar) if has]
    return (lid[0] if lid else mods[0]), (lid[0] if lid else None)


def _fault(rng, wheel, mode):
    f = {"wheel": wheel, "mode": mode}
    if mode == "weak":
        f["g"] = round(float(rng.uniform(*G_RANGE)), 3)
    return f


def _firmware_trial(rng, fid, wheels):
    if fid == "F0":
        return []
    if fid in ("F1", "F2", "F3"):
        mode = {"F1": "weak", "F2": "coast", "F3": "brake"}[fid]
        return [_fault(rng, str(rng.choice(wheels)), mode)]
    ws = rng.choice(wheels, size=2, replace=False)                             # F5
    return [_fault(rng, str(w), str(rng.choice(FW_MODES))) for w in ws]


def _operator_block(rng, fid, asm, n_trials, w_per_10=OOD_WHEEL_FAULT_PER_10):
    """One operator block: a fixed physical condition for n_trials trials (w_per_10: F6w/F7w trials per run of 10)."""
    wheels = list(asm.wheel_corner)
    if fid == "F4":
        w = str(rng.choice(wheels))
        cond = {"kind": "roller_locked", "wheel": w, "instruction": f"zip-tie the rollers of wheel {w}"}
        faults = [{"wheel": w, "mode": "roller_locked"}]
    elif fid == "F6":
        mods = [m for m, *_ in asm.modules]
        lifters = [m for m in mods if m in set(params_lifters(asm))] or mods
        target = str(rng.choice(lifters))
        kg = round(float(rng.uniform(*PAYLOAD_KG)), 2)
        r = round(float(rng.uniform(*PAYLOAD_OFFSET_M)), 3)
        corner = str(rng.choice(["fl", "fr", "rl", "rr"]))
        sx, sy = (1 if corner[0] == "f" else -1), (1 if corner[1] == "l" else -1)
        k = mods.index(target)
        yaw = 0.5 * np.pi * int(asm.modules[k][3])
        ux, uy = sx * r / np.sqrt(2), sy * r / np.sqrt(2)                       # module frame, toward the corner
        dx = round(float(np.cos(yaw) * ux - np.sin(yaw) * uy), 4)               # assembly frame
        dy = round(float(np.sin(yaw) * ux + np.cos(yaw) * uy), 4)
        if kg > PAYLOAD_MAX_KG or r > PAYLOAD_MAX_OFFSET_M:
            raise ValueError("payload outside the safety envelope (> 6 kg or > 12 cm)")
        cond = {"kind": "payload", "module": target, "mass_kg": kg, "offset_m": r, "corner": corner,
                "dx_m": dx, "dy_m": dy,
                "instruction": (f"strap {kg} kg on the lifter deck of {target} (lifter stowed), CG {100 * r:.1f} cm "
                                f"from the module centre toward its {corner} corner")}
        faults = [{"wheel": "none", "mode": "payload", "module": target, "mass_kg": kg, "dx_m": dx, "dy_m": dy}]
    else:
        side = SHEET_SIDE
        mu = round(float(rng.uniform(*SHEET_MU)), 3)
        cond = {"kind": "low_mu", "region": side, "mu": mu,
                "instruction": (f"low-friction sheet (wheel-sheet mu {mu:.2f}; use the sheet whose measured mu is "
                                f"nearest, log the measured value) with its edge under the assembly centre line, "
                                f"covering the {side} wheels")}
        faults = [{"wheel": "none", "mode": "low_mu", "region": side, "mu": mu}]
    trials = [list(faults) for _ in range(n_trials)]
    sub = [fid] * n_trials
    if fid in OOD_WHEEL_FAULT_IDS:
        # F6w/F7w (prereg section 3, D3): in every run of 10 block trials, OOD_WHEEL_FAULT_PER_10 (5) also carry
        # OOD_WHEEL_FAULT_N (2) passive firmware faults (coast/brake uniformly) on distinct wheels (uniformly), at
        # shuffled positions. A shorter last run gets half of its length (rounded down). fault_id stays F6/F7;
        # trial["condition"] says F6w/F7w.
        nf = min(OOD_WHEEL_FAULT_N, len(wheels))
        for c0 in range(0, n_trials, 10):
            run = list(range(c0, min(c0 + 10, n_trials)))
            k = int(w_per_10) if len(run) == 10 else (len(run) * int(w_per_10)) // 10
            for i in sorted(int(j) for j in rng.choice(run, size=k, replace=False)):
                ws = rng.choice(wheels, size=nf, replace=False)
                trials[i] = trials[i] + [_fault(rng, str(w), str(rng.choice(OOD_WHEEL_FAULT_MODES))) for w in ws]
                sub[i] = fid + "w"
    return cond, trials, sub


def params_lifters(asm):
    """Modules with a lifter deck (params module.lifter_modules, attached to the assembly by generate)."""
    return getattr(asm, "_lifters", None) or []


def _kind_key(b):
    return "auto" if b["kind"] == "auto" else b["fault_id"]


def assign_sessions(rng, sessions, n_sessions=N_SESSIONS, order=SESSION_ORDER):
    """Interleave all assembly types into n_sessions field sessions (D3c). Mutates the blocks (adds `field_session`,
    `session_pos`) and returns the top-level `field_sessions` list (run order). For every type, its blocks are grouped
    by kind (auto / F4 / F6 / F7), shuffled within kind and dealt round-robin over the sessions from a random start
    that carries over between kinds (balanced block counts, every cell spread over sessions). Inside a session the
    blocks are shuffled ("block") or grouped per type in a random type order ("segment")."""
    n_sessions = int(n_sessions)
    if order not in ("block", "segment"):
        raise ValueError(order)
    bins = [[] for _ in range(n_sessions)]
    for s in sessions:
        name = s["assembly"]
        kinds = {}
        for b in s["blocks"]:
            kinds.setdefault(_kind_key(b), []).append(b)
        k = int(rng.integers(n_sessions))
        for kk in sorted(kinds):
            bl = kinds[kk]
            for i in rng.permutation(len(bl)):
                bins[k % n_sessions].append((name, bl[int(i)]))
                k += 1
    out = []
    for j, bl in enumerate(bins):
        if order == "block":
            bl = [bl[int(i)] for i in rng.permutation(len(bl))]
        else:
            types = sorted({n for n, _ in bl})
            types = [types[int(i)] for i in rng.permutation(len(types))]
            bl = [x for t in types for x in [y for y in bl if y[0] == t]]
        sid = f"S{j + 1:02d}"
        rows, prev = [], None
        for pos, (name, b) in enumerate(bl, start=1):
            b["field_session"], b["session_pos"] = sid, pos
            rows.append({"pos": pos, "block": b["block"], "assembly": name, "kind": b["kind"],
                         "fault_id": b["fault_id"], "n_trials": len(b["trials"]), "reassemble": name != prev})
            prev = name
        out.append({"id": sid, "blocks": rows, "n_trials": sum(r["n_trials"] for r in rows),
                    "assemblies": sorted({r["assembly"] for r in rows}),
                    "n_reassemblies": sum(r["reassemble"] for r in rows)})
    return out


def generate(seed_hex, assemblies=ASSEMBLIES, counts=None, operator=None, params=None, auto_block=AUTO_BLOCK,
             f6_types=F6_TYPES, type_overrides=None, n_sessions=N_SESSIONS, session_order=SESSION_ORDER):
    """Deterministic schedule from a hex seed. Returns the schedule dict (without the seed).
    f6_types: assembly types that get F6 payload blocks; for the others the F6 blocks are run as F7 blocks (D3).
    type_overrides: {assembly: {counts: {...}, operator: {F: [blocks, size]}, ood_w_per_10: k}} (default
    DEFAULT_TYPE_OVERRIDES when counts and operator are both defaulted, else none).
    n_sessions / session_order: FieldLog sessions, each with blocks of every type (D3c; see assign_sessions).
    The schedule's `sessions` stay one entry per assembly type (consumers index trials by type); the run order is the
    top-level `field_sessions`."""
    if type_overrides is None:
        type_overrides = DEFAULT_TYPE_OVERRIDES if (counts is None and operator is None) else {}
    counts0 = dict(DEFAULT_COUNTS, **(counts or {}))
    operator0 = dict(DEFAULT_OPERATOR, **(operator or {}))
    counts, operator = counts0, operator0
    params = params or A.load_params(Path(__file__).resolve().parents[2] / "params" / "module_measured.yaml")
    rng = np.random.default_rng(int(seed_hex, 16))
    order = [assemblies[i] for i in rng.permutation(len(assemblies))]
    sessions, trial_no, block_no = [], 1, 1
    for name in order:
        ov = type_overrides.get(name, {})
        counts = dict(counts0, **ov.get("counts", {}))
        operator = dict(operator0, **{k: tuple(v) for k, v in ov.get("operator", {}).items()})
        w10 = int(ov.get("ood_w_per_10", OOD_WHEEL_FAULT_PER_10))
        asm = A.physical(name, params)
        asm._lifters = list(params["module"].get("lifter_modules", []))
        wheels = list(asm.wheel_corner)
        auto = [f for f in FIRMWARE_FAULTS for _ in range(int(counts.get(f, 0)))]
        auto = [auto[i] for i in rng.permutation(len(auto))]
        auto_blocks = [auto[i:i + auto_block] for i in range(0, len(auto), auto_block)]
        op_blocks = [fid for fid in OPERATOR_FAULTS for _ in range(int(operator[fid][0]))]
        op_size = {fid: int(operator[fid][1]) for fid in OPERATOR_FAULTS}
        if f6_types is not None and name not in f6_types:                    # D3: F6 blocks run as F7 blocks
            op_blocks = ["F7*" if b_ == "F6" else b_ for b_ in op_blocks]
            op_size["F7*"] = op_size["F6"]
        kinds = ["auto"] * len(auto_blocks) + op_blocks
        kinds = [kinds[i] for i in rng.permutation(len(kinds))]
        blocks, ai = [], 0
        for kind in kinds:
            if kind == "auto":
                fids = auto_blocks[ai]
                ai += 1
                trials = [{"fault_id": f, "condition": f, "faults": _firmware_trial(rng, f, wheels)} for f in fids]
                cond = {"kind": "firmware_auto", "instruction": "automated: no physical change"}
            else:
                fid_ = kind.rstrip("*")
                cond, fl, sub = _operator_block(rng, fid_, asm, op_size[kind], w10)
                kind = fid_
                trials = [{"fault_id": kind, "condition": c, "faults": f} for f, c in zip(fl, sub)]
            for t in trials:
                t["trial"] = trial_no
                t["ood"] = {"F6": "payload", "F7": "low_mu"}.get(t["fault_id"], "none")
                trial_no += 1
            blocks.append({"block": block_no, "kind": "auto" if kind == "auto" else "operator",
                           "fault_id": None if kind == "auto" else kind, "condition": cond, "trials": trials})
            block_no += 1
        frames = [{"module": m, "x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "yaw_deg": 90 * int(yk)}
                  for (m, _, _, yk), p in zip(asm.modules, asm.module_pos)]
        imu_m, lidar_m = sensor_modules(asm)
        sessions.append({"assembly": name, "modules": [list(m) for m in asm.modules], "frames": frames,
                         "imu_module": imu_m, "lidar_module": lidar_m, "blocks": blocks})
    fsess = assign_sessions(rng, sessions, n_sessions, session_order)
    return {"schema": SCHEMA, "generator": "capxfer.field.schedule", "assemblies_order": order,
            "counts_per_assembly": counts0, "operator_blocks_per_assembly": {k: list(v) for k, v in operator0.items()},
            "type_overrides": {k: {kk: (dict(vv) if isinstance(vv, dict) else vv) for kk, vv in v.items()}
                               for k, v in type_overrides.items()},
            "n_sessions": int(n_sessions), "session_order": session_order,
            "auto_block": auto_block, "g_range": list(G_RANGE), "payload_kg": list(PAYLOAD_KG),
            "payload_offset_m": list(PAYLOAD_OFFSET_M), "sheet_mu": list(SHEET_MU), "sheet_side": SHEET_SIDE,
            "f6_types": list(f6_types) if f6_types is not None else None,
            "ood_wheel_faults": dict(n=OOD_WHEEL_FAULT_N, modes=list(OOD_WHEEL_FAULT_MODES),
                                     per_10=OOD_WHEEL_FAULT_PER_10),
            "n_trials": trial_no - 1, "sessions": sessions, "field_sessions": fsess}


def field_sessions(sched):
    """The run order: list of field sessions {id, blocks: [{pos, block, assembly, kind, fault_id, n_trials,
    reassemble}], n_trials, assemblies, n_reassemblies} (D3c). Schedules without it: one session per assembly."""
    if sched.get("field_sessions"):
        return sched["field_sessions"]
    return [{"id": s["assembly"], "assemblies": [s["assembly"]], "n_reassemblies": 1,
             "n_trials": sum(len(b["trials"]) for b in s["blocks"]),
             "blocks": [{"pos": i + 1, "block": b["block"], "assembly": s["assembly"], "kind": b["kind"],
                         "fault_id": b["fault_id"], "n_trials": len(b["trials"]), "reassemble": i == 0}
                        for i, b in enumerate(s["blocks"])]} for s in sched["sessions"]]


def sub_schedule(sched, field_session_id, assembly=None):
    """The schedule restricted to one field session (optionally one assembly type), blocks in run order, same trial
    numbers. For per-session run sheets, the probe runner and synthetic sessions. Not sealed itself: its trials are a
    subset of the sealed schedule's."""
    fs = next((f for f in field_sessions(sched) if f["id"] == field_session_id), None)
    if fs is None:
        raise KeyError(f"no field session {field_session_id}")
    pos = {(r["assembly"], r["block"]): r["pos"] for r in fs["blocks"]}
    ss = []
    for s in sched["sessions"]:
        if assembly is not None and s["assembly"] != assembly:
            continue
        bl = sorted((b for b in s["blocks"] if (s["assembly"], b["block"]) in pos), key=lambda b: pos[(s["assembly"], b["block"])])
        if bl:
            ss.append(dict(s, blocks=bl))
    if not ss:
        raise KeyError(f"field session {field_session_id} has no blocks of {assembly}")
    ss.sort(key=lambda s: min(pos[(s["assembly"], b["block"])] for b in s["blocks"]))
    return dict(sched, sessions=ss, n_trials=sum(len(b["trials"]) for s in ss for b in s["blocks"]),
                field_sessions=[fs], field_session=field_session_id)


def session_runsheet_rows(sched, field_session_id):
    """FieldLog run-sheet rows of one (mixed-type) field session in run order: [(assembly, block, row dict)]."""
    fs = next(f for f in field_sessions(sched) if f["id"] == field_session_id)
    byb = {(s["assembly"], b["block"]): b for s in sched["sessions"] for b in s["blocks"]}
    out, k = [], 0
    for r in fs["blocks"]:
        for t in byb[(r["assembly"], r["block"])]["trials"]:
            k += 1
            if k > 999:
                raise ValueError("more than 999 trials in one FieldLog session: use more sessions")
            out.append((r["assembly"], r["block"], {"row": k, "trial_id": f"gnd_{PORTION_TOKEN}_{k:03d}",
                        "portion_code": PORTION_CODE, "nn": k, "split_code": 1, "arm_code": FAULT_CODE[t["fault_id"]],
                        "block_code": int(r["pos"]), "amp": 0, "sign": 0, "dur_s": 0, "gap_s": 0, "extra_wait_s": 0,
                        "axis_code": 0, "thr_idx": 0, "n_steps": 0}))
    return out


BLOCK_SHEET_HEADER = ["pos", "block", "assembly", "reassemble_before", "kind", "fault_id", "n_trials", "first_row",
                      "last_row", "schedule_trials", "instruction"]


def write_block_sheet(path, sched, field_session_id):
    """Operator block sheet of one field session: block order, reassembly points, instructions, run-sheet rows."""
    fs = next(f for f in field_sessions(sched) if f["id"] == field_session_id)
    byb = {(s["assembly"], b["block"]): b for s in sched["sessions"] for b in s["blocks"]}
    row = 1
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BLOCK_SHEET_HEADER, lineterminator="\n")
        w.writeheader()
        for r in fs["blocks"]:
            b = byb[(r["assembly"], r["block"])]
            n = len(b["trials"])
            w.writerow({"pos": r["pos"], "block": r["block"], "assembly": r["assembly"],
                        "reassemble_before": int(r["reassemble"]), "kind": r["kind"], "fault_id": r["fault_id"] or "",
                        "n_trials": n, "first_row": row, "last_row": row + n - 1,
                        "schedule_trials": f"{b['trials'][0]['trial']}-{b['trials'][-1]['trial']}",
                        "instruction": b["condition"].get("instruction", "")})
            row += n
    return path


def canonical(sched):
    return json.dumps(sched, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def iter_trials(sched, assembly=None):
    for s in sched["sessions"]:
        if assembly is not None and s["assembly"] != assembly:
            continue
        for b in s["blocks"]:
            for t in b["trials"]:
                yield s, b, t


def runsheet_rows(sched, assembly):
    """FieldLog v2 run-sheet rows (15-column layout) for one assembly session: one row per probe trial."""
    rows = []
    for k, (s, b, t) in enumerate(iter_trials(sched, assembly), start=1):
        if k > 999:
            raise ValueError("more than 999 trials in one FieldLog session: split the session")
        rows.append({"row": k, "trial_id": f"gnd_{PORTION_TOKEN}_{k:03d}", "portion_code": PORTION_CODE, "nn": k,
                     "split_code": 1, "arm_code": FAULT_CODE[t["fault_id"]], "block_code": 0, "amp": 0, "sign": 0,
                     "dur_s": 0, "gap_s": 0, "extra_wait_s": 0, "axis_code": 0, "thr_idx": 0, "n_steps": 0})
    return rows


RUNSHEET_HEADER = ["row", "trial_id", "portion_code", "nn", "split_code", "arm_code", "block_code", "amp", "sign", "dur_s",
                   "gap_s", "extra_wait_s", "axis_code", "thr_idx", "n_steps"]


def write_runsheet(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNSHEET_HEADER, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def write(out_dir, seed_hex=None, seed_dir=None, **kw):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_hex = seed_hex or secrets.token_hex(16)
    sched = generate(seed_hex, **kw)
    raw = canonical(sched)
    (out / "schedule.json").write_bytes(raw)
    h = sha256_bytes(raw)
    (out / "schedule.sha256").write_text(f"{h}  schedule.json\n")
    for s in sched["sessions"]:
        write_runsheet(out / f"runsheet_{s['assembly']}.csv", runsheet_rows(sched, s["assembly"]))
    for fs in sched.get("field_sessions", []):                 # D3c: one FieldLog run sheet + block sheet per session
        write_runsheet(out / f"runsheet_{fs['id']}.csv", [r for _, _, r in session_runsheet_rows(sched, fs["id"])])
        write_block_sheet(out / f"blocks_{fs['id']}.csv", sched, fs["id"])
    commit = {"schedule_sha256": h, "seed_sha256": sha256_bytes(seed_hex.encode()), "schema": SCHEMA}
    (out / "commit.json").write_text(json.dumps(commit, indent=1) + "\n")
    if seed_dir is not None:
        sd = Path(seed_dir)
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "seed.txt").write_text(seed_hex + "\n")
        try:
            (sd / "seed.txt").chmod(0o400)
        except OSError:
            pass
    return {"schedule": out / "schedule.json", "sha256": h, "seed_hex": seed_hex, "commit": commit}


def load(path, expect_sha256=None):
    """Load schedule.json and check it against schedule.sha256 next to it (or the given hash). Refuses on mismatch."""
    p = Path(path)
    raw = p.read_bytes()
    h = sha256_bytes(raw)
    want = expect_sha256
    if want is None and (p.parent / "schedule.sha256").is_file():
        want = (p.parent / "schedule.sha256").read_text().split()[0]
    if want is None:
        raise ValueError("no schedule.sha256 to check against: refusing an unsealed schedule")
    if h != want:
        raise ValueError(f"schedule sha256 {h} != sealed {want}")
    return json.loads(raw)


def verify(out_dir, seed_hex, **kw):
    """True if the seed regenerates exactly the sealed schedule. Generation settings are read from the sealed file."""
    out = Path(out_dir)
    want = (out / "schedule.sha256").read_text().split()[0]
    s = json.loads((out / "schedule.json").read_text())
    kw = dict(dict(assemblies=tuple(sorted(s["assemblies_order"], key=ASSEMBLIES.index)), counts=s["counts_per_assembly"],
                   operator={k: tuple(v) for k, v in s["operator_blocks_per_assembly"].items()}, auto_block=s["auto_block"],
                   **({"f6_types": tuple(s["f6_types"]) if s["f6_types"] is not None else None}
                      if "f6_types" in s else {"f6_types": None}),
                   type_overrides=s.get("type_overrides", {}), n_sessions=s.get("n_sessions", 1),
                   session_order=s.get("session_order", SESSION_ORDER)), **kw)
    return sha256_bytes(canonical(generate(seed_hex, **kw))) == want and \
        json.loads((out / "commit.json").read_text())["seed_sha256"] == sha256_bytes(seed_hex.encode())


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--seed-dir", required=True, help="separate (sealed) folder for seed.txt")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply automated counts and operator block sizes")
    a = ap.parse_args(argv)
    counts = {k: max(1, int(round(v * a.scale))) for k, v in DEFAULT_COUNTS.items()}
    operator = {k: (b, max(1, int(round(n * a.scale)))) for k, (b, n) in DEFAULT_OPERATOR.items()}
    r = write(a.out_dir, seed_dir=a.seed_dir, counts=counts, operator=operator)
    print(f"schedule {r['schedule']}  sha256 {r['sha256']}  trials {json.loads(Path(r['schedule']).read_text())['n_trials']}")
    print(f"seed written to {a.seed_dir}/seed.txt (keep sealed; publish commit.json)")


if __name__ == "__main__":
    main()

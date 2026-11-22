"""FieldLog v2 session (capxfer layout, fieldlog_config) -> list[RawTrial] (CONTRACTS C1), one per assembly.

  load_session(path, allow_unverified=False, params=None, truth_latency_s=None, tag_offset=(0, 0, 0))

* Gate: the session is screened with FieldLog's verify_session.verify(); anything but FIELD_OK is refused unless
  allow_unverified=True (then FIELD_WITH_WARNINGS, SIMULATION and INVALID sessions load, labelled by their own label).
  `source` is "FIELD" only when session.yaml says FIELD **and** the verifier agrees; otherwise "SIMULATION".
  This module only reads; it never writes inside a session.
* Segmentation: probe trials are delimited by the probe runner's trial markers (src TRIAL: start ... end; an "abort"
  drops the trial) inside each FieldLog trial file; trial t = 0 is the receipt time of the start marker.
* Timing: every stream is placed by its logger receipt time (t_mono_ns), i.e. as received, like the sim sensor model;
  AprilTag latency stays in truth_pose and featurize shifts its windows by meta['truth_latency_s'].
* Resampling: zero-order hold onto the 100 Hz grid t = 0, 0.01, ..., < Schedule.T (before the first sample: the first
  sample). Absent streams -> NaN (lidar_twist if no rf2o, current if not wired).
* Frames: IMU (IMU module frame) and rf2o odometry (lidar module centre, module frame; TEAM-CONFIRM the mount offset)
  are rotated/transported into the assembly frame. Truth pose = tag pose corrected by `tag_offset` (dx, dy, dyaw of the
  tag in the assembly frame) to the assembly origin. truth_pose and labels are EVALUATOR-ONLY.
* Labels from the fault_state echoes logged between the start marker and the first non-zero command:
  mode (B,n) / g (B,n); payload / low_mu operator entries -> labels['ood'] (per-trial list in labels['ood_per_trial']).
* Trial ids / fault ids (FIX M4, CONTRACTS C1 labels, evaluator-only): every RawTrial's meta['labels'] also carries
  per-trial lists `schedule_trial` (int, the probe runner's trial number from the TRIAL marker), `fault_id` (F0..F7;
  "F6+F2"-style when an out-of-model condition is combined with a wheel fault), `fault_id_source`, `n_faulted`,
  `module_set` ("M1+M3", sorted module ids), plus scalar `assembly_type`. Without a schedule, fault_id is inferred from
  the echoes (`fault_id_source` = "echo"). With `schedule=` (a sealed schedule.json path, hash-checked, or the loaded
  dict) the trial is joined on (assembly type, schedule trial number): fault_id, mode, g and ood come from the sealed
  schedule (`fault_id_source` = "schedule"; echo-derived values kept as `mode_echo` / `g_echo` / `ood_echo`), and
  every trial is cross-checked against its echoes -> `schedule_match` (bool) and `schedule_mismatch` (reasons).
  Firmware faults (weak/coast/brake) must be echoed exactly (g within 1e-3); operator conditions (roller_locked,
  payload, low_mu) are physical, so a missing echo is accepted but a contradicting one is not. A mismatch is the
  prereg exclusion 7(b); `on_mismatch="raise"` refuses the session instead of flagging. A trial number absent from the
  schedule always raises.
* WS-P2 (2026-11-21): a trial's probe schedule is schedule.trial_schedule(): the sealed trial's "probes" subset (e.g.
  P1+P5+P2, 12.7 s, masking/split and module-choice trials), else the known probe set closest to the logged duration.
  Trials are grouped per (layout, probe set); meta["pids"] names the set. Episode-link fields of the sealed schedule
  (LINK_FIELDS: episode, phase, round, rep, episode_modules, fault_carrier, episode_faults, choice_block) are copied
  into meta["labels"] when present. `load_trial_files` reads single trial files of a still-open session
  (PROVISIONAL; field-side module choice only).
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from .. import assembly as A
from . import fieldlog_config as FC
from . import schedule as SCH

GRID_HZ = 100.0
PARAMS_PATH = Path(__file__).resolve().parents[2] / "params" / "module_measured.yaml"
_SRC_NAME = {v: k for k, v in FC.SRC.items()}
_ASM_NAME = {1: "single", 2: "pair", 3: "L3", 4: "square4"}
_MODE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
_OOD = {5: "payload", 6: "low_mu"}
_SCHED_MODE = {"weak": 1, "coast": 2, "brake": 3, "roller_locked": 4}
_FID_OF_MODE = {1: "F1", 2: "F2", 3: "F3", 4: "F4"}
_FID_OF_OOD = {"payload": "F6", "low_mu": "F7"}
OPERATOR_MODES = (4,)                                    # roller_locked: physical, may not be echoed by firmware


class SessionRefused(PermissionError):
    pass


def verify(path):
    if str(FC.FIELDLOG) not in sys.path:
        sys.path.insert(0, str(FC.FIELDLOG))
    import verify_session  # noqa: E402
    return verify_session.verify(Path(path))


def _f(x):
    try:
        return float(x) if x != "" else math.nan
    except ValueError:
        return math.nan


def read_trial_file(p):
    """-> {src: (t_mono_ns int64 (k,), {col: float array (k,)})} keeping only the src's own filled columns."""
    rows = defaultdict(list)
    with open(p, newline="") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        ci = {c: i for i, c in enumerate(hdr)}
        for r in rd:
            rows[int(r[ci["src"]])].append(r)
    table = FC.source_table()
    out = {}
    for src, rr in rows.items():
        cols = table.get(src, (None, []))[1]
        cols = [c for c in cols if c in ci]
        tm = np.array([int(r[ci["t_mono_ns"]]) for r in rr], dtype=np.int64)
        out[src] = (tm, {c: np.array([_f(r[ci[c]]) for r in rr]) for c in cols})
    return out


def _zoh(t_rel, values, grid):
    """Zero-order hold of samples at t_rel (k,) onto grid (T,). values (k, C). Before the first sample: first sample."""
    if len(t_rel) == 0:
        return np.full((len(grid), values.shape[1]), np.nan)
    order = np.argsort(t_rel, kind="stable")
    t_rel, values = t_rel[order], values[order]
    idx = np.searchsorted(t_rel, grid, side="right") - 1
    return values[np.clip(idx, 0, len(t_rel) - 1)]


def _window(streams, t0, t1):
    """Slice every src to receipt times in [t0, t1] (ns)."""
    out = {}
    for src, (tm, cols) in streams.items():
        m = (tm >= t0) & (tm <= t1)
        out[src] = (tm[m], {c: v[m] for c, v in cols.items()})
    return out


def _segments(streams):
    """Probe trials from the TRIAL marker src: [(trial_num, asm_code, t_start_ns, t_end_ns)] (aborted ones dropped)."""
    if FC.SRC["TRIAL"] not in streams:
        return []
    tm, c = streams[FC.SRC["TRIAL"]]
    segs, open_ = [], None
    for k in np.argsort(tm, kind="stable"):
        ev, num = int(c["trial_event_code"][k]), int(c["trial_num"][k])
        if ev == FC.TRIAL_EVENT_CODES["start"]:
            open_ = (num, int(c["asm_type_code"][k]) if np.isfinite(c["asm_type_code"][k]) else 0, int(tm[k]))
        elif open_ is not None and num == open_[0]:
            if ev == FC.TRIAL_EVENT_CODES["end"]:
                segs.append((open_[0], open_[1], open_[2], int(tm[k])))
            open_ = None
    return segs


def _assembly(streams, asm_code, params):
    """Rebuild the Assembly from the logged module frames (modules in id order)."""
    side = float(params["module"]["side_x_m"])
    frames = {}
    for m in FC.MODULES:
        s = FC.SRC[f"ASM_{m}"]
        if s in streams and len(streams[s][0]):
            c = streams[s][1]
            frames[m] = (c[f"asm_x_m_{FC.mk(m)}"][-1], c[f"asm_y_m_{FC.mk(m)}"][-1], c[f"asm_yaw_deg_{FC.mk(m)}"][-1])
    if not frames:
        raise ValueError("no assembly frame logged for this trial")
    mods = sorted(frames)
    xs = np.array([frames[m][0] for m in mods])
    ys = np.array([frames[m][1] for m in mods])
    spec = [(m, int(round((frames[m][0] - xs.min()) / side)), int(round((frames[m][1] - ys.min()) / side)),
             int(round(frames[m][2] / 90.0)) % 4) for m in mods]
    asm = A.build(_ASM_NAME.get(asm_code, "custom"), spec, params)
    if not np.allclose(asm.module_pos, np.stack([xs, ys], 1), atol=0.02):
        raise ValueError("logged module frames are not on the module grid (check side length / frames)")
    return asm, spec, {m: frames[m] for m in mods}


def _labels(streams, asm, t_first_motion):
    n = asm.n
    mode = np.zeros(n, np.int64)
    g = np.ones(n, np.float32)
    ood = "none"
    for m in FC.MODULES:
        s = FC.SRC[f"FLT_{m}"]
        if s not in streams:
            continue
        tm, c = streams[s]
        for k in np.argsort(tm, kind="stable"):
            if tm[k] > t_first_motion:
                break
            mc = int(c[f"fs_mode_code_{FC.mk(m)}"][k])
            wc = c[f"fs_wheel_code_{FC.mk(m)}"][k]
            if mc in _OOD:
                ood = _OOD[mc]
                continue
            if mc not in _MODE or not np.isfinite(wc) or int(wc) >= 4:
                continue
            name = f"{m}:{FC.CORNERS[int(wc)]}"
            if name not in asm.wheel_corner:
                continue
            w = asm.wheel_corner.index(name)
            mode[w] = _MODE[mc]
            gv = c[f"fs_g_{FC.mk(m)}"][k]
            g[w] = float(gv) if (mc == 1 and np.isfinite(gv)) else 1.0
    return mode, g, ood


def infer_fault_id(mode, ood):
    """Echo-derived fault id: F0 healthy, F1-F4 one weak/coast/brake/roller wheel, F5 two (or more) faulted wheels,
    F6 payload, F7 low_mu; an OOD condition plus wheel faults gives e.g. "F6+F2" / "F7+F5"."""
    fw = [int(x) for x in np.asarray(mode) if int(x) != 0]
    wf = "F0" if not fw else (_FID_OF_MODE.get(fw[0], "F?") if len(fw) == 1 else "F5")
    if ood in _FID_OF_OOD:
        return _FID_OF_OOD[ood] if wf == "F0" else f"{_FID_OF_OOD[ood]}+{wf}"
    return wf


def _schedule_dict(schedule):
    if schedule is None or isinstance(schedule, dict):
        return schedule
    return SCH.load(schedule)                             # refuses an unsealed / altered schedule


def schedule_index(sched):
    """{(assembly_type, trial_number): trial dict} for a schedule dict."""
    return {(s["assembly"], int(t["trial"])): t for s, _, t in SCH.iter_trials(sched)}


def schedule_labels(trial, asm):
    """Sealed-schedule trial -> (mode (n,) i64, g (n,) f32, ood str, fault_id str)."""
    mode = np.zeros(asm.n, np.int64)
    g = np.ones(asm.n, np.float32)
    for f in trial["faults"]:
        if f["mode"] in _SCHED_MODE:
            w = asm.wheel_corner.index(f["wheel"])
            mode[w] = _SCHED_MODE[f["mode"]]
            if f["mode"] == "weak":
                g[w] = float(f["g"])
    return mode, g, trial.get("ood", "none"), str(trial["fault_id"])


def compare_echo(mode_s, g_s, ood_s, mode_e, g_e, ood_e):
    """Reasons (list[str]) why the logged echoes contradict the sealed schedule entry; [] = consistent."""
    why = []
    for w in range(len(mode_s)):
        s, e = int(mode_s[w]), int(mode_e[w])
        if s in OPERATOR_MODES and e == 0:
            continue                                      # physical condition, not echoed
        if s != e:
            why.append(f"wheel {w}: schedule mode {s}, echo {e}")
        elif s == 1 and abs(float(g_s[w]) - float(g_e[w])) > 1e-3:
            why.append(f"wheel {w}: schedule g {float(g_s[w]):.3f}, echo {float(g_e[w]):.3f}")
    if ood_e != "none" and ood_e != ood_s:
        why.append(f"ood: schedule {ood_s}, echo {ood_e}")
    return why


def _trial_arrays(streams, t0_ns, asm, frames, grid, imu_m, lidar_m, tag_offset):
    rel = lambda tm: (tm - t0_ns) * 1e-9                                        # noqa: E731
    T, n = len(grid), asm.n
    get = lambda src: streams.get(src, (np.zeros(0, np.int64), {}))             # noqa: E731

    tm, c = get(FC.SRC["CTL"])
    cmd = _zoh(rel(tm), np.stack([c.get(k, np.zeros(0)) for k in FC.CTL_COLS], 1) if len(tm) else np.zeros((0, 3)), grid)
    cmd = np.nan_to_num(cmd, nan=0.0)
    counts = np.zeros((T, n))
    duty = np.full((T, n), np.nan)
    current = np.full((T, n), np.nan)
    for w, name in enumerate(asm.wheel_corner):
        m, corner = name.split(":")
        tm, c = get(FC.SRC[f"WHL_{m}"])
        if len(tm):
            v = _zoh(rel(tm), np.stack([c[f"cnt_{corner}_{FC.mk(m)}"], c[f"duty_{corner}_{FC.mk(m)}"]], 1), grid)
            counts[:, w], duty[:, w] = v[:, 0], v[:, 1]
        else:
            raise ValueError(f"no wheel_state for module {m}")
        tm, c = get(FC.SRC[f"CUR_{m}"])
        if len(tm):
            current[:, w] = _zoh(rel(tm), c[f"cur_{corner}_{FC.mk(m)}"][:, None], grid)[:, 0]
    counts = np.nan_to_num(counts, nan=0.0)
    counts = counts - counts[:1]
    # IMU: module frame -> assembly frame
    tm, c = get(FC.SRC["IMU"])
    if len(tm):
        v = _zoh(rel(tm), np.stack([c["wz_radps"], c["ax_mps2"], c["ay_mps2"]], 1), grid)
        yaw = math.radians(frames[imu_m][2]) if imu_m in frames else 0.0
        cy, sy = math.cos(yaw), math.sin(yaw)
        gyro = v[:, 0]
        acc = np.stack([cy * v[:, 1] - sy * v[:, 2], sy * v[:, 1] + cy * v[:, 2]], 1)
    else:
        gyro, acc = np.full(T, np.nan), np.full((T, 2), np.nan)
    # rf2o: lidar-module-centre velocity in the module frame -> assembly-origin twist
    tm, c = get(FC.SRC["LODOM"])
    lidar = np.full((T, 3), np.nan)
    if len(tm) and lidar_m is not None and lidar_m in frames and asm.has_lidar():
        v = _zoh(rel(tm), np.stack([c["lo_vx_mps"], c["lo_vy_mps"], c["lo_wz_radps"]], 1), grid)
        x, y, yd = frames[lidar_m]
        yaw = math.radians(yd)
        cl, sl = math.cos(yaw), math.sin(yaw)
        vxa, vya = cl * v[:, 0] - sl * v[:, 1], sl * v[:, 0] + cl * v[:, 1]
        wz = v[:, 2]
        lidar = np.stack([vxa + wz * y, vya - wz * x, wz], 1)
    # truth: tag pose -> assembly origin pose (evaluator-only)
    tm, c = get(FC.SRC["TRUTH"])
    if len(tm):
        th_tag = 2 * np.arctan2(c["gt_qz"], c["gt_qw"])
        dx, dy, dth = tag_offset
        th = th_tag - dth
        X = c["gt_x_m"] - (np.cos(th) * dx - np.sin(th) * dy)
        Y = c["gt_y_m"] - (np.sin(th) * dx + np.cos(th) * dy)
        truth = _zoh(rel(tm), np.stack([X, Y, np.unwrap(th)], 1), grid)
    else:
        truth = np.full((T, 3), np.nan)
    return dict(cmd=cmd, counts=counts, duty=duty, gyro_z=gyro, acc=acc, lidar_twist=lidar, current=current, truth_pose=truth)


# WS-P2 episode-link fields (masking/split and module-choice schedules): copied per trial from the sealed schedule into
# meta["labels"] (lists; "" when the schedule has no such field). Evaluator-only, like every label.
LINK_FIELDS = ("episode", "phase", "round", "rep", "episode_modules", "fault_carrier", "episode_faults", "choice_block")


def load_session(path, allow_unverified=False, params=None, truth_latency_s=None, tag_offset=(0.0, 0.0, 0.0),
                 imu_module=None, lidar_module=None, verdict=None, schedule=None, on_mismatch="flag"):
    """FieldLog v2 capxfer session folder -> list[RawTrial] (one per assembly found in it).
    schedule: sealed schedule.json path or dict -> labels joined from it (see module docstring)."""
    if on_mismatch not in ("flag", "raise"):
        raise ValueError(on_mismatch)
    d = Path(path)
    meta_s = yaml.safe_load((d / "session.yaml").read_text())
    if meta_s.get("schema_version") != 2:
        raise ValueError(f"{d}: not a FieldLog v2 session")
    rep = verdict if verdict is not None else verify(d)
    v = rep["verdict"] if isinstance(rep, dict) else str(rep)
    if v != "FIELD_OK" and not allow_unverified:
        raise SessionRefused(f"{d.name}: verify_session verdict {v} (not FIELD_OK); pass allow_unverified=True to load anyway")
    source = "FIELD" if (meta_s.get("label") == "FIELD" and v in ("FIELD_OK", "FIELD_WITH_WARNINGS")) else "SIMULATION"
    files = sorted(p for p in d.glob(f"gnd_{FC.PORTION_TOKEN}_*.csv"))
    return _load_files(files, d.name, source, v, meta_s.get("label"), params=params, truth_latency_s=truth_latency_s,
                       tag_offset=tag_offset, imu_module=imu_module, lidar_module=lidar_module, schedule=schedule,
                       on_mismatch=on_mismatch)


def load_trial_files(files, schedule=None, params=None, truth_latency_s=None, tag_offset=(0.0, 0.0, 0.0),
                     imu_module=None, lidar_module=None, trials=None):
    """PROVISIONAL read of individual trial files of a FieldLog session that may still be OPEN (no session.yaml, no
    SHA256SUMS yet): for the field-side module choice (capxfer.field.choose_module) only. Never field data for the
    analysis: source "PROVISIONAL", verdict "UNVERIFIED_OPEN_SESSION". `trials`: keep only these schedule trial numbers.
    Read-only, like load_session."""
    files = [Path(f) for f in files]
    sess = files[0].parent.name if files else ""
    return _load_files(files, sess, "PROVISIONAL", "UNVERIFIED_OPEN_SESSION", None, params=params,
                       truth_latency_s=truth_latency_s, tag_offset=tag_offset, imu_module=imu_module,
                       lidar_module=lidar_module, schedule=schedule, on_mismatch="flag", keep=trials)


def _load_files(files, sess_name, source, v, label, params=None, truth_latency_s=None, tag_offset=(0.0, 0.0, 0.0),
                imu_module=None, lidar_module=None, schedule=None, on_mismatch="flag", keep=None):
    fault_sched = _schedule_dict(schedule)
    sidx = schedule_index(fault_sched) if fault_sched is not None else None
    keep = None if keep is None else {int(k) for k in keep}
    params = params or A.load_params(PARAMS_PATH)
    lat = float(truth_latency_s if truth_latency_s is not None else params["truth"]["apriltag_latency_s"])
    cpr = float(params["motor"]["counts_per_output_rev"])
    groups = {}
    for p in files:
        streams = read_trial_file(p)
        for num, asm_code, t0, t1 in _segments(streams):
            if keep is not None and int(num) not in keep:
                continue
            st = _window(streams, t0 - int(0.2e9), t1 + int(0.5e9))       # truth arrives up to its latency after the end marker
            asm, spec, frames = _assembly(_window(streams, t0 - int(0.2e9), t1), asm_code, params)
            tr = None
            if sidx is not None:
                tr = sidx.get((asm.name, int(num)))
                if tr is None:
                    raise ValueError(f"{sess_name}:{p.stem}: trial {num} of {asm.name} is not in the sealed schedule")
            sched = SCH.trial_schedule(asm, tr, duration_s=(t1 - t0) * 1e-9)   # WS-P2: full library or the probe subset
            pids = tuple(s_.pid for s_ in sched.segments[::2])
            key = (tuple(spec), pids)
            grid = np.arange(int(round(sched.T * GRID_HZ))) / GRID_HZ
            imu_d, lid_d = SCH.sensor_modules(asm)
            arr = _trial_arrays(st, t0, asm, frames, grid, imu_module or imu_d, lidar_module or lid_d, tag_offset)
            tm, c = st.get(FC.SRC["CTL"], (np.zeros(0, np.int64), {}))
            moving = [tm[k] for k in range(len(tm)) if any(abs(c[q][k]) > 1e-6 for q in FC.CTL_COLS)]
            t_motion = min(moving) if moving else t1
            mode, g, ood = _labels(_window(streams, t0 - int(0.2e9), t1), asm, t_motion)
            g_ = groups.setdefault(key, dict(asm=asm, sched=sched, grid=grid, arrs=[], ids=[], mode=[], g=[], ood=[],
                                             nums=[], mode_e=[], g_e=[], ood_e=[], fid=[], match=[], why=[],
                                             link={k: [] for k in LINK_FIELDS}))
            g_["arrs"].append(arr)
            g_["ids"].append(f"{sess_name}:{p.stem}:{num}")
            g_["nums"].append(int(num))
            g_["mode_e"].append(mode)
            g_["g_e"].append(g)
            g_["ood_e"].append(ood)
            for k in LINK_FIELDS:
                g_["link"][k].append((tr or {}).get(k, ""))
            if tr is not None:
                ms, gs, os_, fid = schedule_labels(tr, asm)
                why = compare_echo(ms, gs, os_, mode, g, ood)
                if why and on_mismatch == "raise":
                    raise ValueError(f"{sess_name}:{p.stem}:{num}: echoes contradict the sealed schedule: {why}")
                mode, g, ood = ms, gs, os_
            else:
                fid, why = infer_fault_id(mode, ood), []
            g_["mode"].append(mode)
            g_["g"].append(g)
            g_["ood"].append(ood)
            g_["fid"].append(fid)
            g_["match"].append(not why)
            g_["why"].append(why)
    out = []
    for key, g_ in groups.items():
        st = {k: np.stack([a[k] for a in g_["arrs"]]) for k in g_["arrs"][0]}
        oods = g_["ood"]
        asm_ = g_["asm"]
        mset = "+".join(sorted(m for m, *_ in asm_.modules))
        B = len(g_["ids"])
        raw = dict(t=torch.tensor(g_["grid"], dtype=torch.float32),
                   cmd=torch.tensor(st["cmd"], dtype=torch.float32),
                   counts=torch.tensor(np.round(st["counts"]), dtype=torch.long),
                   duty=torch.tensor(st["duty"], dtype=torch.float32),
                   gyro_z=torch.tensor(st["gyro_z"], dtype=torch.float32),
                   acc=torch.tensor(st["acc"], dtype=torch.float32),
                   lidar_twist=torch.tensor(st["lidar_twist"], dtype=torch.float32),
                   current=torch.tensor(st["current"], dtype=torch.float32),
                   truth_pose=torch.tensor(st["truth_pose"], dtype=torch.float32))
        labels = dict(mode=torch.tensor(np.stack(g_["mode"])), g=torch.tensor(np.stack(g_["g"])),
                      ood=oods[0] if len(set(oods)) == 1 else "mixed", ood_per_trial=oods,
                      schedule_trial=list(g_["nums"]), fault_id=list(g_["fid"]),
                      fault_id_source=["schedule" if fault_sched is not None else "echo"] * B,
                      n_faulted=[int((np.asarray(m) != 0).sum()) for m in g_["mode"]],
                      module_set=[mset] * B, assembly_type=asm_.name,
                      mode_echo=torch.tensor(np.stack(g_["mode_e"])),
                      g_echo=torch.tensor(np.stack(g_["g_e"])), ood_echo=list(g_["ood_e"]),
                      schedule_match=list(g_["match"]), schedule_mismatch=list(g_["why"]))
        if any(any(x != "" for x in vals) for vals in g_["link"].values()):
            labels.update({k: list(vals) for k, vals in g_["link"].items()})
        raw["meta"] = dict(assembly=g_["asm"], cpr=cpr, schedule=g_["sched"], source=source, trial_ids=g_["ids"],
                           wheel_radius=float(params["wheel"]["radius_m"]), truth_latency_s=lat, session=sess_name,
                           verdict=v, label=label, pids=key[1],
                           fault_schedule_sha256=(SCH.sha256_bytes(SCH.canonical(fault_sched))
                                                  if fault_sched is not None else None),
                           labels=labels)
        out.append(raw)
    return out

"""Returned FieldLog sessions -> the CSV inputs of the frozen PROTOCOL_V2 analysis (`capxfer.analysis.realv2`) (WS-I3F).

  python -m capxfer.field.builders mask    OUT/mask.csv  --schedule MASK_SHEETS/schedule.json SESSION_DIR...
  python -m capxfer.field.builders r3      OUT/r3.csv    --schedule ... --bundle BUNDLE [--splits CAL] SESSION_DIR...
  python -m capxfer.field.builders lambda  OUT/lambda_v2.json --r3 OUT/r3.csv --mask OUT/mask.csv --bundle BUNDLE
  python -m capxfer.field.builders carry   OUT/r3.csv --lambda OUT/lambda_v2.json          # fills carry_over_lb
  python -m capxfer.field.builders docks   OUT/dock_scores.csv --sheets DEMO_SHEETS [--arena arena.yaml] DEMO_SESSION...
  python -m capxfer.field.builders demo    OUT/demo.csv  --sheets DEMO_SHEETS --choices .../choices.csv \
         --choice-runs manual/choice_runs.csv [--dock-scores OUT/dock_scores.csv] DEMO_SESSION_DIR...
  python -m capxfer.field.builders manifest OUT/manifest.yaml --mask ... --pred ... --r3 ... --demo ... [--lambda ...]

Order on the field day (PROTOCOL_V2 §9 blinding): after MASK, `mask` + `r3 --splits CAL` + `lambda` ONLY (no TEST-unit
model output, no R1-R3 metric); lambda_v2.json -> freeze record -> DEMO. After all collection: `r3` (CAL + TEST),
`carry`, `docks`, `demo`, `manifest`, then `realv2.run_realv2(manifest)`.

Columns written (realv2 reads only its *_COLS + rho_<key>; the extra columns are provenance / diagnostics):
* mask.csv  realv2.MASK_COLS + rho_<14 keys> + z, source, session_status, start_utc, echo_ok, r3_split, episode, phase,
  realv2_seq, exclusion. `build` = the sealed round (square4, L3, pair, single_Mk); module/wheel/firmware from the
  schedule-joined labels (empty / "ok" for healthy). Exclusions (PROTOCOL_V2 §7): (a) session not FIELD_OK (with
  allow_simulation the SIMULATION label is reported in session_status but does not exclude; the provenance gate then
  fails, as it must), (b) fault echo != sealed schedule, (c) aborted (trials.csv outcome_code != 1; aborted markers are
  already dropped by the loader), (d) a robot-visible stream gap > 100 ms in the P1/P5/P2 holds. A leg whose AprilTag
  truth covers < 80 % of its hold window is NaN.
* r3.csv    realv2.R3_COLS + bb_q05 (black-box q05 of the current-body strafe), z_assembly (measured), scores_json.
  One row per (MASK assembly trial, module). score = sim_lb = the bundle's COMP lower bound (choose_module
  .certificate_bounds; realv2 uses `score > lambda_hat` for the certificate and `sim_lb >= tau` for the sim-only one).
  carry_over_lb = bb_q05 - carry_over_offset (from lambda_v2.json; empty until `carry`).
* demo.csv  realv2.DEMO_COLS (D15: L3, y/s/z_alone per module M2-M4) + episode_id, exclusion, y_source, dock_list,
  missing_docks. s_<M> and carry_over_lb are the values choose_module logged on the field (choices.csv, not
  recomputed); z_alone from the DEMO lone probes' truth; y_<M> from dock_scores.csv (automatic, tag-based,
  DOCKING_PROTOCOL §4) when given, else the operator's call in choice_runs.csv; NaN if M was not docked.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from ..probes import TARGET_KEYS
from . import fieldlog_config as FC
from . import loader as L
from . import schedule as SCH

KEYS = tuple(TARGET_KEYS)
STRAFE = (KEYS.index("P2+"), KEYS.index("P2-"))
MODE_NAME = {0: "ok", 1: "weak", 2: "coast", 3: "brake", 4: "roller_locked"}
ROBOT_PIDS = ("P1", "P5", "P2")
GAP_MAX_S = 0.100
TRUTH_COVER_MIN = 0.80
MASK_EXTRA = ("z", "source", "session_status", "start_utc", "echo_ok", "r3_split", "episode", "phase", "realv2_seq",
              "exclusion")


# ================================================================================================ helpers
def _sched(schedule):
    return schedule if (schedule is None or isinstance(schedule, dict)) else SCH.load(schedule)


def _read_csv(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(p, rows, cols):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in cols})
    return p


def _fmt(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return "inf" if v == math.inf else ("-inf" if v == -math.inf else repr(v))
    return v


def trials_index(session_dir, meta=None):
    """trials.csv of a FieldLog session -> {trial_id: row} (WS-KIT2: meta["trials_rows"] of a CSV-kit RawTrial)."""
    if meta is not None and "trials_rows" in meta:
        return meta["trials_rows"]
    p = Path(session_dir) / "trials.csv"
    return {r["trial_id"]: r for r in _read_csv(p)} if p.is_file() else {}


def _utc(ns):
    import datetime as dt
    try:
        return dt.datetime.fromtimestamp(int(ns) * 1e-9, dt.timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return ""


def subset_raw(raw, idx):
    """RawTrial batch restricted to trial positions idx (tensors and per-trial label lists)."""
    import torch
    idx = list(idx)
    out = {k: (v[idx] if torch.is_tensor(v) and v.dim() > 1 else v) for k, v in raw.items() if k != "meta"}
    out["t"] = raw["t"]
    meta = dict(raw["meta"])
    meta["trial_ids"] = [raw["meta"]["trial_ids"][i] for i in idx]
    B = len(raw["meta"]["trial_ids"])
    lab = {}
    for k, v in raw["meta"]["labels"].items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B:
            lab[k] = v[idx]
        elif isinstance(v, list) and len(v) == B:
            lab[k] = [v[i] for i in idx]
        else:
            lab[k] = v
    meta["labels"] = lab
    out["meta"] = meta
    return out


def truth_rho(raw):
    """Evaluator-only capability ratios (B, 14) from the truth pose (featurize.targets, same estimator as sim)."""
    from ..featurize import as_assembly, targets
    y, _ = targets(raw, as_assembly(raw["meta"]["assembly"]), float(raw["meta"]["truth_latency_s"]))
    return y.cpu().numpy().astype(float)


def file_checks(session_dir, trial_id, sched, asm, lat, truth_hz=30.0, meta=None):
    """(max gap s of the robot-visible streams inside the P1/P5/P2 holds, {key: truth coverage}) for one trial.
    WS-KIT2: a RawTrial from capxfer.field.csv_adapter carries these checks precomputed in meta["csv_checks"]."""
    if meta is not None and "csv_checks" in meta:
        return meta["csv_checks"].get(trial_id, (math.nan, {}))
    _, stem, num = trial_id.rsplit(":", 2)
    p = Path(session_dir) / f"{stem}.csv"
    if not p.is_file():
        return math.nan, {}
    st = L.read_trial_file(p)
    seg = [s for s in L._segments(st) if int(s[0]) == int(num)]
    if not seg:
        return math.nan, {}
    t0 = seg[0][2]
    srcs = [FC.SRC["CTL"], FC.SRC["IMU"]] + [FC.SRC[f"WHL_{m}"] for m, *_ in asm.modules]
    gap = 0.0
    cover = {}
    tt = st.get(FC.SRC["TRUTH"], (np.zeros(0, np.int64), {}))[0]
    for pid, leg, ta, tb, _ in sched.windows():
        key = f"{pid}{'+' if leg > 0 else '-'}"
        a, b = t0 + int(ta * 1e9), t0 + int(tb * 1e9)
        if pid in ROBOT_PIDS:
            for s in srcs:
                tm = np.sort(st.get(s, (np.zeros(0, np.int64), {}))[0])
                w = tm[(tm >= a) & (tm <= b)]
                edges = np.concatenate([[a], w, [b]])
                gap = max(gap, float(np.diff(edges).max()) * 1e-9)
        la, lb = a + int(lat * 1e9), b + int(lat * 1e9)
        n = int(((tt >= la) & (tt <= lb)).sum())
        cover[key] = n / max(1.0, (tb - ta) * truth_hz)
    return gap, cover


def load(session_dirs, schedule, allow_simulation=False, tag_offsets=None):
    """[(session_dir, raw)] for every RawTrial group of every session (loader.load_session, schedule-joined)."""
    sched = _sched(schedule)
    out = []
    for d in session_dirs:
        off = (tag_offsets or {}).get(Path(d).name, (0.0, 0.0, 0.0))
        for raw in L.load_session(d, allow_unverified=allow_simulation, schedule=sched, tag_offset=off):
            out.append((Path(d), raw))
    return out


def _fault_of(asm, mode_row):
    fw = [int(w) for w in np.flatnonzero(np.asarray(mode_row) != 0)]
    if not fw:
        return "", "", "ok"
    m, c = asm.wheel_corner[fw[0]].split(":")
    return m, c, MODE_NAME.get(int(np.asarray(mode_row)[fw[0]]), "?")


def _lab(labels, k, i, default=""):
    v = labels.get(k)
    return v[i] if isinstance(v, list) and i < len(v) else default


def _is_demo(labels):
    v = labels.get("choice_block")
    return bool(v) and any(x not in ("", None) for x in v)


def _strial(sidx, raw, i):
    """The sealed schedule entry of trial i of a RawTrial batch ({} without a schedule). WS-KIT2: CSV-kit RawTrials
    carry the run-sheet fields (r3_split, realv2_seq) in their labels and need no FieldLog schedule."""
    lab = raw["meta"]["labels"]
    if "sheet_r3_split" in lab:
        return {"r3_split": lab["sheet_r3_split"][i], "realv2_seq": lab["sheet_realv2_seq"][i]}
    if sidx is None:
        return {}
    num = raw["meta"]["labels"]["schedule_trial"][i]
    return sidx.get((raw["meta"]["assembly"].name, int(num)), {})


# ================================================================================================ mask.csv
def mask_rows(loaded, schedule=None, allow_simulation=False):
    from ..analysis import realv2 as RV
    sidx = L.schedule_index(_sched(schedule)) if schedule is not None else None
    rows = []
    for d, raw in loaded:
        meta, lab = raw["meta"], raw["meta"]["labels"]
        asm = meta["assembly"]
        rho = truth_rho(raw)
        tix = trials_index(d, meta)
        lat = float(meta["truth_latency_s"])
        for i, tid in enumerate(meta["trial_ids"]):
            m, w, fw = _fault_of(asm, lab["mode"][i].numpy())
            gap, cover = file_checks(d, tid, meta["schedule"], asm, lat, meta=meta)
            r = rho[i].copy()
            for j, k in enumerate(KEYS):
                if k in cover and cover[k] < TRUTH_COVER_MIN:
                    r[j] = math.nan
            stem = tid.rsplit(":", 2)[1]
            trow = tix.get(stem, {})
            why = []
            if meta["verdict"] != "FIELD_OK" and not (allow_simulation and meta["source"] == "SIMULATION"):
                why.append(f"a:session_{meta['verdict']}")
            if not lab["schedule_match"][i]:
                why.append("b:echo_mismatch")
            if trow and str(trow.get("outcome_code", "1")) != "1":
                why.append(f"c:outcome_{trow.get('outcome_code')}")
            if np.isfinite(gap) and gap > GAP_MAX_S:
                why.append(f"d:gap_{gap:.3f}s")
            st_ = _strial(sidx, raw, i)
            build = _lab(lab, "round", i) or (asm.name if asm.name != "single" else f"single_{asm.modules[0][0]}")
            row = dict(trial_id=tid, session=meta["session"], build=build, assembly_type=asm.name, module=m, wheel=w,
                       firmware=fw, excluded=bool(why), exclusion=";".join(why), z=float(np.nanmin(r[list(STRAFE)]))
                       if np.isfinite(r[list(STRAFE)]).all() else math.nan, source=meta["source"],
                       session_status=meta["verdict"], start_utc=_utc(trow.get("t_start_wall_ns")),
                       echo_ok=bool(lab["schedule_match"][i]), r3_split=st_.get("r3_split") or "NONE",
                       episode=_lab(lab, "episode", i), phase=_lab(lab, "phase", i), realv2_seq=st_.get("realv2_seq", ""))
            row.update({c: float(v) for c, v in zip(RV.RHO_COLS, r)})
            rows.append(row)
    return rows


def build_mask(out, session_dirs, schedule, allow_simulation=False, tag_offsets=None, loaded=None):
    from ..analysis import realv2 as RV
    loaded = loaded if loaded is not None else load(session_dirs, schedule, allow_simulation, tag_offsets)
    loaded = [(d, r) for d, r in loaded if not _is_demo(r["meta"]["labels"])]
    rows = mask_rows(loaded, schedule, allow_simulation)
    return _write_csv(out, rows, list(RV.MASK_COLS) + list(RV.RHO_COLS) + list(MASK_EXTRA))


# ================================================================================================ r3.csv
def r3_rows(loaded, schedule, bundle_dir, splits=("CAL", "TEST"), device="cpu", allow_simulation=False, posterior=None):
    """Certificate scores of the MASK ASSEMBLY trials whose sealed r3_split is in `splits` (PROTOCOL_V2 §9: CAL only
    before DEMO). Healthy-baseline trials (r3_split NONE) are never scored."""
    from . import choose_module as CM
    b = CM.load_bundle(bundle_dir)
    models = CM.Models(b, device)
    sched = _sched(schedule)
    sidx = L.schedule_index(sched) if sched is not None else None
    post = load_posterior_table(posterior) if posterior is not None else None
    rows = []
    for d, raw in loaded:
        meta, lab = raw["meta"], raw["meta"]["labels"]
        asm = meta["assembly"]
        if asm.name not in ("pair", "L3", "square4") or _is_demo(lab):
            continue
        sp = [_strial(sidx, raw, i).get("r3_split") or "NONE" for i in range(len(meta["trial_ids"]))]
        idx = [i for i, s in enumerate(sp) if s in splits]
        if not idx:
            continue
        sub = subset_raw(raw, idx)
        P, zs = models.predict(sub)
        rho = truth_rho(sub)
        mids = [m for m, *_ in asm.modules]
        excl = {r["trial_id"]: r for r in mask_rows([(d, sub)], sched, allow_simulation)}
        for k, i in enumerate(idx):
            tid = meta["trial_ids"][i]
            s = CM.certificate_bounds(P[k], asm.wheel_module, mids, b["_ws"], b["_bands"])
            m, w, _ = _fault_of(asm, lab["mode"][i].numpy())
            q05 = float(np.quantile(zs[k].reshape(-1), 0.05))
            za = float(np.min(rho[k][list(STRAFE)]))
            ps = posterior_summaries(P[k], asm.wheel_module, mids, post) if post is not None else {}
            for mid in mids:
                rows.append(dict(trial_id=tid, build=_lab(lab, "round", i) or asm.name, assembly_type=asm.name,
                                 unit_module=mid, fault_mw=f"{m}:{w}" if m else "none", r3_split=sp[i],
                                 score=float(s[mid]), sim_lb=float(s[mid]), carry_over_lb=math.nan, bb_q05=q05,
                                 z_assembly=za, excluded=excl[tid]["excluded"], exclusion=excl[tid]["exclusion"],
                                 scores_json=json.dumps({k_: float(v) for k_, v in s.items()}),
                                 bundle_sha256=b["_sha"],
                                 pm=ps.get(mid, (math.nan, math.nan))[0], u_q05=ps.get(mid, (math.nan, math.nan))[1],
                                 posterior_sha256=(post or {}).get("_sha", "")))
    return rows


R3_EXTRA = ("bb_q05", "z_assembly", "exclusion", "scores_json", "bundle_sha256", "pm", "u_q05", "posterior_sha256")


def build_r3(out, session_dirs, schedule, bundle_dir, splits=("CAL", "TEST"), allow_simulation=False, loaded=None,
             tag_offsets=None, device="cpu", posterior=None):
    from ..analysis import realv2 as RV
    loaded = loaded if loaded is not None else load(session_dirs, schedule, allow_simulation, tag_offsets)
    rows = r3_rows(loaded, schedule, bundle_dir, splits, device, allow_simulation, posterior)
    return _write_csv(out, rows, list(RV.R3_COLS) + list(R3_EXTRA))


# ================================================================================================ lambda_v2.json
def build_lambda(out, r3_csv, mask_csv, bundle_dir=None, status="FINAL", cfg=None, note=""):
    """lambda_v2.json from the CAL units only (realv2.cal_calibration; capxfer.lambda_v2/v1 via bundle.make_lambda_record)."""
    import pandas as pd
    from ..analysis import realv2 as RV
    from . import bundle as BU
    units = pd.read_csv(r3_csv)
    trials = pd.read_csv(mask_csv)
    c = RV.cal_calibration(units, trials, cfg)
    bsha = None
    if bundle_dir is not None:
        bsha = BU.sha256_file(Path(bundle_dir) / "bundle.yaml")
    shas = set(units.get("bundle_sha256", pd.Series(dtype=str)).dropna().astype(str))
    if bsha and shas and shas != {bsha}:
        raise ValueError(f"r3 scores come from bundle(s) {sorted(shas)}, not {bsha}")
    rec = BU.make_lambda_record(c["lambda_hat"], c["lambda_hat_alpha05"], c["n_null_cal"], c["carry_over_offset"],
                                c["n_carry_cal"], status=status, bundle_sha256=bsha or (sorted(shas)[0] if shas else None),
                                inputs={"r3_units": BU.sha256_file(r3_csv), "mask_trials": BU.sha256_file(mask_csv)},
                                alpha=c["alpha"], alpha_sens=c["alpha_sens"], tau=c["tau"], note=note)
    return BU.write_lambda(out, rec)


def apply_carry_over(r3_csv, lambda_path, out=None):
    """carry_over_lb = bb_q05 - carry_over_offset (offset +inf/null -> -inf: never go)."""
    from . import bundle as BU
    lam = BU.read_lambda(lambda_path)
    off = lam["carry_over_offset"]
    if off is None:
        raise ValueError("lambda file has no carry_over_offset (legacy format)")
    rows = _read_csv(r3_csv)
    for r in rows:
        q = float(r["bb_q05"]) if r.get("bb_q05") not in (None, "") else math.nan
        r["carry_over_lb"] = q - off if math.isfinite(off) else -math.inf
    cols = list(rows[0].keys()) if rows else []
    for r in rows:
        r["carry_over_lb"] = _fmt(float(r["carry_over_lb"]))
    return _write_csv(out or r3_csv, rows, cols)


# ================================================================================================ dock scoring
def dock_truth(session_dir, fieldlog_row, tag_offset=(0.0, 0.0, 0.0)):
    """Truth samples (t s, pose (k,3) arena frame, assembly origin) of the last complete FieldLog trial of run-sheet
    row `fieldlog_row`. EVALUATOR-ONLY."""
    tix = [r for r in trials_index(session_dir).values() if str(r.get("runsheet_row")) == str(fieldlog_row)
           and str(r.get("outcome_code", "1")) == "1"]
    if not tix:
        return None
    tid = sorted(tix, key=lambda r: int(r.get("t_start_mono_ns") or 0))[-1]["trial_id"]
    p = Path(session_dir) / f"{tid}.csv"
    if not p.is_file():
        return None
    st = L.read_trial_file(p)
    if FC.SRC["TRUTH"] not in st:
        return None
    tm, c = st[FC.SRC["TRUTH"]]
    o = np.argsort(tm)
    th_tag = 2 * np.arctan2(c["gt_qz"][o], c["gt_qw"][o])
    dx, dy, dth = tag_offset
    th = th_tag - dth
    X = c["gt_x_m"][o] - (np.cos(th) * dx - np.sin(th) * dy)
    Y = c["gt_y_m"][o] - (np.sin(th) * dx + np.cos(th) * dy)
    return (tm[o] - tm[o][0]) * 1e-9, np.stack([X, Y, np.unwrap(th)], 1), tid


def score_dock(t, pose, asm, scene, rest_speed=0.01, rest_s=1.0):
    """DOCKING_PROTOCOL §4 from truth samples: final pose = median over the last >= rest_s window at < rest_speed;
    contact = footprint (no inflation) outside the free floor + bay at any sample. Wall-tag motion is not logged in
    CSV today (CSV_SPEC §7): the operator's contact_seen is OR-ed in by build_demo."""
    from ..plan import dock as DK
    t, pose = np.asarray(t, float), np.asarray(pose, float)
    if len(t) < 3:
        return dict(label="invalid", success=False, pos_err=math.nan, head_err=math.nan, contact=False)
    sp = np.r_[np.hypot(np.diff(pose[:, 0]), np.diff(pose[:, 1])) / np.maximum(np.diff(t), 1e-6), 0.0]
    still = np.flatnonzero(sp >= rest_speed)
    k0 = (still[-1] + 1) if len(still) else 0
    if t[-1] - t[min(k0, len(t) - 1)] < rest_s - 1e-6:
        fin = None
    else:
        sel = (t >= t[-1] - rest_s) & (np.arange(len(t)) >= k0)
        fin = np.r_[np.median(pose[sel, :2], 0), math.atan2(np.median(np.sin(pose[sel, 2])), np.median(np.cos(pose[sel, 2])))]
    contact = DK.collides(scene, DK.footprint_points(asm, 0.02), pose, every=1)
    if fin is None:
        return dict(label="not_at_rest", success=False, pos_err=math.nan, head_err=math.nan, contact=bool(contact))
    g = np.asarray(scene.goal, float)
    pe = float(math.hypot(fin[0] - g[0], fin[1] - g[1]))
    he = float(abs(DK.wrap(fin[2] - g[2])))
    ok = (not contact) and pe <= DK.POS_TOL and he <= DK.HEAD_TOL
    return dict(label="success" if ok else ("collision" if contact else "pose_error"), success=bool(ok),
                pos_err=pe, head_err=he, contact=bool(contact))


def build_dock_scores(out, demo_session_dirs, sheets, arena=None, tag_offsets=None, params=None):
    """dock_scores.csv (episode, module, fieldlog_dir, fieldlog_row, trial_id, label, success, pos_err, head_err,
    contact_truth) for every DOCK row of choice_runsheet.csv found in the given DEMO sessions."""
    from . import dock_executor as DX
    sheet = _read_csv(Path(sheets) / "choice_runsheet.csv")
    rows = []
    for r in sheet:
        if r["step"] != "DOCK":
            continue
        asm, _ = DX.build_assembly(module=r["module"], params=params)
        ar = DX.load_arena(arena, asm) if arena else DX.default_arena(asm)
        scene = DX.arena_scene(asm, ar)
        for d in demo_session_dirs:
            tr = dock_truth(d, r["fieldlog_row"], (tag_offsets or {}).get(r["module"], (0.0, 0.0, 0.0)))
            if tr is None:
                continue
            s = score_dock(tr[0], tr[1], asm, scene)
            rows.append(dict(episode=r["episode"], module=r["module"], fieldlog_dir=Path(d).name,
                             fieldlog_row=r["fieldlog_row"], trial_id=tr[2], label=s["label"], success=s["success"],
                             pos_err=s["pos_err"], head_err=s["head_err"], contact_truth=s["contact"]))
    return _write_csv(out, rows, ("episode", "module", "fieldlog_dir", "fieldlog_row", "trial_id", "label", "success",
                                  "pos_err", "head_err", "contact_truth"))


# ================================================================================================ demo.csv
def parse_choices(choices_csv):
    """choices.csv (choose_module) -> {episode: {"arms": {arm: module or ""}, "s": {module: score}, "carry_over_lb",
    "docks": [modules], "n": rows}} (the LAST logged choice of an episode wins if it was re-run)."""
    out = {}
    for r in _read_csv(choices_csv):
        e = out.setdefault(r["episode"], {"arms": {}, "s": {}, "n": 0, "carry_over_lb": math.nan, "docks": []})
        e["n"] += 1
        step = r.get("step", "pre_split")
        if r["arm"] == "dock_list":                                       # finalize step (after the lone probes)
            e["docks"] = [m for m in r["module"].split("+") if m]
        elif r["arm"] == "-" and step == "pre_split":
            if "step" not in r:                                           # older one-step log: module = dock list
                e["docks"] = [m for m in r["module"].split("+") if m]
            for kv in r["scores"].split(";"):
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                if k.startswith("s_"):
                    e["s"][k[2:]] = float(v)
                elif k == "carry_over_lb":
                    e["carry_over_lb"] = float(v)
        else:
            e["arms"][r["arm"]] = r["module"]
    return out


def lone_truth_demo(loaded):
    """{(episode, module): z_alone} from the DEMO lone probes (phase split), truth strafe (mean over repeats)."""
    acc = {}
    for _, raw in loaded:
        lab = raw["meta"]["labels"]
        asm = raw["meta"]["assembly"]
        if asm.name != "single" or not _is_demo(lab):
            continue
        rho = truth_rho(raw)
        for i in range(len(raw["meta"]["trial_ids"])):
            if not lab["schedule_match"][i]:
                continue
            z = float(np.min(rho[i][list(STRAFE)]))
            acc.setdefault((_lab(lab, "episode", i), asm.modules[0][0]), []).append(z)
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def _dock_y(runs, auto, e, m):
    """(y or NaN if module m was not (validly) docked in episode e, source, choice_logged flag)."""
    rr = [r for r in runs if r["episode"] == e and r["module"] == m and r.get("invalid_reason", "none") in ("", "none")]
    if not rr:
        return math.nan, "", True
    r = sorted(rr, key=lambda r: int(r.get("run_global") or 0))[-1]
    logged = str(r.get("choice_logged", "1")) == "1"
    a = auto.get((e, m))
    if a is not None and a["label"] not in ("invalid", "not_at_rest"):
        return float(str(a["success"]) in ("1", "True") and str(r.get("contact_seen", "0")) != "1"), "tag", logged
    return float(r.get("operator_outcome") == "success"), "operator", logged


def build_demo(out, sheets, choices_csv, choice_runs_csv, demo_session_dirs=(), dock_scores=None,
               allow_simulation=False, loaded=None, tag_offsets=None):
    """demo.csv (realv2.DEMO_COLS, D15 / CHOICE.md: L3, deduplicated docks). episode = choice_block - 1 (the
    realv2_design.expand_schedule index); y_<M> NaN iff M was not docked (realv2 reports an arm whose module lacks a dock
    as an invalid episode); carry_over_lb and s_<M> are the values logged on the field (choices.csv)."""
    from ..analysis import realv2 as RV
    from ..analysis import realv2_design as RD
    sched = SCH.load(Path(sheets) / "schedule.json")
    if loaded is None:
        loaded = load(demo_session_dirs, sched, allow_simulation, tag_offsets)
    za = lone_truth_demo(loaded)
    ch = parse_choices(choices_csv)
    runs = _read_csv(choice_runs_csv)
    auto = {(r["episode"], r["module"]): r for r in _read_csv(dock_scores)} if dock_scores else {}
    mods = tuple(RD.DEMO_MODULES)
    rows = []
    for ep in sched["choice"]["episodes"]:
        e = ep["episode"]
        why, src = [], set()
        c = ch.get(e)
        if c is None or "certificate" not in c["arms"]:
            why.append("no_choice_logged")
        co = ep["coins"]
        near = list(ep.get("bb_nearest_goal") or [])
        row = dict(episode=int(ep["choice_block"]) - 1, faulted_modules="|".join(sorted(ep["faulted_modules"])),
                   carry_over_lb=(c or {}).get("carry_over_lb", math.nan),
                   nearest_goal_module=(near[0] if len(near) == 1 else co.get("carry_over_tie_coin", "")),
                   random_module=co["random_coin"], cert_tie_coin=co["cert_tie_coin"],
                   oracle_tie_coin=co["oracle_tie_coin"], carry_over_tie_coin=co.get("carry_over_tie_coin", ""))
        for m in mods:
            y, sr, logged = _dock_y(runs, auto, e, m)
            if sr:
                src.add(sr)
            if not logged:
                why.append(f"choice_not_logged_before_dock_{m}")
            row[f"y_{m}"] = y
            row[f"s_{m}"] = (c or {}).get("s", {}).get(m, math.nan)
            row[f"z_alone_{m}"] = za.get((e, m), math.nan)
            if (e, m) not in za:
                why.append(f"no_lone_truth_{m}")
        if c is not None and c["docks"]:
            miss = [m for m in c["docks"] if not np.isfinite(row[f"y_{m}"])]
            if miss:
                row["missing_docks"] = "+".join(miss)
        row.update(excluded=bool(why), exclusion=";".join(sorted(set(why))), episode_id=e,
                   y_source="+".join(sorted(src)), dock_list="+".join((c or {}).get("docks", [])))
        rows.append(row)
    return _write_csv(out, rows, list(RV.DEMO_COLS) + ["episode_id", "exclusion", "y_source", "dock_list", "missing_docks"])


# ================================================================================================ WS-KIT2 (D16)
# Uncalibrated structured posterior summaries (POSTERIOR RANKING arm, docs/ideas/H1B.md U-mean / U-q05) and the OFFLINE
# DEMO choices: the team runs its own robot code, docks all three L3 modules in every episode, and every arm's choice is
# computed afterwards from the pre-split assembly probe (frozen bundle) and the lone probes' truth (oracle), BEFORE any
# dock outcome is read (build_demo_choices writes demo_choices.csv; build_demo_v3 then joins the docks).
POST_SCHEMA = "capxfer.posterior_table/v1"
POST_NQ = 99                                   # quantile points per (module, mode vector)
POST_GRID = np.linspace(-0.3, 1.4, 341)        # as capxfer.ideas.split.evaluate.GRID
POST_LEVEL = 0.05                              # the uncalibrated default one-sided 95 % lower quantile (U-q05)
POST_K = 2


def posterior_table_from_library(library_npz, out, relative=True):
    """posterior.npz from a SPLIT single-module library (modes, keys, Z_<mid> (rows, plants, 14)): per (module,
    mode-vector key) the mean and POST_NQ quantile points of z = min(rho_P2+, rho_P2-) pooled over weak-gain corners
    and plants (theta ~ the library prior). v4 (D17): relative=True divides each plant's rho by that plant's healthy
    rho (key 0) first, so the table is on relative capability like tau. Hashed into the freeze record."""
    L = np.load(library_npz)
    keys = L["keys"]
    lev = (np.arange(POST_NQ) + 0.5) / POST_NQ
    mid_, key_, mean_, q_ = [], [], [], []
    for mid in sorted(k[2:] for k in L.files if k.startswith("Z_")):
        Z = L[f"Z_{mid}"].astype(float)
        if relative:                                     # v4 (D17): per plant, each leg / the plant's healthy rho
            h = Z[keys == 0].mean(0)                     # (plants, 14), healthy key 0
            Z = Z / h[None]
        for u in np.unique(keys):
            zu = np.minimum(Z[keys == u][..., STRAFE[0]], Z[keys == u][..., STRAFE[1]]).reshape(-1)
            zu = zu[np.isfinite(zu)]
            mid_.append(mid)
            key_.append(int(u))
            mean_.append(float(zu.mean()))
            q_.append(np.quantile(zu, lev))
    return _save_post(out, mid_, key_, mean_, q_, source=str(library_npz))


def synthetic_posterior_table(out, bundle_dir):
    """SYNTHETIC (tests / dry run only): a posterior table matching a synthetic bundle's toy bands (mean = band + 0.05,
    spread 0.03)."""
    b = np.load(Path(bundle_dir) / "bands.npz")
    lev = (np.arange(POST_NQ) + 0.5) / POST_NQ
    from scipy.stats import norm
    q = [float(v) + 0.05 + 0.03 * norm.ppf(lev) for v in b["lb"]]
    return _save_post(out, [str(m) for m in b["mid"]], [int(k) for k in b["key"]], [float(v) + 0.05 for v in b["lb"]],
                      q, source="SYNTHETIC")


def _save_post(out, mid, key, mean, q, source):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, schema=np.array(POST_SCHEMA), source=np.array(source), mid=np.array(mid), key=np.array(key),
             mean=np.array(mean, float), q=np.array(q, float), levels=(np.arange(POST_NQ) + 0.5) / POST_NQ)
    return out


def load_posterior_table(path):
    from . import bundle as BU
    T = np.load(path)
    if str(T["schema"]) != POST_SCHEMA:
        raise ValueError(f"{path}: not a {POST_SCHEMA} file")
    tab = {(str(m), int(k)): (float(mu), q) for m, k, mu, q in zip(T["mid"], T["key"], T["mean"], T["q"])}
    return {"tab": tab, "levels": T["levels"], "_sha": BU.sha256_file(path), "source": str(T["source"])}


def posterior_summaries(P, wheel_module, mids, post, K=POST_K, level=POST_LEVEL):
    """Per module of the assembly: (posterior mean of z, posterior `level` quantile of z) of the uncalibrated
    posterior-predictive mixture: mode vectors u on the module's 4 wheels with <= K faults, weights prod_j P_j(u_j)
    (the assembly posterior restricted to the module), components from the posterior table. No conformal step."""
    import itertools
    from . import choose_module as CM
    lev = post["levels"]
    out = {}
    for k, mid in enumerate(mids):
        w_idx = np.flatnonzero(np.asarray(wheel_module) == k)
        Pm = np.asarray(P)[w_idx]
        ws, means, F = [], [], []
        for mv in itertools.product(range(CM.N_MODES), repeat=len(w_idx)):
            if sum(1 for v in mv if v) > K:
                continue
            ent = post["tab"].get((mid, CM.mode_key(mv)))
            if ent is None:
                continue
            ws.append(float(np.prod([Pm[j, v] for j, v in enumerate(mv)])))
            means.append(ent[0])
            F.append(np.interp(POST_GRID, ent[1], lev, left=0.0, right=1.0))
        ws = np.asarray(ws)
        if not len(ws) or ws.sum() <= 0:
            out[mid] = (math.nan, math.nan)
            continue
        ws = ws / ws.sum()
        Fm = ws @ np.asarray(F)
        qi = int(np.clip(np.searchsorted(Fm, level), 0, len(POST_GRID) - 1))
        out[mid] = (float(ws @ np.asarray(means)), float(POST_GRID[qi]))
    return out


DEMO_CHOICE_COLS = (("episode", "episode_id", "faulted_modules", "carry_over_lb", "bb_q05", "nearest_goal_module",
                     "random_module", "cert_tie_coin", "oracle_tie_coin", "carry_over_tie_coin")
                    + tuple(f"{c}_{m}" for m in ("M2", "M3", "M4") for c in ("s", "pm", "uq05", "z_alone", "z_alone_abs"))
                    + ("capability",)
                    + ("choice_cert", "choice_carry_over", "choice_random", "choice_oracle", "choice_post_rank",
                       "choice_cert_ranking", "lambda_hat", "probe_exclusion", "assembly_trial_id",
                       "bundle_sha256", "lambda_sha256", "posterior_sha256", "computed_utc"))


def lone_truth_demo_relative(loaded, base):
    """v4 (SPINE D17): {(episode, module): (relative z, absolute z)} of the DEMO lone probes; relative = each leg's
    mean rho / that module's single_<M> healthy MASK baseline median (realv2.healthy_baseline), then min(P2+, P2-)."""
    acc = {}
    for _, raw in loaded:
        lab = raw["meta"]["labels"]
        asm = raw["meta"]["assembly"]
        if asm.name != "single" or not _is_demo(lab):
            continue
        rho = truth_rho(raw)
        for i in range(len(raw["meta"]["trial_ids"])):
            if lab["schedule_match"][i]:
                acc.setdefault((_lab(lab, "episode", i), asm.modules[0][0]), []).append(rho[i])
    out = {}
    for (e, m), rows in acc.items():
        mean = np.nanmean(np.stack(rows), 0)
        za = float(np.min(mean[list(STRAFE)]))
        b = f"single_{m}"
        if base is not None and b in base.index:
            h = base.loc[b].to_numpy(float)
            out[(e, m)] = (float(np.min((mean / h)[list(STRAFE)])), za)
        else:
            out[(e, m)] = (math.nan if base is not None else za, za)
    return out


def demo_choice_rows(loaded, sched_v2, bundle_dir, lambda_path, posterior=None, device="cpu", mask_csv=None):
    """OFFLINE DEMO choices (D16; PROTOCOL_V2 v3). loaded: [(dir, raw)] of the DEMO probe trials (csv_adapter labels:
    choice_block = episode + 1, phase assembly/split). sched_v2: realv2_design.expand_schedule(seed) (the coins).
    Reads robot-visible streams of the assembly probe (frozen bundle) and the lone probes' truth only; never a dock."""
    import datetime as dt
    from ..analysis import realv2 as RV
    from ..analysis import realv2_design as RD
    from . import bundle as BU
    from . import choose_module as CM
    from .predict import robot_view
    b = CM.load_bundle(bundle_dir)
    models = CM.Models(b, device)
    lam = BU.read_lambda(lambda_path)
    off = float(b["bb_offset"]) if lam["carry_over_offset"] is None else lam["carry_over_offset"]
    post = load_posterior_table(posterior) if posterior is not None else None
    import pandas as pd
    base = RV.healthy_baseline(pd.read_csv(mask_csv)) if mask_csv is not None else None
    za = lone_truth_demo_relative(loaded, base)            # v4: relative to the MASK singles' healthy baseline
    lone_bad = set()
    asm_probe = {}
    for _, raw in loaded:
        lab = raw["meta"]["labels"]
        if not _is_demo(lab):
            continue
        for i in range(len(raw["meta"]["trial_ids"])):
            e = int(lab["choice_block"][i]) - 1
            if raw["meta"]["assembly"].name == "single":
                if not lab["schedule_match"][i]:
                    lone_bad.add((e, raw["meta"]["assembly"].modules[0][0]))
            elif raw["meta"]["assembly"].name == RD.DEMO_ASSEMBLY:
                asm_probe.setdefault(e, []).append((raw, i))
    mods = tuple(RD.DEMO_MODULES)
    rows = []
    for ep in sched_v2["demo"]:
        e = int(ep["episode"])
        why = []
        row = dict(episode=e, episode_id=f"D{e + 1:02d}", faulted_modules="|".join(sorted(ep["faulted_modules"])),
                   nearest_goal_module=nearest_goal_module(), random_module=ep["random_module"],
                   cert_tie_coin=ep["cert_tie_coin"], oracle_tie_coin=ep["oracle_tie_coin"],
                   carry_over_tie_coin=ep["carry_over_tie_coin"], lambda_hat=lam["lambda_hat"],
                   bundle_sha256=b["_sha"], lambda_sha256=lam["sha256"], posterior_sha256=(post or {}).get("_sha", ""),
                   computed_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        pr = asm_probe.get(e, [])
        if not pr:
            why.append("no_assembly_probe")
        else:
            raw, i = pr[-1]                                  # the last complete assembly probe of the episode
            if not raw["meta"]["labels"]["schedule_match"][i]:
                why.append("assembly_echo_mismatch")
            sub = subset_raw(raw, [i])
            rv = robot_view(sub, keep_echo=bool(b.get("synthetic")))
            P, zs = models.predict(rv)
            asm = raw["meta"]["assembly"]
            mids = [m for m, *_ in asm.modules]
            s = CM.certificate_bounds(P[0], asm.wheel_module, mids, b["_ws"], b["_bands"])
            ps = posterior_summaries(P[0], asm.wheel_module, mids, post) if post is not None else {}
            q05 = float(np.quantile(zs[0].reshape(-1), 0.05))
            row.update(bb_q05=q05, carry_over_lb=(q05 - off) if math.isfinite(off) else -math.inf,
                       assembly_trial_id=raw["meta"]["trial_ids"][i])
            for m in mods:
                row[f"s_{m}"] = float(s.get(m, math.nan))
                row[f"pm_{m}"], row[f"uq05_{m}"] = ps.get(m, (math.nan, math.nan))
        row["capability"] = "relative" if base is not None else "absolute(no MASK baseline given)"
        for m in mods:
            row[f"z_alone_{m}"], row[f"z_alone_abs_{m}"] = za.get((f"D{e + 1:02d}", m), (math.nan, math.nan))
            if not np.isfinite(row[f"z_alone_{m}"]):
                why.append(f"no_lone_truth_{m}")
            if (e, m) in lone_bad:
                why.append(f"lone_echo_mismatch_{m}")
        if not why:
            ch = RV.demo_choices(row, lam["lambda_hat"], RV.DEFAULTS["tau"], mods)
            for a in ("cert", "carry_over", "random", "oracle", "post_rank", "cert_ranking"):
                row[f"choice_{a}"] = ch.get(a) or ("ABSTAIN" if a in ch else "")
        row["probe_exclusion"] = ";".join(sorted(set(why)))
        rows.append(row)
    return rows


def nearest_goal_module():
    """The black-box carry-over arm's module: largest assembly y in the L3 layout (M4; PROTOCOL_V2 R4)."""
    from .. import assembly as A
    lay = dict((m, j) for m, i, j, k in A.PHYSICAL_TYPES["L3"])
    top = max(lay.values())
    near = [m for m, j in lay.items() if j == top]
    return near[0]


def build_demo_choices(out, loaded, sched_v2, bundle_dir, lambda_path, posterior=None, device="cpu", mask_csv=None):
    """demo_choices.csv (+ its sha256 in <out>.sha256): computed and hashed BEFORE build_demo_v3 reads any dock.
    mask_csv: the MASK mask.csv (v4: the singles' healthy baseline that makes z_alone relative; required for R4)."""
    from . import bundle as BU
    rows = demo_choice_rows(loaded, sched_v2, bundle_dir, lambda_path, posterior, device, mask_csv)
    p = _write_csv(out, rows, DEMO_CHOICE_COLS)
    Path(str(p) + ".sha256").write_text(BU.sha256_file(p) + "  " + Path(p).name + "\n")
    return p


def build_demo_v3(out, choices_csv, dock_rows):
    """demo.csv (realv2.DEMO_COLS + pm_<M> + dock_design=all3) from demo_choices.csv and the dock outcomes.
    dock_rows: [{episode (0-based int), module, y (0/1 or NaN), invalid (bool), source}] (csv_adapter.dock_outcomes).
    An episode is excluded (blind to the arms) iff a probe check failed or any of its three docks is missing/invalid."""
    from ..analysis import realv2 as RV
    ch = _read_csv(choices_csv)
    d = {}
    for r in dock_rows:
        d[(int(r["episode"]), r["module"])] = r                # last row wins (repeats are listed after the original)
    mods = ("M2", "M3", "M4")
    rows = []
    for c in ch:
        e = int(c["episode"])
        why = [x for x in c.get("probe_exclusion", "").split(";") if x]
        row = {k: c.get(k, "") for k in RV.DEMO_COLS if k in c}
        row.update(episode=e, dock_design="all3", episode_id=c["episode_id"])
        src = set()
        for m in mods:
            for k in ("s", "pm", "z_alone", "z_alone_abs"):
                row[f"{k}_{m}"] = float(c[f"{k}_{m}"]) if c.get(f"{k}_{m}") not in (None, "") else math.nan
            r = d.get((e, m))
            if r is None:
                why.append(f"dock_missing_{m}")
                row[f"y_{m}"] = math.nan
            elif r["invalid"] or not np.isfinite(float(r["y"])):
                why.append(f"dock_invalid_{m}")
                row[f"y_{m}"] = math.nan
            else:
                row[f"y_{m}"] = float(r["y"])
                src.add(r.get("source", ""))
        row["carry_over_lb"] = float(c["carry_over_lb"]) if c.get("carry_over_lb") not in (None, "") else math.nan
        row.update(excluded=bool(why), exclusion=";".join(sorted(set(why))), y_source="+".join(sorted(src)))
        rows.append(row)
    cols = (list(RV.DEMO_COLS) + list(RV.DEMO_OPT_COLS) + [f"z_alone_abs_{m}" for m in mods]
            + ["episode_id", "exclusion", "y_source"])
    return _write_csv(out, rows, cols)


# ================================================================================================ manifest
def write_manifest(out, mask, pred, r3=None, demo=None, lambda_v2=None, freeze_record=None, seed_hex=None,
                   output_dir="out", extra=None):
    import yaml
    out = Path(out)
    rel = lambda p: None if p is None else str(Path(p).resolve())                             # noqa: E731
    m = {"output_dir": output_dir, "mask_trials": rel(mask), "predictions": rel(pred)}
    for k, v in (("r3_units", r3), ("demo", demo), ("lambda_v2", lambda_v2), ("freeze_record", freeze_record)):
        if v is not None:
            m[k] = rel(v)
    if seed_hex:
        m["seed_hex"] = seed_hex
    m.update(extra or {})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(m, sort_keys=False))
    return out


# ================================================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("mask", "r3"):
        p = sub.add_parser(name)
        p.add_argument("out")
        p.add_argument("sessions", nargs="+")
        p.add_argument("--schedule", required=True)
        p.add_argument("--allow-simulation", action="store_true")
        if name == "r3":
            p.add_argument("--bundle", required=True)
            p.add_argument("--splits", default="CAL,TEST")
    p = sub.add_parser("lambda")
    p.add_argument("out")
    p.add_argument("--r3", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--status", default="FINAL", choices=("FINAL", "SYNTHETIC"))
    p = sub.add_parser("carry")
    p.add_argument("r3")
    p.add_argument("--lambda", dest="lam", required=True)
    p = sub.add_parser("docks")
    p.add_argument("out")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--sheets", required=True)
    p.add_argument("--arena")
    p = sub.add_parser("demo")
    p.add_argument("out")
    p.add_argument("sessions", nargs="*")
    p.add_argument("--sheets", required=True)
    p.add_argument("--choices", required=True)
    p.add_argument("--choice-runs", required=True)
    p.add_argument("--dock-scores")
    p.add_argument("--allow-simulation", action="store_true")
    p = sub.add_parser("manifest")
    p.add_argument("out")
    for k in ("mask", "pred", "r3", "demo", "lambda", "freeze"):
        p.add_argument(f"--{k}")
    p.add_argument("--seed-hex")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    if a.cmd == "mask":
        print(build_mask(a.out, a.sessions, a.schedule, a.allow_simulation))
    elif a.cmd == "r3":
        print(build_r3(a.out, a.sessions, a.schedule, a.bundle, tuple(a.splits.split(",")), a.allow_simulation))
    elif a.cmd == "lambda":
        print(build_lambda(a.out, a.r3, a.mask, a.bundle, a.status))
    elif a.cmd == "carry":
        print(apply_carry_over(a.r3, a.lam))
    elif a.cmd == "docks":
        print(build_dock_scores(a.out, a.sessions, a.sheets, a.arena))
    elif a.cmd == "demo":
        print(build_demo(a.out, a.sheets, a.choices, a.choice_runs, a.sessions, a.dock_scores, a.allow_simulation))
    else:
        print(write_manifest(a.out, a.mask, a.pred, a.r3, a.demo, getattr(a, "lambda"), a.freeze, a.seed_hex))
    return 0


if __name__ == "__main__":
    sys.exit(main())

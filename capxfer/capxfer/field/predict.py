"""Prediction files of the PROTOCOL_V2 real-robot study (WS-PRED): the frozen simulator predictions (`pred.csv`, R1/R2 +
fidelity) and the robot-side certificate / black-box outputs per (trial, module) (`pred_units.csv`, R3/R4 cross-check).

  # at the FREEZE (no field data; sim at theta-hat, the same estimator as the field truth):
  python -m capxfer.field.predict conditions OUT/pred_full.csv [--schedule MASK_SHEETS/schedule.json] \
         [--theta theta_hat.json] [--bundle BUNDLE] [--qs] [--seed 20261121]
  # without --schedule both passive modes (coast, brake) are predicted for every (module, wheel) (the modes are sealed);
  # after the seed reveal (or on the robot laptop) the sealed modes select exactly one row per condition:
  python -m capxfer.field.predict select OUT/pred.csv --full OUT/pred_full.csv --schedule MASK_SHEETS/schedule.json
  # after collection (robot-visible streams only; NEVER reads truth_pose / AprilTag / evaluator labels):
  python -m capxfer.field.predict units OUT/pred_units.csv --bundle BUNDLE --lambda lambda_v2.json \
         --schedule MASK_SHEETS/schedule.json [--demo-sheets DEMO_SHEETS] [--allow-simulation] SESSION_DIR...

pred.csv (manifest key `predictions`; realv2.PRED_COLS first, one row per (build, module, wheel) incl. one healthy row per
build with module/wheel empty):
  build, module, wheel, L_pred, z_pred, k_star, assembly_type, firmware, L_qs, z_qs, k_star_qs, z_healthy_pred,
  n_sim_fault, n_sim_healthy, rho_<14 keys> (predicted condition-mean rho; the same rows are written to
  pred_rho.csv = manifest key `predicted_rho`, realv2 fidelity), theta_source, theta_sha256, params_sha256,
  bundle_sha256, lambda_sha256 (empty: the prediction file is frozen before lambda exists), code_sha256, git_commit,
  seed, created_utc.
  L_pred = max_k (mean healthy rho_k - mean faulted rho_k) over n_sim_fault faulted and n_sim_healthy healthy simulated
  trials of the build (PROTOCOL_V2 §4 R1: 3 + 6), each through capxfer.sensors.to_raw (sensor + AprilTag noise) and
  capxfer.featurize.targets (the field estimator); k_star its argmax leg; z_pred = min(rho_P2+, rho_P2-) of the faulted
  mean (assembly z for multi-module builds, z_alone for single_*). *_qs: the quasi-static solver
  (capxfer.ideas.i2.theory.qs_solve) at the same theta, noise-free (reported alongside, PROTOCOL_V2 R1).
pred_units.csv (not read by realv2; R3/R4 robot-side predictions, cross-checked against builders' r3.csv / choices.csv):
  part (MASK|DEMO), trial_id, session, build, assembly_type, unit_module, fault_mw (sealed design), r3_split, episode,
  score (= sim_lb, SPLIT (a) COMP lower bound), go_a10 / go_a05 (score > lambda_hat at alpha 0.10 / 0.05), go_sim_tau08 /
  go_sim_tau05 (sim_lb >= tau), bb_q05 (black-box current-body q05 of z), carry_over_lb (= bb_q05 - carry_over_offset),
  carry_go_tau08 / carry_go_tau05, cert_choice / cert_choice_a05 / sim_choice / rank_choice (per trial; DEMO ties by
  the sealed coins), carry_choice (DEMO only: the nearest-goal module iff carry_over_lb >= 0.8), source, session_status, bundle_sha256, lambda_sha256, lambda_status, code_sha256.
No analytic-Bayes column: PROTOCOL_V2 has no analytic-Bayes arm or metric.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..probes import TARGET_KEYS

REPO = Path(__file__).resolve().parents[2]
PARAMS = REPO / "params" / "module_measured.yaml"
KEYS = tuple(TARGET_KEYS)
RHO_COLS = tuple(f"rho_{k}" for k in KEYS)
STRAFE = (KEYS.index("P2+"), KEYS.index("P2-"))
PRED_COLS = ("build", "module", "wheel", "L_pred", "z_pred", "k_star")
PRED_EXTRA = ("assembly_type", "firmware", "L_qs", "z_qs", "k_star_qs", "z_healthy_pred", "n_sim_fault",
              "n_sim_healthy")
PROV_COLS = ("theta_source", "theta_sha256", "params_sha256", "bundle_sha256", "lambda_sha256", "code_sha256",
             "git_commit", "seed", "created_utc")
UNIT_COLS = ("part", "trial_id", "session", "build", "assembly_type", "unit_module", "fault_mw", "r3_split", "episode",
             "score", "sim_lb", "go_a10", "go_a05", "go_sim_tau08", "go_sim_tau05", "bb_q05", "carry_over_lb",
             "carry_go_tau08", "carry_go_tau05", "cert_choice", "cert_choice_a05", "sim_choice", "carry_choice",
             "rank_choice", "source", "session_status", "bundle_sha256", "lambda_sha256", "lambda_status",
             "code_sha256")
TAUS = (("tau08", 0.8), ("tau05", 0.5))
MODE_CODE = {"ok": 0, "weak": 1, "coast": 2, "brake": 3, "roller_locked": 4}
PASSIVE = ("coast", "brake")
N_FAULT, N_HEALTHY = 3, 6                       # PROTOCOL_V2 R1: "3 faulted and 6 healthy simulated trials"
CODE_FILES = ("capxfer/field/predict.py", "capxfer/field/choose_module.py", "capxfer/field/bundle.py",
              "capxfer/field/loader.py", "capxfer/field/schedule.py", "capxfer/sim.py", "capxfer/sensors.py",
              "capxfer/featurize.py", "capxfer/probes.py", "capxfer/assembly.py", "capxfer/sysid.py",
              "capxfer/ideas/i2/theory.py", "capxfer/analysis/realv2_design.py")

# robot-visible RawTrial streams (CONTRACTS C1 minus truth_pose) and the non-evaluator labels the unit pass may keep
ROBOT_STREAMS = ("t", "cmd", "counts", "duty", "gyro_z", "acc", "lidar_twist", "current")
SAFE_LABELS = ("schedule_trial", "schedule_match", "round", "episode", "phase", "choice_block", "module_set",
               "assembly_type")


# ================================================================================================ helpers
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256(repo=REPO, files=CODE_FILES):
    """sha256 over the sorted (path, sha256) list of the prediction code (as bundle.code_version)."""
    items = sorted((f, sha256_file(Path(repo) / f)) for f in files if (Path(repo) / f).is_file())
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def _git_commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:                                                                          # noqa: BLE001
        return None


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if math.isnan(v):
            return ""
        return "inf" if v == math.inf else ("-inf" if v == -math.inf else repr(v))
    return "" if v is None else v


def _write_csv(p, rows, cols):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in cols})
    return p


def _mw(m, w):
    return f"{m}:{w}" if m and w else "none"


# ================================================================================================ conditions
def layouts(schedule=None, params=None):
    """{build: (assembly_type, modules list, probe pids or None)}. From the sealed MASK schedule's session layouts when
    given (the field's module placement), else capxfer.assembly.PHYSICAL_TYPES (+ each module alone at the origin)."""
    from .. import assembly as A
    from ..analysis import realv2_design as D
    out = {}
    if schedule is not None:
        for s in schedule["sessions"]:
            mods = [tuple(m) for m in s["modules"]]
            for blk in s["blocks"]:
                for t in blk["trials"]:
                    b = (blk.get("condition") or {}).get("build") or t.get("round") or s["assembly"]
                    out.setdefault(b, (s["assembly"], mods, t.get("probes")))
    for t in ("pair", "L3", "square4"):
        out.setdefault(t, (t, list(A.PHYSICAL_TYPES[t]), None))
    for m in D.MODULES:
        out.setdefault(f"single_{m}", ("single", [(m, 0, 0, 0)], None))
    return out


def sealed_modes(schedule):
    """{'M:w': mode} from a sealed MASK schedule (its episodes' single injected passive fault)."""
    out = {}
    for e in schedule.get("episodes", []):
        f = e.get("faults") or []
        if len(f) == 1 and f[0]["mode"] in PASSIVE:
            out[f[0]["wheel"]] = f[0]["mode"]
    return out


def conditions(schedule=None, builds=None, modes=None):
    """[(build, module, wheel, firmware)] of the MASK design (realv2_design.mask_builds). The mode is the sealed one when
    `schedule` (or a {'M:w': mode} `modes` map) is given, else both passive modes."""
    from ..analysis import realv2_design as D
    if modes is None and schedule is not None:
        modes = sealed_modes(schedule)
    out = []
    for b, conds in D.mask_builds().items():
        if builds is not None and b not in builds:
            continue
        for m, w in conds:
            for fw in ((modes[f"{m}:{w}"],) if modes else PASSIVE):
                out.append((b, m, w, fw))
    return out


def theta_domain(B, params, theta=None):
    """Deterministic sim domain at theta-hat (sysid.nominal_domain; theta None -> nominal)."""
    from .. import sysid
    return sysid.nominal_domain(B, params, vals=theta)


def _qs_dom(params, theta):
    from .. import sysid
    from ..ideas.i2 import theory as TH
    v = {**sysid.nominal_values(params), **(theta or {})}
    return {k: float(v[k]) for k in TH.NOM if k in v}


def _sim_rho(asm, params, theta, modes, sched, seed, device, dt):
    """(B,14) field-estimator rho of B simulated trials (sim -> sensors.to_raw -> featurize.targets)."""
    import torch
    from .. import sensors, sim
    from ..featurize import targets
    B = modes.shape[0]
    dom = theta_domain(B, params, theta)
    g = torch.ones(modes.shape)
    R = sim.simulate(asm, params, dom, torch.as_tensor(modes, dtype=torch.long), g, schedule=sched, dt=dt, device=device,
                     seed=seed)
    R = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in R.items()}
    raw = sensors.to_raw(R, params, dom, np.random.default_rng(seed))
    y, _ = targets(raw, asm, float(params["truth"]["apriltag_latency_s"]))
    return y.cpu().numpy().astype(float)


def _qs_rho(asm, params, theta, modes):
    from ..ideas.i2 import theory as TH
    tw = TH.legs(asm)
    xi, _, _ = TH.qs_solve(asm, params, modes, np.ones(modes.shape), tw, dom=_qs_dom(params, theta))
    return TH.rho_of(xi, tw[None], asm.char_length())


def _loss(h, f):
    a = np.asarray(h, float) - np.asarray(f, float)
    if not np.isfinite(a).any():
        return math.nan, ""
    j = int(np.nanargmax(a))
    return float(a[j]), KEYS[j]


def _z(r):
    return float(min(r[STRAFE[0]], r[STRAFE[1]]))


def condition_predictions(conds, params=None, theta=None, schedule=None, n_fault=N_FAULT, n_healthy=N_HEALTHY,
                          seed=20261121, device="cpu", dt=0.005, qs=True, log=None):
    """Per-condition simulator predictions (rows of pred.csv, without provenance). One batched simulation per build:
    n_healthy healthy + n_fault per condition, at theta-hat, through the sensor/AprilTag model and the field estimator."""
    from .. import assembly as A
    from ..field import schedule as SCH
    params = params or A.load_params(PARAMS)
    lay = layouts(schedule, params)
    by_build = {}
    for c in conds:
        by_build.setdefault(c[0], []).append(c)
    rows = []
    for k, (b, cs) in enumerate(by_build.items()):
        at, mods, pids = lay[b]
        asm = A.build(at, mods, params)
        sched = SCH.trial_schedule(asm, {"probes": pids} if pids else None)
        n = asm.n
        M = np.zeros((n_healthy + n_fault * len(cs), n), np.int64)
        for i, (_, m, w, fw) in enumerate(cs):
            M[n_healthy + i * n_fault: n_healthy + (i + 1) * n_fault, asm.wheel_corner.index(f"{m}:{w}")] = MODE_CODE[fw]
        rho = _sim_rho(asm, params, theta, M, sched, seed + 1000 * k, device, dt)
        h = np.nanmean(rho[:n_healthy], 0)
        if qs:
            Mq = np.concatenate([np.zeros((1, n), np.int64), M[n_healthy::n_fault]], 0)
            rq = _qs_rho(asm, params, theta, Mq)
        base = dict(build=b, assembly_type=at, n_sim_healthy=n_healthy, n_sim_fault=n_fault)
        zh = _z(h)
        rows.append(dict(base, module="", wheel="", firmware="ok", L_pred=0.0, z_pred=zh, k_star="", z_healthy_pred=zh,
                         L_qs=0.0 if qs else math.nan, z_qs=_z(rq[0]) if qs else math.nan, k_star_qs="",
                         n_sim_fault=0, **dict(zip(RHO_COLS, h))))
        for i, (_, m, w, fw) in enumerate(cs):
            f = np.nanmean(rho[n_healthy + i * n_fault: n_healthy + (i + 1) * n_fault], 0)
            L, ks = _loss(h, f)
            r = dict(base, module=m, wheel=w, firmware=fw, L_pred=L, z_pred=_z(f), k_star=ks, z_healthy_pred=zh,
                     **dict(zip(RHO_COLS, f)))
            if qs:
                Lq, kq = _loss(rq[0], rq[1 + i])
                r.update(L_qs=Lq, z_qs=_z(rq[1 + i]), k_star_qs=kq)
            rows.append(r)
        if log:
            log(f"{b}: {len(cs)} conditions simulated ({M.shape[0]} trials)")
    return rows


def provenance(params_path=PARAMS, theta_path=None, bundle_dir=None, lambda_path=None, seed=None):
    return dict(theta_source=("sysid:" + Path(theta_path).name) if theta_path else "nominal",
                theta_sha256=sha256_file(theta_path) if theta_path else "",
                params_sha256=sha256_file(params_path),
                bundle_sha256=sha256_file(Path(bundle_dir) / "bundle.yaml") if bundle_dir else "",
                lambda_sha256=sha256_file(lambda_path) if lambda_path else "",
                code_sha256=code_sha256(), git_commit=_git_commit() or "", seed=seed if seed is not None else "",
                created_utc=_now())


def load_theta(path):
    """theta-hat JSON/YAML: a flat {key: value} dict or a sys-ID fit with a 'best' dict (sysid.fit_params)."""
    import yaml
    if path is None:
        return None
    d = yaml.safe_load(Path(path).read_text())
    d = d.get("best", d) if isinstance(d, dict) else d
    return {str(k): float(v) for k, v in d.items() if isinstance(v, (int, float))}


def write_pred(out, conds=None, schedule=None, params_path=PARAMS, theta_path=None, bundle_dir=None, seed=20261121,
               n_fault=N_FAULT, n_healthy=N_HEALTHY, qs=True, device="cpu", dt=0.005, rho_out="auto", log=None):
    """pred.csv (+ pred_rho.csv next to it). conds default: the full MASK design (sealed modes if `schedule`)."""
    from .. import assembly as A
    from . import schedule as SCH
    sched = SCH.load(schedule) if isinstance(schedule, (str, Path)) else schedule
    conds = conds if conds is not None else conditions(sched)
    params = A.load_params(params_path)
    rows = condition_predictions(conds, params, load_theta(theta_path), sched, n_fault, n_healthy, seed, device, dt,
                                 qs, log)
    prov = provenance(params_path, theta_path, bundle_dir, None, seed)
    for r in rows:
        r.update(prov)
    cols = list(PRED_COLS) + list(PRED_EXTRA) + list(RHO_COLS) + list(PROV_COLS)
    p = _write_csv(out, rows, cols)
    if rho_out == "auto":
        rho_out = Path(out).with_name(Path(out).stem + "_rho.csv")
    if rho_out:
        _write_csv(rho_out, rows, ["build", "module", "wheel", "firmware"] + list(RHO_COLS) + list(PROV_COLS))
    return p


def select_modes(full_csv, schedule, out, rho_out="auto"):
    """pred.csv from a both-modes prediction file: keep the healthy rows and, per (build, module, wheel), the row of the
    sealed mode. Refuses anything that leaves a predicted condition without exactly one row (realv2 merges on
    (build, mw)), e.g. a condition whose mode the schedule does not seal."""
    import pandas as pd
    from . import schedule as SCH
    sched = SCH.load(schedule) if isinstance(schedule, (str, Path)) else schedule
    modes = sealed_modes(sched)
    f = pd.read_csv(full_csv, keep_default_na=False, dtype={"module": str, "wheel": str, "k_star": str})
    keep = [(r.module == "" or modes.get(f"{r.module}:{r.wheel}") == r.firmware) for r in f.itertuples()]
    s = f[keep]
    dup = s.groupby(["build", "module", "wheel"]).size()
    if (dup != 1).any():
        raise ValueError(f"not one row per condition: {dup[dup != 1].to_dict()}")
    want = {(b, m, w) for b, m, w in zip(f["build"], f["module"], f["wheel"]) if m}
    have = {(b, m, w) for b, m, w in zip(s["build"], s["module"], s["wheel"]) if m}
    if want - have:
        raise ValueError(f"no prediction for {sorted(want - have)[:5]}")
    s.to_csv(out, index=False)
    if rho_out == "auto":
        rho_out = Path(out).with_name(Path(out).stem + "_rho.csv")
    if rho_out:
        s[["build", "module", "wheel", "firmware"] + list(RHO_COLS) + list(PROV_COLS)].to_csv(rho_out, index=False)
    return Path(out)


# ================================================================================================ units (robot-side)
def robot_view(raw, keep_echo=False):
    """A RawTrial restricted to what the robot sees: no truth_pose, no evaluator labels (mode, g, domain, ...). Only the
    synthetic dry-run stub posterior (`stub:echo`) may keep the logged firmware fault echoes (keep_echo)."""
    out = {k: raw[k] for k in ROBOT_STREAMS if k in raw}
    meta = {k: v for k, v in raw["meta"].items() if k != "labels"}
    lab = raw["meta"].get("labels") or {}
    meta["labels"] = {k: lab[k] for k in SAFE_LABELS if k in lab}
    if keep_echo and "mode_echo" in lab:
        meta["labels"]["mode_echo"] = lab["mode_echo"]
    out["meta"] = meta
    return out


def _choice(s, thr, tie, strict=True):
    """Largest s among s > thr (strict, certificate) or s >= thr (sim-only); exact tie -> sealed coin; None = abstain."""
    ok = [m for m in sorted(s) if (s[m] > thr if strict else s[m] >= thr)]
    if not ok:
        return None
    top = max(s[m] for m in ok)
    best = [m for m in ok if s[m] == top]
    return best[0] if len(best) == 1 else (tie if tie in best else best[0])


def _design_fault(trial):
    return "|".join(sorted(f["wheel"] for f in (trial or {}).get("faults", []) if f.get("mode") in MODE_CODE)) or "none"


def unit_rows(loaded, schedule, bundle_dir, lambda_path, demo_schedule=None, splits=("CAL", "TEST"), device="cpu"):
    """Robot-side per-(trial, module) predictions of the MASK assembly trials (r3_split in `splits`) and of the DEMO L3
    probes (when `demo_schedule` is given). Reads only robot_view(raw) and the sealed design."""
    from . import bundle as BU
    from . import choose_module as CM
    from . import loader as L
    from . import schedule as SCH
    b = CM.load_bundle(bundle_dir)
    models = CM.Models(b, device)
    lam = BU.read_lambda(lambda_path)
    off = float(b["bb_offset"]) if lam["carry_over_offset"] is None else lam["carry_over_offset"]
    sched = SCH.load(schedule) if isinstance(schedule, (str, Path)) else schedule
    sidx = L.schedule_index(sched) if sched is not None else {}
    dsched = SCH.load(demo_schedule) if isinstance(demo_schedule, (str, Path)) else demo_schedule
    didx = L.schedule_index(dsched) if dsched is not None else {}
    deps = {int(e["assembly_trial"]): e for e in ((dsched or {}).get("choice") or {}).get("episodes", [])}
    csha = code_sha256()
    rows = []
    for _, raw in loaded:
        meta = raw["meta"]
        asm = meta["assembly"]
        if asm.name not in ("pair", "L3", "square4"):
            continue
        lab = meta["labels"]
        nums = [int(x) for x in lab["schedule_trial"]]
        is_demo = any(x not in ("", None) for x in (lab.get("choice_block") or []))
        idx, info = [], []
        for i, num in enumerate(nums):
            if is_demo:
                ep = deps.get(num)
                if ep is None:
                    continue
                idx.append(i)
                info.append(("DEMO", didx.get((asm.name, num), {}), "NONE", ep))
            else:
                st = sidx.get((asm.name, num), {})
                sp = st.get("r3_split") or "NONE"
                if sp in splits:
                    idx.append(i)
                    info.append(("MASK", st, sp, None))
        if not idx:
            continue
        from .builders import subset_raw
        rv = robot_view(subset_raw(raw, idx), keep_echo=bool(b.get("synthetic")))
        P, zs = models.predict(rv)
        mids = [m for m, *_ in asm.modules]
        for k, i in enumerate(idx):
            part, st, sp, ep = info[k]
            s = CM.certificate_bounds(P[k], asm.wheel_module, mids, b["_ws"], b["_bands"])
            q05 = float(np.quantile(zs[k].reshape(-1), 0.05))
            co = q05 - off if math.isfinite(off) else -math.inf
            coins = (ep or {}).get("coins", {})
            near = list((ep or {}).get("bb_nearest_goal") or [])
            near_m = (near[0] if len(near) == 1 else coins.get("carry_over_tie_coin")) if near else None
            tie = coins.get("cert_tie_coin")
            trial = dict(cert_choice=_choice(s, lam["lambda_hat"], tie), cert_choice_a05=_choice(s, lam["lambda_hat_alpha05"], tie),
                         sim_choice=_choice(s, 0.8, tie, strict=False), rank_choice=_choice(s, -math.inf, tie),
                         carry_choice=(near_m if co >= 0.8 else None) if part == "DEMO" else None)
            for mid in mids:
                r = dict(part=part, trial_id=meta["trial_ids"][i], session=meta.get("session", ""),
                         build=st.get("round") or (lab.get("round") or [""] * len(nums))[i] or asm.name,
                         assembly_type=asm.name, unit_module=mid, fault_mw=_design_fault(st), r3_split=sp,
                         episode=(ep or {}).get("episode", st.get("episode", "")), score=float(s[mid]),
                         sim_lb=float(s[mid]), go_a10=bool(s[mid] > lam["lambda_hat"]),
                         go_a05=bool(s[mid] > lam["lambda_hat_alpha05"]), bb_q05=q05, carry_over_lb=co,
                         source=meta.get("source", ""), session_status=meta.get("verdict", ""),
                         bundle_sha256=b["_sha"], lambda_sha256=lam["sha256"], lambda_status=lam["status"],
                         code_sha256=csha, **{c: (v or "") for c, v in trial.items()})
                for name, tau in TAUS:
                    r[f"go_sim_{name}"] = bool(s[mid] >= tau)
                    r[f"carry_go_{name}"] = bool(co >= tau)
                rows.append(r)
    return rows


def build_units(out, session_dirs, schedule, bundle_dir, lambda_path, demo_schedule=None, demo_session_dirs=(),
                allow_simulation=False, splits=("CAL", "TEST"), device="cpu", loaded=None, tag_offsets=None):
    from . import builders as BL
    from . import schedule as SCH
    if loaded is None:
        loaded = BL.load(session_dirs, schedule, allow_simulation, tag_offsets) if session_dirs else []
        if demo_session_dirs:
            loaded += BL.load(demo_session_dirs, SCH.load(demo_schedule), allow_simulation, tag_offsets)
    rows = unit_rows(loaded, schedule, bundle_dir, lambda_path, demo_schedule, splits, device)
    return _write_csv(out, rows, UNIT_COLS)


def check_units(units_csv, r3_csv=None, choices_csv=None, tol=1e-6):
    """Consistency of pred_units.csv with builders' r3.csv (score / sim_lb per (trial, module); carry_over_lb if filled)
    and with the field-logged choices.csv (s_<M> and the certificate arm per DEMO episode). -> report dict."""
    import pandas as pd
    u = pd.read_csv(units_csv, keep_default_na=False)
    rep = {"n_units": int(len(u))}
    if r3_csv is not None:
        r = pd.read_csv(r3_csv)
        m = u[u["part"] == "MASK"].merge(r, on=["trial_id", "unit_module"], suffixes=("", "_r3"))
        d = (m["score"].astype(float) - m["score_r3"].astype(float)).abs().replace(np.nan, 0)
        rep["r3"] = dict(n_matched=int(len(m)), n_r3=int(len(r)), max_abs_score_diff=float(d.max()) if len(d) else 0.0,
                         ok=bool(len(m) == len(r) and (d <= tol).all()))
        if "carry_over_lb_r3" in m and m["carry_over_lb_r3"].notna().any():
            a, c = m["carry_over_lb"].astype(float).to_numpy(), m["carry_over_lb_r3"].astype(float).to_numpy()
            with np.errstate(invalid="ignore"):
                same = (a == c) | (np.abs(a - c) <= tol)
            rep["r3"]["carry_over_ok"] = bool(same.all())
    if choices_csv is not None:
        from .builders import parse_choices
        ch = parse_choices(choices_csv)
        dm = u[u["part"] == "DEMO"]
        bad = []
        for e, g in dm.groupby("episode"):
            c = ch.get(e)
            if c is None:
                continue
            for _, r in g.iterrows():
                if abs(float(r["score"]) - c["s"].get(r["unit_module"], math.nan)) > 1e-3:      # logged at 4 decimals
                    bad.append((e, r["unit_module"]))
            if (g["cert_choice"].iloc[0] or "") != (c["arms"].get("certificate") or ""):
                bad.append((e, "certificate"))
        rep["choices"] = dict(n_episodes=int(dm["episode"].nunique()), mismatches=bad, ok=not bad)
    return rep


# ================================================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("conditions")
    p.add_argument("out")
    p.add_argument("--schedule")
    p.add_argument("--params", default=str(PARAMS))
    p.add_argument("--theta")
    p.add_argument("--bundle")
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--no-qs", action="store_true")
    p.add_argument("--builds", help="comma list (default: all MASK builds)")
    p = sub.add_parser("select")
    p.add_argument("out")
    p.add_argument("--full", required=True)
    p.add_argument("--schedule", required=True)
    p = sub.add_parser("units")
    p.add_argument("out")
    p.add_argument("sessions", nargs="*")
    p.add_argument("--bundle", required=True)
    p.add_argument("--lambda", dest="lam", required=True)
    p.add_argument("--schedule", required=True)
    p.add_argument("--demo-sheets")
    p.add_argument("--demo-sessions", nargs="*", default=())
    p.add_argument("--splits", default="CAL,TEST")
    p.add_argument("--allow-simulation", action="store_true")
    p = sub.add_parser("check")
    p.add_argument("units")
    p.add_argument("--r3")
    p.add_argument("--choices")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    if a.cmd == "conditions":
        from . import schedule as SCH
        sched = SCH.load(a.schedule) if a.schedule else None
        builds = set(a.builds.split(",")) if a.builds else None
        print(write_pred(a.out, conditions(sched, builds), sched, a.params, a.theta, a.bundle, a.seed, qs=not a.no_qs,
                         log=print))
    elif a.cmd == "select":
        print(select_modes(a.full, a.schedule, a.out))
    elif a.cmd == "units":
        ds = str(Path(a.demo_sheets) / "schedule.json") if a.demo_sheets else None
        print(build_units(a.out, a.sessions, a.schedule, a.bundle, a.lam, ds, a.demo_sessions, a.allow_simulation,
                          tuple(a.splits.split(","))))
    else:
        print(json.dumps(check_units(a.units, a.r3, a.choices), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

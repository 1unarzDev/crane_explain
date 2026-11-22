"""Adapters between workstream interfaces (each one is a documented mismatch in docs/coordination/I.md).

Nothing here changes another package; every function only reshapes data between published APIs.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

GPU_GB = 3.0                     # WS-I budget (docs/CONTRACTS.md rule 4 as assigned to this workstream)
_GPU_TOTAL_GB = None


# ----------------------------------------------------------------------------------------------- GPU budget
def gpu_budget(gb=GPU_GB):
    """Cap this process's CUDA allocator at `gb` GB. Returns the device string.
    M1: Predictor.load() re-applies the checkpoint's own mem_frac (0.25 of 16 GB = 4 GB), so call this again after
    every load (load_predictor does)."""
    global _GPU_TOTAL_GB
    if not torch.cuda.is_available() or os.environ.get("CAPXFER_I_CPU") == "1":
        return "cpu"
    if _GPU_TOTAL_GB is None:
        _GPU_TOTAL_GB = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb / _GPU_TOTAL_GB))
    return "cuda"


def load_predictor(path, device=None):
    from capxfer.learn.base import Predictor
    dev = device or gpu_budget()
    m = Predictor.load(str(path), device=dev)
    gpu_budget()                                                     # M1: undo the checkpoint's mem_frac
    return m


def model_path(model_dir, method, suffix=""):
    """M2: WS-C's model dir holds two generations (`<m>.pt` = stub data, `<m>_A.pt` = final WS-A data). The frozen
    model dir should hold exactly one file per method; until then the caller passes the suffix explicitly."""
    p = Path(model_dir) / f"{method}{suffix}.pt"
    if not p.is_file():
        raise FileNotFoundError(f"model {method} not found as {p}")
    return p


# ----------------------------------------------------------------------------------------------- RawTrial helpers
def subset_raw(raw, idx):
    """Select trials `idx` of a RawTrial (one assembly). M3: sysid._check_healthy_single needs a RawTrial whose labels
    are all healthy and whose labels['ood'] == 'none', but the loader returns one RawTrial per assembly with all faults
    mixed (ood 'mixed'); this slices tensors, trial_ids and labels consistently."""
    idx = [int(i) for i in idx]
    B = raw["duty"].shape[0]
    out = {}
    for k, v in raw.items():
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[0] == B:
            out[k] = v[idx]
        elif torch.is_tensor(v) and v.dim() == 1 and k != "t" and v.shape[0] == B:
            out[k] = v[idx]
        else:
            out[k] = v
    meta = dict(raw["meta"])
    meta["trial_ids"] = [raw["meta"]["trial_ids"][i] for i in idx]
    lab = dict(meta.get("labels") or {})
    for k in ("mode", "g"):
        if k in lab:
            lab[k] = lab[k][idx]
    if "ood_per_trial" in lab:
        o = [lab["ood_per_trial"][i] for i in idx]
        lab["ood_per_trial"] = o
        lab["ood"] = o[0] if len(set(o)) == 1 else "mixed"
    meta["labels"] = lab
    out["meta"] = meta
    if "labels" in raw:
        out["labels"] = lab
    return out


def concat_raws(raws):
    """Concatenate RawTrials of the SAME assembly (e.g. the same assembly logged in several sessions)."""
    if len(raws) == 1:
        return raws[0]
    out = {}
    for k, v in raws[0].items():
        if k in ("meta", "labels", "t"):
            continue
        out[k] = torch.cat([r[k] for r in raws], 0)
    out["t"] = raws[0]["t"]
    meta = dict(raws[0]["meta"])
    meta["trial_ids"] = [t for r in raws for t in r["meta"]["trial_ids"]]
    meta["session"] = [r["meta"]["session"] for r in raws for _ in r["meta"]["trial_ids"]]
    labs = [r["meta"]["labels"] for r in raws]
    o = [x for lb in labs for x in lb["ood_per_trial"]]
    meta["labels"] = dict(mode=torch.cat([lb["mode"] for lb in labs]), g=torch.cat([lb["g"] for lb in labs]),
                          ood=o[0] if len(set(o)) == 1 else "mixed", ood_per_trial=o)
    out["meta"] = meta
    return out


def trial_sessions(raw):
    """Per-trial session names (the loader stores one session name per RawTrial; concat_raws stores a list)."""
    s = raw["meta"].get("session")
    n = len(raw["meta"]["trial_ids"])
    return list(s) if isinstance(s, (list, tuple)) else [str(s)] * n


def trial_num(trial_id):
    """Loader trial ids are '<session>:<fieldlog trial file>:<schedule trial number>'."""
    return int(str(trial_id).rsplit(":", 1)[1])


def asm_key(asm):
    return tuple((m, int(c), int(r), int(y)) for m, c, r, y in asm.modules)


# ----------------------------------------------------------------------------------------------- schedule join
def schedule_index(sched):
    """{(assembly_name, trial_number): trial dict} from the sealed schedule."""
    from capxfer.field import schedule as SCH
    # D3c: each trial also carries its block's scheduled field session (the cluster unit of the bootstrap)
    return {(s["assembly"], int(t["trial"])): (dict(t, field_session=b["field_session"]) if "field_session" in b else t)
            for s, b, t in SCH.iter_trials(sched)}


def trial_meta(raw, sidx):
    """M4: PredTables need per-trial `fault` (F0..F7) and `module_set`, which the loader does not emit (it recovers
    mode/g/ood from the fault_state echoes only; F-ids are not logged). Join on the schedule trial number and
    cross-check the logged labels against the schedule. Returns dict of per-trial arrays for analysis.table.STR_KEYS
    plus `schedule_trial` and `n_faulted`."""
    asm = raw["meta"]["assembly"]
    name = asm.name
    ids = list(raw["meta"]["trial_ids"])
    lab = raw["meta"]["labels"]
    oods = list(lab.get("ood_per_trial") or [lab.get("ood", "none")] * len(ids))
    mode = lab["mode"].numpy()
    faults, conds, fsess, mism = [], [], [], []
    for i, tid in enumerate(ids):
        tr = sidx.get((name, trial_num(tid)))
        if tr is None:
            raise ValueError(f"trial {tid}: no schedule entry for ({name}, {trial_num(tid)})")
        faults.append(tr["fault_id"])
        conds.append(str(tr.get("condition", tr["fault_id"])))
        fsess.append(tr.get("field_session"))
        if tr["ood"] != oods[i]:
            mism.append((tid, "ood", tr["ood"], oods[i]))
        # wheel faults the schedule implies: every scheduled fault entry on a real wheel (F6w/F7w trials carry
        # firmware wheel fault(s) on top of the OOD condition; D3: two passive faults)
        fid = tr["fault_id"]
        want = ({"F0": 0, "F1": 1, "F2": 1, "F3": 1, "F4": 1, "F5": 2}[fid] if fid not in ("F6", "F7")
                else sum(1 for f in tr["faults"] if f.get("wheel", "none") != "none"))
        if int((mode[i] != 0).sum()) != want:
            mism.append((tid, "n_faulted", want, int((mode[i] != 0).sum())))
    if mism:
        raise ValueError(f"logged labels disagree with the sealed schedule: {mism[:5]} ({len(mism)} total)")
    mods = "+".join(sorted(m[0] for m in asm.modules))
    N = len(ids)
    # D3c: the cluster unit `session` is the sealed schedule's field session (a FieldLog session holds blocks of all
    # types; one scheduled session may span several FieldLog folders after a restart). Without one: the folder name.
    dirs = trial_sessions(raw)
    sess = [f if f is not None else d_ for f, d_ in zip(fsess, dirs)]
    return dict(trial_id=np.array(ids), assembly_type=np.full(N, name), session=np.array(sess),
                fieldlog_session=np.array(dirs),
                module_set=np.full(N, mods), fault=np.array(faults), condition=np.array(conds), ood=np.array(oods),
                source=np.full(N, raw["meta"]["source"]),
                schedule_trial=np.array([trial_num(t) for t in ids]))


def healthy_single_idx(raw, meta):
    """Indices of healthy (F0, all modes OK, in-model) trials of a single-module RawTrial."""
    asm = raw["meta"]["assembly"]
    if len(asm.modules) != 1:
        return []
    mode = raw["meta"]["labels"]["mode"].numpy()
    return [i for i in range(len(meta["fault"]))
            if meta["fault"][i] == "F0" and meta["ood"][i] == "none" and not mode[i].any()]


# ----------------------------------------------------------------------------------------------- predictions
def pred_numpy(pred, q_levels):
    """C3 Pred -> numpy dict with explicit q_levels (M5: Pred has no q_levels key; table.from_preds silently defaults
    to (0.05, 0.5, 0.95), which is correct for WS-C's cfg but is now passed explicitly)."""
    out = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in pred.items()}
    out["q_levels"] = np.asarray(q_levels, float)
    return out


def finite_or_nan(x):
    x = np.asarray(x, float)
    return np.where(np.isfinite(x), x, math.nan)

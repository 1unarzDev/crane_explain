"""PredTable: the flat, padded on-disk format the frozen analysis reads (one .npz per method x rung).

Producers (integrator / WS-C eval script) call `from_preds(...)` with C3 Preds and C2 Feat labels for every trial,
then `save(path)`. Rows are trials; wheel arrays are padded to a common width W with `wheel_mask` (FIX M10: W is
arbitrary = max(n over the batches, `pad_to`), so 20/24-wheel sim families work; `MAX_WHEELS` is only the default
minimum width, kept so 4..16-wheel tables keep their historical shape).

Required keys
  trial_id (N,) str, assembly_type (N,) str in {single,pair,L3,square4} (sim: shape family id),
  session (N,) str, module_set (N,) str (e.g. "M1+M3"), fault (N,) str (F0..F7), ood (N,) str (none|payload|low_mu),
  source (N,) str (FIELD|SIMULATION), y (N,14) f8, y_q (N,14,K) f8, q_levels (K,), ppc_score (N,) f8
Optional keys
  mode_prob (N,W,5), wheel_mask (N,W) bool, mode_true (N,W) i8 (-1 = pad), n_wheels (N,)
  condition (N,) str: the sealed-schedule condition (F0..F7, F6w/F7w = OOD + firmware wheel fault; D3)
  fieldlog_session (N,) str: the FieldLog folder (`session` = the scheduled field session, the cluster unit; D3c)
Tables of different W are combined with `concat` (pads to the wider one); `pad_wheels(t, W)` re-pads one table;
`crop_wheels(t, n)` cuts a single-assembly subset back to its own n (for consumers that want (B,n,5)).
"""
from __future__ import annotations

import numpy as np

from ..conformal._common import to_np

MAX_WHEELS = 16          # default MINIMUM pad width (not a cap): wider assemblies widen the table
WHEEL_KEYS = ("mode_prob", "wheel_mask", "mode_true")
_PAD = {"mode_prob": np.nan, "wheel_mask": False, "mode_true": -1}
STR_KEYS = ("trial_id", "assembly_type", "session", "module_set", "fault", "ood", "source")
REQ = STR_KEYS + ("y", "y_q", "q_levels", "ppc_score")
OPT_STR_KEYS = ("condition", "fieldlog_session")


def from_preds(preds, ys, labels, meta, pad_to=MAX_WHEELS):
    """preds: list of Pred (one per assembly batch); ys: list of (B,14); labels: list of {"mode": (B,n)};
    meta: list of dicts of per-trial string arrays for STR_KEYS. -> dict (a PredTable).
    Wheel arrays are padded to W = max(pad_to, max n); any n works (pad_to=None: exactly max n)."""
    T = {k: [] for k in REQ + ("mode_prob", "wheel_mask", "mode_true", "n_wheels")}
    opt = [k for k in OPT_STR_KEYS if meta and all(k in m for m in meta)]
    for k in opt:
        T[k] = []
    W = max([int(to_np(lab["mode"]).shape[1]) for lab in labels] + [int(pad_to or 0)])
    for p, y, lab, m in zip(preds, ys, labels, meta):
        B = to_np(y).shape[0]
        for k in STR_KEYS + tuple(opt):
            T[k].append(np.asarray(m[k]).astype(str).reshape(B))
        T["y"].append(to_np(y, float))
        T["y_q"].append(to_np(p["y_q"], float))
        T["ppc_score"].append(to_np(p["ppc_score"], float))
        mp = to_np(p.get("mode_prob"))
        n = to_np(lab["mode"]).shape[1]
        T["n_wheels"].append(np.full(B, n))
        pad = np.zeros((B, W, 5)) + np.nan
        mask = np.zeros((B, W), bool)
        mt = np.full((B, W), -1, np.int64)
        mask[:, :n] = True
        mt[:, :n] = to_np(lab["mode"]).astype(np.int64)
        if mp is not None:
            if mp.shape[1] != n:
                raise ValueError(f"mode_prob has {mp.shape[1]} wheels but labels have {n}")
            pad[:, :n] = mp
        T["mode_prob"].append(pad)
        T["wheel_mask"].append(mask)
        T["mode_true"].append(mt)
    out = {k: np.concatenate(v) for k, v in T.items() if k != "q_levels"}
    out["q_levels"] = to_np(preds[0].get("q_levels", np.array([0.05, 0.5, 0.95])), float)
    if np.all(np.isnan(out["mode_prob"])):
        del out["mode_prob"]
    return out


def n_wheel_slots(t):
    return int(t["wheel_mask"].shape[1]) if "wheel_mask" in t else 0


def pad_wheels(t, W):
    """Re-pad a table's wheel arrays to width W >= its current width (pads are masked out)."""
    w0 = n_wheel_slots(t)
    if "wheel_mask" not in t or W == w0:
        return t
    if W < w0:
        raise ValueError(f"cannot pad {w0} wheel slots down to {W}; use crop_wheels")
    out = dict(t)
    for k in WHEEL_KEYS:
        if k in t:
            v = t[k]
            shp = (v.shape[0], W - w0) + v.shape[2:]
            out[k] = np.concatenate([v, np.full(shp, _PAD[k], v.dtype)], 1)
    return out


def crop_wheels(t, n=None):
    """Cut the wheel arrays to the first n slots (default: the largest real n in the table). Refuses to drop real
    wheels."""
    if "wheel_mask" not in t:
        return t
    real = int(t["wheel_mask"].sum(1).max()) if t["wheel_mask"].size else 0
    n = real if n is None else int(n)
    if t["wheel_mask"][:, n:].any():
        raise ValueError(f"crop to {n} wheels would drop real wheels (max n = {real})")
    out = dict(t)
    for k in WHEEL_KEYS:
        if k in t:
            out[k] = t[k][:, :n]
    return out


def concat(tables):
    """Row-concatenate PredTables with possibly different wheel widths (all must share q_levels)."""
    tables = list(tables)
    W = max(n_wheel_slots(t) for t in tables)
    q = tables[0]["q_levels"]
    keys = set(tables[0])
    for t in tables[1:]:
        if not np.allclose(t["q_levels"], q):
            raise ValueError("concat: q_levels differ")
        keys &= set(t)
    tables = [pad_wheels(t, W) for t in tables]
    out = {k: np.concatenate([t[k] for t in tables]) for k in keys if k != "q_levels"}
    out["q_levels"] = q
    return out


def save(path, table):
    validate(table)
    np.savez_compressed(path, **table)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        t = {k: z[k] for k in z.files}
    for k in STR_KEYS + tuple(k_ for k_ in OPT_STR_KEYS if k_ in t):
        t[k] = t[k].astype(str)
    validate(t)
    return t


def validate(t):
    miss = [k for k in REQ if k not in t]
    if miss:
        raise ValueError(f"PredTable missing keys {miss}")
    N = t["trial_id"].shape[0]
    if len(set(t["trial_id"].tolist())) != N:
        raise ValueError("duplicate trial_id in PredTable")
    for k in REQ:
        if k != "q_levels" and t[k].shape[0] != N:
            raise ValueError(f"{k} has {t[k].shape[0]} rows, expected {N}")
    if t["y"].shape[1] != 14 or t["y_q"].shape[1] != 14:
        raise ValueError("y / y_q must have 14 targets")
    if "wheel_mask" in t:
        W = t["wheel_mask"].shape[1]
        for k in ("mode_prob", "mode_true"):
            if k in t and t[k].shape[1] != W:
                raise ValueError(f"{k} has {t[k].shape[1]} wheel slots, wheel_mask has {W}")
        if "n_wheels" in t and not np.array_equal(t["wheel_mask"].sum(1), t["n_wheels"]):
            raise ValueError("n_wheels disagrees with wheel_mask")


def subset(t, idx):
    idx = np.asarray(idx)
    return {k: (v if k == "q_levels" else v[idx]) for k, v in t.items()}


def align(t, trial_ids):
    """Reorder table rows to `trial_ids` (all must be present)."""
    pos = {tid: i for i, tid in enumerate(t["trial_id"].tolist())}
    try:
        idx = np.array([pos[x] for x in trial_ids])
    except KeyError as e:
        raise ValueError(f"trial {e} missing from PredTable") from None
    return subset(t, idx)


def as_pred(t):
    """PredTable rows -> a C3-like Pred dict (padded wheels) for the conformal layer."""
    p = dict(y_q=t["y_q"], q_levels=t["q_levels"], ppc_score=t["ppc_score"])
    if "mode_prob" in t:
        mp = np.where(t["wheel_mask"][..., None], t["mode_prob"], 0.2)
        p.update(mode_prob=np.nan_to_num(mp, nan=0.2), wheel_mask=t["wheel_mask"])
    return p

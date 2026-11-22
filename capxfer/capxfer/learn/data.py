"""Data access for WS-C: shard loading, evidence views (rung / probe subset), batching across assemblies.

SWAP POINT: `SOURCE` below. "C_stub" reads ~/.cache/capxfer/C/data/<name>/manifest.json written by `_stub.py`.
When WS-A datagen shards (~/.cache/capxfer/A/<name>/manifest.json) + WS-B featurize land, set
SOURCE = "A_featurize" (implemented in `_load_shard_A`, which featurizes RawTrials with all pids and E3 and maps the
channel layout onto LAYOUT); every model consumes only what `apply_view` returns.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from capxfer.probes import LIBRARY, PIDS, TARGET_KEYS
from . import _stub

SOURCE = "C_stub"   # <- the one-line swap: "A_featurize"

RUNGS = ("E0", "E1", "E2", "E3")
DEFAULT_PIDS = ("P1", "P5")
# channel layout the models expect (stub layout; _load_shard_A maps WS-B's layout onto it)
WHEEL_CH = _stub.WHEEL_CH          # speed, duty, current, ref
BODY_CH = _stub.BODY_CH            # cmd(3), gyro, acc(2), lidar(3)
SEG = _stub.seg_slices()           # pid -> time slice (both legs), LIBRARY order
SEG_LEN = _stub.SEG_LEN


def manifest_path(name):
    if SOURCE == "C_stub":
        return _stub.ROOT / "data" / name / "manifest.json"
    own = _stub.ROOT / "A_gen" / name / "manifest.json"          # WS-C-generated sets with WS-A's datagen
    return own if own.exists() else Path(os.path.expanduser(f"~/.cache/capxfer/A/{name}/manifest.json"))


B_WHEEL_ORDER = (1, 2, 3, 0)   # WS-B featurize wheel channels [ref, enc, duty, current] -> ours [enc, duty, current, ref]


def from_featurize(F):
    """WS-B Feat (allowed primitives only, its channel order; seg_pid/seg_leg give the layout) -> WS-C layout
    (all 7 fixed-length primitive blocks in PIDS order, unseen blocks zero, our channel order, (B,C) masks)."""
    wx, bx = F["wheel_x"].float(), F["body_x"].float()
    B, n, L, _ = wx.shape
    seg_pid = torch.as_tensor(F["seg_pid"]).cpu().long()
    Lfull = sum(2 * SEG_LEN[p] for p in PIDS)
    tidx = torch.empty(L, dtype=torch.long)
    pos = {}
    for j in range(L):
        k = int(seg_pid[j])
        tidx[j] = SEG[PIDS[k]].start + pos.get(k, 0)
        pos[k] = pos.get(k, 0) + 1
    dev = wx.device
    tidx = tidx.to(dev)
    wfull = torch.zeros(B, n, Lfull, 4, device=dev)
    wfull[:, :, tidx] = wx[..., list(B_WHEEL_ORDER)]
    bfull = torch.zeros(B, Lfull, bx.shape[-1], device=dev)
    bfull[:, tidx] = bx
    wm = torch.as_tensor(F["wheel_mask"], device=dev).float()
    wm = (wm.amax(1) if wm.dim() == 3 else wm.expand(B, -1))[..., list(B_WHEEL_ORDER)] > 0
    bm = torch.as_tensor(F["body_mask"], device=dev).float()
    bm = (bm.expand(B, -1) if bm.dim() == 1 else bm) > 0
    pm = torch.as_tensor(F["pid_mask"], device=dev).float()
    pm = (pm.expand(B, -1) if pm.dim() == 1 else pm) > 0
    out = dict(wheel_x=wfull, wheel_mask=wm, wheel_geom=F["wheel_geom"].float(), body_x=bfull, body_mask=bm,
               pid_mask=pm, n_wheels=F["n_wheels"], wheel_valid=torch.ones(B, n, dtype=torch.bool, device=dev))
    for k in ("y", "drift", "labels"):
        if k in F and F[k] is not None:
            out[k] = F[k]
    return out


def _load_shard_A(path):
    """One WS-A datagen shard -> list of stub-format Feat dicts (all pids, E3), labels attached."""
    from capxfer import datagen, featurize
    feats = []
    for raw in datagen.load_shard(path):
        F = from_featurize(featurize.featurize(raw, "E3", PIDS))
        lab = raw["labels"]
        F = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in F.items()}
        F["wheel_x"] = F["wheel_x"].half()
        F["body_x"] = F["body_x"].half()
        F["labels"] = dict(mode=lab["mode"].cpu().long(), g=lab["g"].cpu().float(), ood=lab.get("ood", "none"),
                           family=(f"phys_{lab['asm_type']}" if lab.get("asm_type") not in (None, "random")
                                   else lab.get("family")), asm=raw["meta"]["assembly"].name)
        feats.append(F)
    return feats


def load_shard(path):
    if SOURCE == "C_stub":
        return [torch.load(path, weights_only=False)]
    return _load_shard_A(path)


def load_feats(name, max_shards=None, ood=None, keep=None):
    """List of Feat dicts (one per assembly x ood condition), all pids, all rungs. keep(F) -> bool filters groups
    (e.g. by labels['family'] to build a family split from a mixed WS-A dataset)."""
    mp = Path(manifest_path(name))
    m = json.loads(mp.read_text())
    want = _stub.params_hash()
    if m.get("params_sha256") not in (None, want):
        print(f"WARNING: {name} manifest params hash {m.get('params_sha256', '')[:8]} != current {want[:8]}")
    shards = [str(mp.parent / s["file"]) if isinstance(s, dict) else s for s in m["shards"]]
    if ood is not None:
        shards = [s for s in shards if f"_{ood}_" in Path(s).name or s.endswith(f"_{ood}.pt")]
    if max_shards:
        shards = shards[:max_shards]
    return [F for s in shards for F in load_shard(s) if keep is None or keep(F)]


# ----------------------------------------------------------------------------- evidence views
def pid_time_mask(pid_mask, L):
    """(B,7) bool -> (B,L) bool in the LIBRARY-order block layout."""
    tm = torch.zeros(pid_mask.shape[0], L, dtype=torch.bool, device=pid_mask.device)
    for k, p in enumerate(PIDS):
        tm[:, SEG[p]] = pid_mask[:, k:k + 1]
    return tm


def view_masks(B, rung, pids, has_lidar):
    """Per-sample channel masks. rung/pids: scalars or per-sample lists. has_lidar: (B,) bool."""
    rungs = [rung] * B if isinstance(rung, str) else list(rung)
    pidl = [pids] * B if (len(pids) == 0 or isinstance(pids[0], str)) else list(pids)
    ri = torch.tensor([RUNGS.index(r) for r in rungs])
    wm = torch.ones(B, len(WHEEL_CH), dtype=torch.bool)
    wm[:, 2] = ri >= 3
    bm = torch.zeros(B, len(BODY_CH), dtype=torch.bool)
    bm[:, :3] = True
    bm[:, 3:6] = (ri >= 1)[:, None]
    bm[:, 6:9] = ((ri >= 2) & has_lidar.cpu())[:, None]
    pm = torch.tensor([[p in pl for p in PIDS] for pl in pidl], dtype=torch.bool)
    return wm, bm, pm


def apply_view(F, rung="E0", pids=DEFAULT_PIDS, idx=None):
    """Feat restricted to what the robot may see at `rung` from primitives `pids` (zeroed, with masks)."""
    idx = torch.arange(F["wheel_x"].shape[0]) if idx is None else idx
    wx = F["wheel_x"][idx].float()
    bx = F["body_x"][idx].float()
    B, n, L, _ = wx.shape
    has_lidar = bx[..., 6:9].abs().sum((1, 2)) > 0
    wm, bm, pm = view_masks(B, rung, pids, has_lidar)
    tm = pid_time_mask(pm, L)
    wx = wx * wm[:, None, None, :] * tm[:, None, :, None]
    bx = bx * bm[:, None, :] * tm[:, :, None]
    out = dict(wheel_x=wx, wheel_mask=wm, wheel_geom=F["wheel_geom"][idx].float(), body_x=bx, body_mask=bm,
               pid_mask=pm, n_wheels=F["n_wheels"][idx], wheel_valid=torch.ones(B, n, dtype=torch.bool))
    for k in ("y", "drift"):
        if k in F:
            out[k] = F[k][idx].float()
    if "labels" in F:
        lab = F["labels"]
        out["labels"] = dict(mode=lab["mode"][idx], g=lab["g"][idx].float(), ood=lab.get("ood", "none"),
                             family=lab.get("family"), asm=lab.get("asm"))
    return out


def ensure_batch_fields(f):
    """Fill optional fields for a Feat produced by someone else's featurize (single assembly, (C,) masks)."""
    if "seg_pid" in f:                       # straight from WS-B featurize (field or sim)
        f = from_featurize(f)
    B, n = f["wheel_x"].shape[:2]
    f = dict(f)
    f.setdefault("wheel_valid", torch.ones(B, n, dtype=torch.bool))
    for k in ("wheel_mask", "body_mask", "pid_mask"):
        if k in f and f[k].dim() == 1:
            f[k] = f[k][None].expand(B, -1)
    f.setdefault("wheel_mask", torch.ones(B, f["wheel_x"].shape[-1], dtype=torch.bool))
    f.setdefault("body_mask", torch.ones(B, f["body_x"].shape[-1], dtype=torch.bool))
    f.setdefault("pid_mask", torch.ones(B, 7, dtype=torch.bool))
    return f


def collate(views):
    """Pad a list of viewed Feats (different n) into one batch."""
    nmax = max(v["wheel_x"].shape[1] for v in views)
    out = {}
    for k in ("wheel_x", "wheel_geom", "wheel_valid"):
        parts = []
        for v in views:
            x = v[k]
            pad = nmax - x.shape[1]
            if pad:
                x = torch.cat([x, torch.zeros((x.shape[0], pad) + x.shape[2:], dtype=x.dtype)], 1)
            parts.append(x)
        out[k] = torch.cat(parts)
    for k in ("wheel_mask", "body_x", "body_mask", "pid_mask", "n_wheels", "y", "drift"):
        if k in views[0]:
            out[k] = torch.cat([v[k] for v in views])
    if "labels" in views[0]:
        modes, gs = [], []
        for v in views:
            m, g = v["labels"]["mode"], v["labels"]["g"]
            pad = nmax - m.shape[1]
            modes.append(torch.cat([m, torch.zeros(m.shape[0], pad, dtype=m.dtype)], 1))
            gs.append(torch.cat([g, torch.ones(g.shape[0], pad)], 1))
        out["labels"] = dict(mode=torch.cat(modes), g=torch.cat(gs))
    return out


def random_pids(rng, p_default=0.4):
    if rng.uniform() < p_default:
        return DEFAULT_PIDS
    k = int(rng.integers(1, len(PIDS) + 1))
    return tuple(PIDS[j] for j in sorted(rng.choice(len(PIDS), size=k, replace=False).tolist()))


def batch_iter(feats, batch_size=64, per_asm=16, seed=0, rung=None, pids=None):
    """Infinite iterator of mixed-assembly training batches with random rung dropout and probe-subset masking.
    rung/pids None -> randomised per sample."""
    rng = np.random.default_rng(seed)
    k = max(1, batch_size // per_asm)
    while True:
        views = []
        for _ in range(k):
            F = feats[int(rng.integers(len(feats)))]
            B = F["wheel_x"].shape[0]
            idx = torch.tensor(rng.choice(B, size=min(per_asm, B), replace=False))
            r = rung or [RUNGS[int(i)] for i in rng.integers(4, size=len(idx))]
            if pids is None:
                pl = [random_pids(rng) for _ in range(len(idx))]
            else:
                pl = pids
            views.append(apply_view(F, r, pl, idx))
        yield collate(views)


def eval_iter(feats, rung, pids=DEFAULT_PIDS, batch_size=256):
    """Deterministic pass: one assembly per batch (batch_size trials), yields (Feat, meta)."""
    for F in feats:
        B = F["wheel_x"].shape[0]
        for s in range(0, B, batch_size):
            idx = torch.arange(s, min(B, s + batch_size))
            yield apply_view(F, rung, pids, idx)


# ----------------------------------------------------------------------------- summaries (ppc + analytic Bayes)
def hold_slices():
    """Per (pid, leg) hold-window time indices (20 Hz) in the block layout, TARGET_KEYS order."""
    out = []
    for p in LIBRARY:
        base = SEG[p.pid].start
        L = SEG_LEN[p.pid]
        for legi in range(2):
            a = base + legi * L + int(round((0.3 + 0.2) * 20))          # ramp + 0.2 s settle
            b = base + legi * L + int(round((0.3 + p.hold - 0.1) * 20))
            out.append(slice(a, b))
    return out


HOLDS = hold_slices()
N_SUM_W = 2 * len(HOLDS)     # per wheel: mean (speed - ref)/6, mean duty
N_SUM_B = 4 * len(HOLDS)     # body: gyro - cmd_wz, lidar - cmd (3)


def summaries(f):
    """Observed hold-window summaries + masks. f: viewed batch. Returns sw (B,n,28), mw (B,28), sb (B,56), mb (B,56)."""
    wx, bx = f["wheel_x"], f["body_x"]
    pm = f["pid_mask"]
    sw, sb = [], []
    for h in HOLDS:
        seg = wx[:, :, h]
        sw.append(((seg[..., 0] - seg[..., 3]) / 6.0).mean(-1))
        sw.append(seg[..., 1].mean(-1))
        bs = bx[:, h]
        sb.append(bs[..., 3].mean(-1) - bs[..., 2].mean(-1))
        for j in range(3):
            sb.append((bs[..., 6 + j].mean(-1) - bs[..., j].mean(-1)) / (0.3 if j < 2 else 1.0))
    sw = torch.stack(sw, -1)
    sb = torch.stack(sb, -1)
    pk = pm.repeat_interleave(2, 1)                                     # (B,14) per hold
    mw = pk.repeat_interleave(2, 1)
    bm = f["body_mask"]
    mb = torch.stack([pk & bm[:, 3:4], pk & bm[:, 6:7], pk & bm[:, 7:8], pk & bm[:, 8:9]], -1).reshape(pk.shape[0], -1)
    return sw, mw, sb, mb

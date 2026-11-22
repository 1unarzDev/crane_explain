"""I2: thin wrapper around the batched simulator that returns hold-window means (paired, noise-free)."""
from __future__ import annotations

import numpy as np
import torch

from capxfer import sim
from capxfer.probes import Schedule

_FRAC_SET = False


def gpu():
    global _FRAC_SET
    if torch.cuda.is_available():
        if not _FRAC_SET:
            torch.cuda.set_per_process_memory_fraction(1.8 / 16.0)
            _FRAC_SET = True
        return "cuda"
    return "cpu"


def centred_domain(B, vals=None, device="cpu"):
    """Every DR key at the DR_DEFAULT midpoint (overridable), no CG offset, no wheel yaw error -> exact pairing."""
    d = {k: torch.full((B,), 0.5 * (v[0] + v[1]), dtype=torch.float32, device=device)
         for k, v in sim.DR_DEFAULT.items() if isinstance(v, tuple)}
    d["cg_dx"] = torch.zeros(B, device=device)
    d["cg_dy"] = torch.zeros(B, device=device)
    d["wheel_yaw_err_std"] = 0.0
    for k, v in (vals or {}).items():
        d[k] = torch.as_tensor(v, dtype=torch.float32, device=device).reshape(-1).expand(B).clone()
    return d


def windows_reduce(R):
    """-> dict rho (B,14), drift (B,14), om/duty (B,14,n), twist (B,14,3), omref (14,n)"""
    rho, drift = sim.targets_from_twist(R)
    t = R["t"]
    out = dict(rho=rho.cpu().numpy(), drift=drift.cpu().numpy())
    om, du, tw = [], [], []
    for pid, leg, ta, tb, xc in R["meta"]["schedule"].windows():
        m = (t >= ta) & (t <= tb)
        om.append(R["omega"][:, m].float().mean(1))
        du.append(R["duty"][:, m].float().mean(1))
        tw.append(R["twist"][:, m].float().mean(1))
    out["om"] = torch.stack(om, 1).cpu().numpy()
    out["duty"] = torch.stack(du, 1).cpu().numpy()
    out["twist"] = torch.stack(tw, 1).cpu().numpy()
    return out


def run(asm, params, modes, gs, dom=None, chunk=None, payload=None, mu_factor=None, seed=0):
    """modes/gs: np (B,n). dom: dict of (B,) tensors (default centred). Returns windows_reduce dict (numpy)."""
    dev = gpu()
    B, n = modes.shape
    chunk = chunk or max(1, 8192 // n)
    dom = dom or centred_domain(B)
    outs = []
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        dd = {k: (v[s:e] if torch.is_tensor(v) else v) for k, v in dom.items()}
        pl = None if payload is None else {k: v[s:e] for k, v in payload.items()}
        mf = None if mu_factor is None else mu_factor[s:e]
        R = sim.simulate(asm, params, dd, torch.as_tensor(modes[s:e]), torch.as_tensor(gs[s:e], dtype=torch.float32),
                         device=dev, payload=pl, mu_factor=mf, seed=seed)
        outs.append(windows_reduce(R))
        del R
    return {k: np.concatenate([o[k] for o in outs], 0) for k in outs[0]}

"""Predictor base class (CONTRACTS C3) and shared helpers."""
from __future__ import annotations

import copy
import importlib
import time

import numpy as np
import torch

from . import data as D

DEFAULT_CFG = dict(
    steps=4000, lr=3e-4, wd=1e-4, device="cuda", mem_frac=0.25, q_levels=(0.05, 0.5, 0.95), val_every=500,
    val_batches=20, seed=0, d=128, layers=3, log_every=100,
)


def current_mem_frac():
    """The process's current CUDA memory-fraction cap (1.0 = uncapped), or None without CUDA."""
    if not torch.cuda.is_available():
        return None
    try:
        return float(torch.cuda.get_per_process_memory_fraction())
    except (AttributeError, RuntimeError):
        return None


def cap_memory(frac):
    """Lower the process's CUDA memory cap to `frac`; NEVER raises an existing (lower) cap (integrator M1)."""
    if frac is None or not torch.cuda.is_available():
        return current_mem_frac()
    cur = current_mem_frac()
    want = float(frac) if cur is None else min(float(frac), cur)
    try:
        torch.cuda.set_per_process_memory_fraction(want)
    except RuntimeError:
        pass
    return want


def setup_device(cfg):
    dev = cfg.get("device", "cuda")
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    if dev == "cuda":
        cap_memory(cfg.get("mem_frac", 0.25))
    return torch.device(dev)


def to_dev(f, dev):
    out = {}
    for k, v in f.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev)
        elif isinstance(v, dict):
            out[k] = {kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def quantiles_from_samples(ys, levels):
    """ys (B,S,14) -> (B,14,K)."""
    q = torch.tensor(levels, dtype=ys.dtype, device=ys.device)
    return torch.quantile(ys, q, dim=1).permute(1, 2, 0)


def chunked(predict_fn, feat, chunk, *a, **k):
    """Run predict_fn on trial chunks and concatenate (keeps GPU memory bounded)."""
    B = feat["wheel_x"].shape[0]
    if B <= chunk:
        return predict_fn(feat, *a, **k)
    parts = []
    for s in range(0, B, chunk):
        sub = {kk: (v[s:s + chunk] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B else v)
               for kk, v in feat.items() if kk != "labels"}
        parts.append(predict_fn(sub, *a, **k))
    return {kk: (torch.cat([p[kk] for p in parts]) if parts[0][kk] is not None else None) for kk in parts[0]}


class Predictor:
    name = "base"

    def __init__(self, cfg=None):
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.dev = setup_device(self.cfg)
        self.history = []

    # -- subclasses implement: modules() -> dict[str, nn.Module], fit, predict; extra_state()/load_extra()
    def modules(self):
        return {}

    def extra_state(self):
        return {}

    def load_extra(self, st):
        pass

    def save(self, path):
        torch.save(dict(cls=f"{type(self).__module__}:{type(self).__name__}", cfg=self.cfg,
                        state={k: m.state_dict() for k, m in self.modules().items()}, extra=self.extra_state(),
                        history=self.history), path)

    @classmethod
    def load(cls, path, device=None, mem_frac=None):
        """mem_frac: GPU cap for this process. Loading only ever LOWERS the current cap (to mem_frac, or to the
        checkpoint's training mem_frac if that is lower than the current cap); it never raises it (M1)."""
        ck = torch.load(path, weights_only=False, map_location="cpu")
        mod, name = ck["cls"].split(":")
        klass = getattr(importlib.import_module(mod), name)
        cfg = dict(ck["cfg"])
        if device:
            cfg["device"] = device
        if mem_frac is not None:
            cfg["mem_frac"] = float(mem_frac)
        obj = klass(cfg)
        for k, m in obj.modules().items():
            m.load_state_dict(ck["state"][k])
        obj.load_extra(ck["extra"])
        obj.history = ck.get("history", [])
        return obj

    # -- generic training loop over a loss(batch) -> (loss, logs)
    def _train_loop(self, params, loss_fn, train_iter, val_iter, cfg):
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
        steps = cfg["steps"]
        warm = max(1, int(0.05 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(i, steps) / steps))))
        best, best_state = float("inf"), None
        val_cache = None
        t0 = time.time()
        for it in range(steps):
            for m in self.modules().values():
                m.train()
            b = to_dev(next(train_iter), self.dev)
            loss, logs = loss_fn(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if (it + 1) % cfg["log_every"] == 0:
                rec = dict(step=it + 1, loss=float(loss.detach()), t=time.time() - t0, **{k: float(v) for k, v in logs.items()})
                self.history.append(rec)
                print(self.name, rec, flush=True)
            if val_iter is not None and ((it + 1) % cfg["val_every"] == 0 or it + 1 == steps):
                if val_cache is None:
                    val_cache = [next(val_iter) for _ in range(cfg["val_batches"])]
                for m in self.modules().values():
                    m.eval()
                with torch.no_grad():
                    vl = float(np.mean([float(loss_fn(to_dev(vb, self.dev))[0]) for vb in val_cache]))
                self.history.append(dict(step=it + 1, val=vl))
                print(self.name, "val", it + 1, vl, flush=True)
                if vl < best:
                    best = vl
                    best_state = {k: copy.deepcopy(m.state_dict()) for k, m in self.modules().items()}
        if best_state is not None:
            for k, m in self.modules().items():
                m.load_state_dict(best_state[k])
        for m in self.modules().values():
            m.eval()
        return dict(best_val=best, train_time_s=time.time() - t0)

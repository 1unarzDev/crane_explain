"""Raw (uncalibrated) evaluation metrics for Pred dicts."""
from __future__ import annotations

import math

import numpy as np
import torch


def y_nll_kde(ys, y):
    """Mean per-dimension NLL of y (B,14) under a Gaussian KDE of samples ys (B,S,14) (Silverman bandwidth)."""
    S = ys.shape[1]
    sd = ys.std(1, keepdim=True)
    h = (1.06 * sd * S ** -0.2).clamp(min=2e-3)
    z = (y[:, None] - ys) / h
    lp = torch.logsumexp(-0.5 * z ** 2, 1) - math.log(S) - torch.log(h[:, 0]) - 0.5 * math.log(2 * math.pi)
    return -lp.mean(-1)                                                              # (B,)


def trial_metrics(pred, f, q_levels=(0.05, 0.5, 0.95)):
    """Per-trial metric arrays (numpy) for aggregation."""
    y = f["y"].float().cpu()
    yq = pred["y_q"].float()
    lv = list(q_levels)
    lo, md, hi = yq[..., lv.index(0.05)], yq[..., lv.index(0.5)], yq[..., lv.index(0.95)]
    out = dict(
        y_mae=(md - y).abs().mean(-1).numpy(),
        y_nll=y_nll_kde(pred["y_samples"].float(), y).numpy(),
        cover90=((y >= lo) & (y <= hi)).float().mean(-1).numpy(),
        width90=(hi - lo).mean(-1).numpy(),
        ppc=pred["ppc_score"].float().numpy(),
    )
    if pred.get("drift_q") is not None and "drift" in f:
        dq = pred["drift_q"].float()
        dr = torch.nan_to_num(f["drift"].float().cpu())
        out["drift_mae"] = (dq[..., lv.index(0.5)] - dr).abs().mean(-1).numpy()
        out["drift_cover_up95"] = (dr <= dq[..., lv.index(0.95)]).float().mean(-1).numpy()
    if "labels" in f:
        faulted = ((f["labels"]["mode"].cpu() != 0) & f["wheel_valid"].cpu()).any(1)
        # capability-changing fault: some direction loses > 0.1 of its commanded ratio (healthy y ~ 1)
        out["consequential"] = (faulted & ((1 - y).amax(-1) > 0.1)).numpy()
        out["healthy_trial"] = (~faulted).numpy()
    if pred.get("mode_prob") is not None and "labels" in f:
        mp = pred["mode_prob"].float().clamp(min=1e-6)
        lab = f["labels"]["mode"].cpu()
        v = f["wheel_valid"].cpu().float()
        nll = -(mp.gather(-1, lab[..., None])[..., 0].log() * v).sum(1) / v.sum(1)
        am = mp.argmax(-1)
        acc = ((am == lab).float() * v).sum(1) / v.sum(1)
        fault = (lab != 0).float() * v
        frec = ((am == lab).float() * fault).sum(1) / fault.sum(1).clamp(min=1)
        exact = (((am == lab) | (v == 0)).all(1)).float()
        # coast and brake merged (nearly indistinguishable in this sim: back-drive torque locks a coasting wheel)
        merge = torch.tensor([0, 1, 2, 2, 4])
        mp_cb = mp.clone()
        mp_cb[..., 2] = mp[..., 2] + mp[..., 3]
        mp_cb[..., 3] = 0
        am_cb, lab_cb = mp_cb.argmax(-1), merge[lab]
        out["mode_acc_cb"] = (((am_cb == lab_cb).float() * v).sum(1) / v.sum(1)).numpy()
        out["fault_recall_cb"] = np.where(fault.sum(1).numpy() > 0, (((am_cb == lab_cb).float() * fault).sum(1)
                                          / fault.sum(1).clamp(min=1)).numpy(), np.nan)
        out.update(mode_nll=nll.numpy(), mode_acc=acc.numpy(), exact=exact.numpy(),
                   fault_recall=np.where(fault.sum(1).numpy() > 0, frec.numpy(), np.nan))
        g = f["labels"]["g"].cpu().float()
        weak = (lab == 1).float() * v
        gm = torch.nan_to_num(pred["g_mean"].float(), nan=0.6)
        out["g_mae_weak"] = np.where(weak.sum(1).numpy() > 0,
                                     ((gm - g).abs() * weak).sum(1).numpy() / np.maximum(weak.sum(1).numpy(), 1),
                                     np.nan)
    return out


FLAGS = ("healthy_trial", "consequential")


def aggregate(rows, where=None):
    """rows: list of trial_metrics dicts -> dict of means (nan-aware). where: None | (flag, bool) trial filter."""
    keys = set().union(*[r.keys() for r in rows])
    sel = [np.ones(len(r["y_mae"]), bool) if where is None else (np.asarray(r[where[0]]) == where[1]) for r in rows]
    out = {}
    for k in sorted(keys):
        if k in FLAGS:
            continue
        v = np.concatenate([np.asarray(r[k], float)[m] for r, m in zip(rows, sel) if k in r])
        out[k] = float(np.nanmean(v)) if v.size and np.isfinite(v).any() else None
    out["n_trials"] = int(sum(m.sum() for m in sel))
    return out


def auroc(scores_in, scores_out):
    """P(score_out > score_in)."""
    a = np.asarray(scores_in)
    b = np.asarray(scores_out)
    if len(a) == 0 or len(b) == 0:
        return None
    allv = np.concatenate([a, b])
    ranks = allv.argsort().argsort() + 1
    rb = ranks[len(a):].sum()
    return float((rb - len(b) * (len(b) + 1) / 2) / (len(a) * len(b)))

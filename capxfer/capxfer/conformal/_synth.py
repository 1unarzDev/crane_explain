"""Synthetic Pred / label generator (local stub for WS-C outputs; tests and the analysis dry run use it).

Generative model per trial (one assembly group g, n wheels):
  latent wheel modes ~ categorical; capability y (14,) = clip(1 - effect(modes) + group_offset_g + noise, 0, 1.2)
  The "model" predicts quantiles from the wheel effect it can see, with spread misscaled by `spread_bias`
  and no knowledge of the group offset (so group offsets act as an assembly-type shift).
  mode_prob: a noisy, sharpened posterior centred on the true mode with accuracy controlled by `sharpness`.
  ppc_score: chi-square-like typicality, larger for out-of-model trials (ood=True).
Everything is numpy; values are meaningless physically.
"""
from __future__ import annotations

import numpy as np

Q_LEVELS = np.array([0.05, 0.5, 0.95])
_Z = 1.6448536269514722


def make_batch(rng, B, n=4, *, group_offset=0.0, group_scale=1.0, noise=0.05, spread_bias=0.8,
               sharpness=2.0, ood_frac=0.0, ood_shift=0.15, p_fault=0.6):
    """-> (pred dict, labels dict, y (B,14))."""
    mode = np.zeros((B, n), int)
    faulty = rng.random(B) < p_fault
    for b in np.flatnonzero(faulty):
        w = rng.integers(n)
        mode[b, w] = rng.integers(1, 5)
    effect_w = np.array([0.0, 0.25, 0.4, 0.5, 0.3])[mode]                 # (B,n)
    load = rng.random((1, n, 14)) * 0.5 + 0.5                              # fixed per batch (assembly)
    mean = 1.0 - (effect_w[..., None] * load).mean(1) * 2.0                # (B,14)
    sig = noise * (1.0 + 2.0 * (mode > 0).any(1, keepdims=True)) * np.ones((1, 14))
    ood = rng.random(B) < ood_frac
    y = mean + group_offset + group_scale * sig * rng.standard_normal((B, 14)) - ood[:, None] * ood_shift
    psig = spread_bias * sig
    y_q = np.stack([mean - _Z * psig, mean, mean + _Z * psig], -1)
    logits = rng.standard_normal((B, n, 5)) * 1.0
    logits[np.arange(B)[:, None], np.arange(n)[None, :], mode] += sharpness
    logits[..., 0] += 0.5
    mp = np.exp(logits - logits.max(-1, keepdims=True))
    mp /= mp.sum(-1, keepdims=True)
    ppc = rng.chisquare(6, B) + ood * rng.chisquare(6, B) * 2.5
    S = 64
    ys = mean[:, None, :] + psig[:, None, :] * rng.standard_normal((B, S, 14))
    g = np.where(mode == 1, rng.uniform(0.3, 0.9, (B, n)), np.where(mode == 0, 1.0, 0.0))
    pred = dict(mode_prob=mp, g_mean=g, g_std=np.full((B, n), 0.05), y_samples=ys, y_q=y_q,
                q_levels=Q_LEVELS.copy(), ppc_score=ppc)
    labels = dict(mode=mode, g=g, ood=np.where(ood, "payload", "none"))
    return pred, labels, y


def make_groups(rng, K, per_group, *, tau=0.0, **kw):
    """K exchangeable groups with offsets ~ N(0, tau^2). -> list of (pred, labels, y, offset)."""
    out = []
    for _ in range(K):
        off = rng.normal(0.0, tau) if tau > 0 else 0.0
        n = int(rng.choice([4, 8, 12, 16]))
        p, l, y = make_batch(rng, per_group, n=n, group_offset=off, **kw)
        out.append((p, l, y, off))
    return out

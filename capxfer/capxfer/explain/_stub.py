"""Synthetic stand-ins for WS-C Pred and WS-D intervals / wheel sets / OOD flags (CONTRACTS C3, C4).

Used only by WS-E tests and demos until the real pipeline exists. Each scenario returns everything explain() and
score() need for B trials of one assembly, with injected ground truth (labels, y_true).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from capxfer.assembly import load_params, physical
from capxfer.probes import TARGET_KEYS

PARAMS = Path(__file__).resolve().parents[2] / "params" / "module_measured.yaml"
OK, WEAK, COAST, BRAKE, ROLLER = range(5)
K = {k: i for i, k in enumerate(TARGET_KEYS)}
SIDEWAYS_DOCK = {"task": "the sideways dock", "required": ["P2+"]}


def assembly(name="single"):
    return physical(name, load_params(PARAMS))


def _y(base=0.95, **over):
    y = np.full(14, base)
    for k, v in over.items():
        y[K[k.replace("p", "+").replace("m", "-")]] = v
    return y


def _sets(n, spec):
    s = np.zeros((n, 5), bool)
    s[:, OK] = True
    for w, modes in spec.items():
        s[w] = False
        s[w, list(modes)] = True
    return s


def _probs(n, spec):
    p = np.zeros((n, 5))
    p[:, OK] = 0.97
    p[:, 1:] = 0.03 / 4
    for w, dist in spec.items():
        p[w] = 0.0
        for m, v in dist.items():
            p[w, m] = v
        rest = 1 - p[w].sum()
        p[w][p[w] == 0] = rest / max((p[w] == 0).sum(), 1)
    return p


def case(name, B=1, seed=0):
    """Scenarios: healthy, singleton_roller, ambiguous_e0, roller_e2, ood_payload, ood_low_mu."""
    rng = np.random.default_rng(seed)
    asm = assembly("single")
    n = asm.n
    ood, rung = "none", "E2"
    mode = np.zeros(n, int)
    if name == "healthy":
        y = _y(0.95)
        half = np.full(14, 0.08)
        sets, probs, flag = _sets(n, {}), _probs(n, {}), False
    elif name in ("singleton_roller", "roller_e2"):
        # fl roller-locked: strafe and one diagonal lost, the rest retained
        mode[0] = ROLLER
        y = _y(0.95, P2p=0.42, P2m=0.40, P6yp=0.38, P6ym=0.37, P3p=0.55, P3m=0.56, P5p=0.86, P5m=0.85)
        half = np.full(14, 0.08)
        sets, probs, flag = _sets(n, {0: [ROLLER]}), _probs(n, {0: {ROLLER: 0.9, COAST: 0.06}}), False
    elif name == "ambiguous_e0":
        # same physical fault, encoders only: sets cannot separate fl from rr, intervals wide; point estimate
        # puts its argmax on the wrong wheel (rr)
        rung = "E0"
        mode[0] = ROLLER
        y = _y(0.95, P2p=0.42, P2m=0.40, P6yp=0.38, P6ym=0.37, P3p=0.55, P3m=0.56, P5p=0.86, P5m=0.85)
        half = np.full(14, 0.20)
        half[[K["P2+"], K["P6y+"]]] = 0.36  # left strafe interval straddles the threshold
        sets = _sets(n, {0: [OK, ROLLER, COAST], 3: [OK, ROLLER]})
        probs = _probs(n, {0: {OK: 0.45, ROLLER: 0.40}, 3: {ROLLER: 0.52, OK: 0.40}})
        flag = False
    elif name in ("ood_payload", "ood_low_mu"):
        ood = "payload" if name == "ood_payload" else "low_mu"
        y = _y(0.62, P1p=0.90, P1m=0.88) if ood == "payload" else _y(0.55)
        half = np.full(14, 0.10)
        # the in-model posterior is fooled: it believes rr is weak
        sets = _sets(n, {3: [WEAK]})
        probs = _probs(n, {3: {WEAK: 0.7, OK: 0.2}})
        flag = True
    else:
        raise ValueError(name)
    Y = np.repeat(y[None], B, 0) + rng.normal(0, 0.01, (B, 14))
    centre = np.repeat(y[None], B, 0) + rng.normal(0, 0.02, (B, 14))
    lo, hi = centre - half, centre + half
    S = 64
    y_samples = centre[:, None, :] + rng.normal(0, 1, (B, S, 14)) * (half / 4.0)
    q = np.stack([centre - half * 0.5, centre, centre + half * 0.5], -1)
    pred = dict(mode_prob=np.repeat(probs[None], B, 0), g_mean=np.ones((B, n)), g_std=np.full((B, n), 0.1),
                y_samples=y_samples, y_q=q, ppc_score=np.full(B, 9.0 if flag else 1.0))
    labels = dict(mode=np.repeat(mode[None], B, 0), g=np.ones((B, n)), ood=ood)
    return dict(pred=pred, intervals=(lo, hi), wheel_sets=np.repeat(sets[None], B, 0), ood_flag=np.full(B, flag),
                asm=asm, rung=rung, labels=labels, y_true=Y)

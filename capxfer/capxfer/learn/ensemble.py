"""ensemble: M independently-seeded members (default 5 x blackbox; cfg member="structured_npe" for structured),
uncalibrated mixture. y_samples = pooled member samples; y_q from the mixture; ppc = mean member score."""
from __future__ import annotations

import os
import tempfile

import torch

from .base import Predictor, quantiles_from_samples
from .blackbox import Blackbox
from .structured_npe import StructuredNPE

MEMBERS = {"blackbox": Blackbox, "structured_npe": StructuredNPE}


class Ensemble(Predictor):
    name = "ensemble"

    def __init__(self, cfg=None):
        cfg = {"n_members": 5, "member": "blackbox", **(cfg or {})}
        super().__init__(cfg)
        klass = MEMBERS[self.cfg["member"]]
        self.members = [klass({**self.cfg, "seed": int(self.cfg["seed"]) + 1000 * (k + 1)})
                        for k in range(self.cfg["n_members"])]

    def modules(self):
        return {f"m{k}.{n}": mod for k, m in enumerate(self.members) for n, mod in m.modules().items()}

    def extra_state(self):
        return [m.extra_state() for m in self.members]

    def load_extra(self, st):
        for m, s in zip(self.members, st):
            m.load_extra(s)

    def fit(self, train_iter, val_iter, cfg=None):
        out = {}
        for k, m in enumerate(self.members):
            out[f"m{k}"] = m.fit(train_iter, val_iter, cfg)
        return out

    @torch.no_grad()
    def predict(self, feat, n_samples=256):
        M = len(self.members)
        per = max(1, -(-n_samples // M))
        ps = [m.predict(feat, per) for m in self.members]
        ys = torch.cat([p["y_samples"] for p in ps], 1)[:, :max(n_samples, M)]
        ds = torch.cat([p["drift_samples"] for p in ps], 1)[:, :max(n_samples, M)]
        mp = None
        if ps[0]["mode_prob"] is not None:
            mp = torch.stack([p["mode_prob"] for p in ps]).mean(0)
        gm = torch.stack([p["g_mean"] for p in ps])
        return dict(mode_prob=mp, g_mean=gm.mean(0), g_std=torch.sqrt((torch.stack([p["g_std"] for p in ps]) ** 2
                                                                         ).mean(0) + gm.var(0)),
                    y_samples=ys, y_q=quantiles_from_samples(ys, self.cfg["q_levels"]),
                    drift_samples=ds, drift_q=quantiles_from_samples(ds, self.cfg["q_levels"]),
                    ppc_score=torch.stack([p["ppc_score"] for p in ps]).mean(0))


class StructuredEnsemble(Ensemble):
    """5 x structured_npe (seeds seed+1000k): mixture posterior over modes (mean of member mode_prob), pooled
    y/drift posterior-predictive samples, mean ppc. Gives ensemble-level spread AND per-wheel diagnosis."""
    name = "structured_ensemble"

    def __init__(self, cfg=None):
        super().__init__({"member": "structured_npe", **(cfg or {})})

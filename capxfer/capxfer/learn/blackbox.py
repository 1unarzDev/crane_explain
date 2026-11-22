"""blackbox: the same wheel-set trunk, pooled -> direct y quantile regression (pinball). No per-wheel latent.
y_samples by inverse-CDF interpolation of the predicted quantiles; ppc_score = Mahalanobis distance of the pooled
embedding to the training embeddings (the only OOD signal a black-box has)."""
from __future__ import annotations

import numpy as np
import torch

from . import data as D
from . import nets as N
from .base import Predictor, chunked, to_dev

LEVELS = (0.01, 0.025, 0.05) + tuple(np.round(np.arange(0.1, 0.91, 0.05), 3)) + (0.95, 0.975, 0.99)


def interp_quantiles(qv, levels, u):
    """qv (B,14,K) at `levels`; u (...) in (0,1) -> values. Linear inside, flat outside [l0, lK]."""
    lv = torch.tensor(levels, dtype=qv.dtype, device=qv.device)
    u = u.clamp(lv[0], lv[-1])
    i = torch.searchsorted(lv, u.contiguous()).clamp(1, len(levels) - 1)
    l0, l1 = lv[i - 1], lv[i]
    w = (u - l0) / (l1 - l0)
    return i, w


class Blackbox(Predictor):
    name = "blackbox"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        torch.manual_seed(self.cfg["seed"])
        d = self.cfg["d"]
        self.trunk = N.Trunk(d, self.cfg["layers"]).to(self.dev)
        self.qhead = N.QuantileHead(d, len(LEVELS)).to(self.dev)
        self.dhead = N.QuantileHead(d, len(LEVELS), base=0.0, step=0.01).to(self.dev)
        self.emb_mu = None
        self.emb_prec = None

    def modules(self):
        return dict(trunk=self.trunk, qhead=self.qhead, dhead=self.dhead)

    def extra_state(self):
        return dict(emb_mu=self.emb_mu, emb_prec=self.emb_prec)

    def load_extra(self, st):
        self.emb_mu = st.get("emb_mu")
        self.emb_prec = st.get("emb_prec")
        if self.emb_mu is not None:
            self.emb_mu, self.emb_prec = self.emb_mu.to(self.dev), self.emb_prec.to(self.dev)

    def _forward(self, f):
        tok, ctx = self.trunk(f)
        q = self.qhead(tok, ctx, f["wheel_valid"])
        emb = torch.cat([N.masked_pool(tok, f["wheel_valid"][..., None], 1), ctx], -1)
        self._dq = self.dhead(tok, ctx, f["wheel_valid"])
        return q, emb

    def loss(self, b):
        q, _ = self._forward(b)
        lv = torch.tensor(LEVELS, device=q.device)
        e = b["y"][..., None] - q
        pin = torch.maximum(lv * e, (lv - 1) * e).mean()
        ed = torch.nan_to_num(b["drift"])[..., None] - self._dq
        pind = torch.maximum(lv * ed, (lv - 1) * ed).mean()
        return pin + pind, dict(pinball=pin, pinball_drift=pind)

    def fit(self, train_iter, val_iter, cfg=None):
        cfg = {**self.cfg, **(cfg or {})}
        out = self._train_loop(list(self.trunk.parameters()) + list(self.qhead.parameters()), self.loss,
                               train_iter, val_iter, cfg)
        with torch.no_grad():
            embs = torch.cat([self._forward(to_dev(next(train_iter), self.dev))[1] for _ in range(40)])
        mu = embs.mean(0)
        C = torch.cov((embs - mu).T)
        C = C + 1e-3 * torch.eye(C.shape[0], device=C.device) * C.diag().mean()
        self.emb_mu, self.emb_prec = mu, torch.linalg.inv(C)
        return out

    def predict(self, feat, n_samples=256):
        return chunked(self._predict, feat, 32, n_samples)

    @torch.no_grad()
    def _predict(self, feat, n_samples=256):
        f = to_dev(D.ensure_batch_fields(feat), self.dev)
        q, emb = self._forward(f)
        B = q.shape[0]
        gen = torch.Generator(device=self.dev).manual_seed(int(self.cfg["seed"]) + 29)
        u = torch.rand(B, 14, n_samples, device=self.dev, generator=gen)
        i, w = interp_quantiles(q, LEVELS, u)
        ys = (torch.gather(q, 2, i - 1) * (1 - w) + torch.gather(q, 2, i) * w).permute(0, 2, 1)
        uq = torch.tensor(self.cfg["q_levels"], device=self.dev).expand(B, 14, -1)
        i, w = interp_quantiles(q, LEVELS, uq)
        y_q = torch.gather(q, 2, i - 1) * (1 - w) + torch.gather(q, 2, i) * w
        dq = self._dq
        i, w = interp_quantiles(dq, LEVELS, uq)
        drift_q = (torch.gather(dq, 2, i - 1) * (1 - w) + torch.gather(dq, 2, i) * w).clamp(min=0)
        u = torch.rand(B, 14, n_samples, device=self.dev, generator=gen)
        i, w = interp_quantiles(dq, LEVELS, u)
        drift_s = (torch.gather(dq, 2, i - 1) * (1 - w) + torch.gather(dq, 2, i) * w).permute(0, 2, 1).clamp(min=0)
        if self.emb_mu is not None:
            dlt = emb - self.emb_mu
            ppc = ((dlt @ self.emb_prec) * dlt).sum(-1) / dlt.shape[-1]
        else:
            ppc = torch.zeros(B, device=self.dev)
        n = f["wheel_x"].shape[1]
        nanb = torch.full((B, n), float("nan"))
        return dict(mode_prob=None, g_mean=nanb, g_std=nanb.clone(), y_samples=ys.cpu(), y_q=y_q.cpu(),
                    drift_samples=drift_s.cpu(), drift_q=drift_q.cpu(), ppc_score=ppc.cpu())

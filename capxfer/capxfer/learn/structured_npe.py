"""structured_npe (proposed) and nominal_sim (H5 ablation: same model, nominal-domain training data).

Per-wheel amortized posterior q(mode_i, g_i | features) from the wheel-set trunk; capability is decoded by sampling
health assignments and passing them through the physics-structured KinematicDecoder (or, optionally, the simulator
itself). ppc_score = -(1/dims) log E_q[p(observed hold summaries | health)] under the learned ForwardModel.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as Fn

from . import data as D
from . import nets as N
from .base import Predictor, chunked, quantiles_from_samples, to_dev


def logit(g):
    g = g.clamp(1e-3, 1 - 1e-3)
    return torch.log(g) - torch.log1p(-g)


class StructuredNPE(Predictor):
    name = "structured_npe"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        torch.manual_seed(self.cfg["seed"])
        d = self.cfg["d"]
        self.trunk = N.Trunk(d, self.cfg["layers"]).to(self.dev)
        self.head = N.HealthHead(d).to(self.dev)
        self.dec = N.KinematicDecoder().to(self.dev)
        self.fwd = N.ForwardModel().to(self.dev)

    def modules(self):
        return dict(trunk=self.trunk, head=self.head, dec=self.dec, fwd=self.fwd)

    def loss(self, b):
        valid = b["wheel_valid"]
        mode, g = b["labels"]["mode"], b["labels"]["g"]
        tok, ctx = self.trunk(b)
        logits, mu, ls = self.head(tok)
        vf = valid.float()
        ce = (Fn.cross_entropy(logits.reshape(-1, N.N_MODES), mode.reshape(-1), reduction="none")
              .view_as(vf) * vf).sum() / vf.sum()
        weak = ((mode == 1) & valid).float()
        z = logit(g)
        gnll = ((0.5 * ((z - mu) / ls.exp()) ** 2 + ls) * weak).sum() / weak.sum().clamp(min=1)
        htok = N.health_tokens(mode, g)[:, None]
        ym, ys, dm, ds = self.dec(htok, b["wheel_geom"], valid, return_drift=True)
        ynll = (0.5 * ((b["y"][:, None] - ym) / ys) ** 2 + ys.log()).mean()
        ld = torch.log(torch.nan_to_num(b["drift"]).clamp(min=0) + N.DRIFT_EPS)[:, None]
        dnll = (0.5 * ((ld - dm) / ds) ** 2 + ds.log()).mean()
        sw, mw, sb, mb = D.summaries(b)
        ll, cnt = N.ForwardModel.loglik(self.fwd(htok, b["wheel_geom"], valid), sw, mw, sb, mb, valid)
        fnll = -(ll[:, 0] / cnt.clamp(min=1)).mean()
        loss = ce + 0.3 * gnll + ynll + 0.3 * fnll + 0.3 * dnll
        return loss, dict(ce=ce, gnll=gnll, ynll=ynll, fnll=fnll, dnll=dnll)

    def fit(self, train_iter, val_iter, cfg=None):
        cfg = {**self.cfg, **(cfg or {})}
        return self._train_loop(list(self.trunk.parameters()) + list(self.head.parameters()) +
                                list(self.dec.parameters()) + list(self.fwd.parameters()),
                                self.loss, train_iter, val_iter, cfg)

    # ------------------------------------------------------------------ inference
    def posterior(self, f):
        tok, ctx = self.trunk(f)
        logits, mu, ls = self.head(tok)
        return logits.log_softmax(-1), mu, ls

    def sample_health(self, logp, mu, ls, S, gen=None):
        B, n, _ = logp.shape
        mode = torch.multinomial(logp.exp().reshape(-1, N.N_MODES), S, replacement=True, generator=gen)
        mode = mode.view(B, n, S).permute(0, 2, 1)                                       # (B,S,n)
        z = mu[:, None] + ls.exp()[:, None] * torch.randn(B, S, n, device=mu.device, generator=gen)
        return mode, torch.sigmoid(z)

    @torch.no_grad()
    def decode(self, mode, g, geom, valid, chunk=4096):
        B, S, n = mode.shape
        per = max(1, chunk // S)
        outs = []
        for s in range(0, B, per):
            outs.append(self.dec(N.health_tokens(mode[s:s + per], g[s:s + per]), geom[s:s + per], valid[s:s + per],
                                 return_drift=True))
        return tuple(torch.cat([o[k] for o in outs]) for k in range(4))

    def predict(self, feat, n_samples=256, n_ppc=32, decoder="kinematic"):
        if decoder == "sim":
            return self._predict(feat, n_samples, n_ppc, decoder)
        return chunked(self._predict, feat, 32, n_samples, n_ppc, decoder)

    @torch.no_grad()
    def _predict(self, feat, n_samples=256, n_ppc=32, decoder="kinematic"):
        f = to_dev(D.ensure_batch_fields(feat), self.dev)
        valid, geom = f["wheel_valid"], f["wheel_geom"]
        gen = torch.Generator(device=self.dev).manual_seed(int(self.cfg["seed"]) + 17)
        logp, mu, ls = self.posterior(f)
        mode_prob = logp.exp() * valid[..., None]
        zs = mu[..., None] + ls.exp()[..., None] * torch.randn(*mu.shape, 128, device=self.dev, generator=gen)
        gs = torch.sigmoid(zs)
        mode, g = self.sample_health(logp, mu, ls, n_samples, gen)
        if decoder == "sim":
            y_samples, drift_samples = self.decode_sim(mode, g, geom, valid)
        else:
            ym, ysd, dm, dsd = self.decode(mode, g, geom, valid)
            y_samples = ym + ysd * torch.randn(ym.shape, device=self.dev, generator=gen)
            drift_samples = (torch.exp(dm + dsd * torch.randn(dm.shape, device=self.dev, generator=gen))
                             - N.DRIFT_EPS).clamp(min=0)
        S2 = min(n_ppc, n_samples)
        htok = N.health_tokens(mode[:, :S2], g[:, :S2])
        sw, mw, sb, mb = D.summaries(f)
        ll, cnt = N.ForwardModel.loglik(self.fwd(htok, geom, valid), sw, mw, sb, mb, valid)
        ppc = -(torch.logsumexp(ll, 1) - np.log(S2)) / cnt.clamp(min=1)
        return dict(mode_prob=mode_prob.cpu(), g_mean=gs.mean(-1).cpu(), g_std=gs.std(-1).cpu(),
                    y_samples=y_samples.cpu(), y_q=quantiles_from_samples(y_samples, self.cfg["q_levels"]).cpu(),
                    drift_samples=drift_samples.cpu(),
                    drift_q=quantiles_from_samples(drift_samples, self.cfg["q_levels"]).cpu(),
                    ppc_score=ppc.cpu())

    def decode_sim(self, mode, g, geom, valid, S_sim=32, chunk=512):
        """Physics decoding by the simulator itself (nominal-centred domain randomisation); one assembly per call.
        Returns (B,S,14) by resampling S_sim simulated draws per trial."""
        from capxfer import sim
        from ._stub import load_params, simulate_feat
        from .analytic_bayes import assembly_from_geom
        params = load_params()
        B, S, n = mode.shape
        nv = int(valid[0].sum())
        asm = assembly_from_geom(geom[0, :nv].cpu().numpy(), params)
        m = mode[:, :S_sim, :nv].reshape(-1, nv).cpu()
        gg = g[:, :S_sim, :nv].reshape(-1, nv).cpu()
        rng = np.random.default_rng(0)
        ys, ds = [], []
        for s in range(0, m.shape[0], chunk):
            F = simulate_feat(asm, params, m[s:s + chunk].shape[0], rng, mode=m[s:s + chunk], g=gg[s:s + chunk],
                              device=str(self.dev))
            ys.append(F["y"])
            ds.append(F["drift"])
        ys = torch.cat(ys).view(B, S_sim, 14).to(self.dev)
        ds = torch.cat(ds).view(B, S_sim, 14).to(self.dev)
        idx = torch.randint(0, S_sim, (B, S), device=self.dev)[..., None].expand(B, S, 14)
        return torch.gather(ys, 1, idx), torch.gather(ds, 1, idx)


class NominalSim(StructuredNPE):
    """Same model; trained on the nominal-parameter (non-randomised) domain. train.py feeds it `train_nominal`."""
    name = "nominal_sim"

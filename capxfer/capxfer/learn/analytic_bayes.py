"""analytic_bayes: enumerate healthy + single + (capped) double wheel-fault hypotheses, simulate each once with the
NOMINAL parameters (no randomisation, noise-free sensors), and score the observed hold-window summaries under a
Gaussian likelihood. Posterior over hypotheses -> per-wheel mode marginals, weak-gain moments, and a capability
mixture (each hypothesis' nominal-sim y + Gaussian model error). ppc_score = -(1/dims) log marginal likelihood.
The only fitted quantities are the likelihood scale and model-error sigma (grid search on validation data).
cfg["table_dr"]: the fixed plant the hypothesis tables are simulated at (default: _stub.NOMINAL_DR, the parameter-file
nominal). Generation W uses `dr_centre(sim.DR_WIDE)`, the centre of the wide training DR ("wide-DR nominal")."""
from __future__ import annotations

import hashlib
import itertools
import time

import numpy as np
import torch

from capxfer import assembly as A
from capxfer.probes import PIDS
from . import _stub
from . import data as D
from .base import Predictor, quantiles_from_samples, to_dev

G_GRID = (0.35, 0.5, 0.65, 0.8)
MODE_PRIOR = {1: 0.35, 2: 0.15, 3: 0.2, 4: 0.3}
SIG_W = (0.02, 0.05)                      # (speed-ref)/6, duty
SIG_B = (0.03, 0.05, 0.05, 0.05)          # gyro-cmd, lidar-cmd (scaled)


def assembly_from_geom(geom, params):
    """Rebuild a simulatable Assembly from wheel_geom (n,8) (positions relative to the nominal CG)."""
    geom = np.asarray(geom, float)
    p, w, r = geom[:, :2], geom[:, 2:4], geom[:, 4:6]
    mod = np.round(geom[:, 7] * 4).astype(int)
    ids = sorted(set(mod.tolist()))
    remap = {m: k for k, m in enumerate(ids)}
    wm = np.array([remap[m] for m in mod])
    mpos = np.stack([p[wm == k].mean(0) for k in range(len(ids))])
    mass = float(np.mean(list(params["module"]["mass_kg"].values())))
    side = float(params["module"]["side_x_m"])
    return A.Assembly(name="from_geom", modules=[(f"M{k}", 0, 0, 0) for k in range(len(ids))], wheel_pos=p.copy(),
                      w_hat=w.copy(), r_hat=r.copy(), wheel_module=wm, wheel_corner=[f"w{i}" for i in range(len(p))],
                      module_mass=np.full(len(ids), mass), module_pos=mpos, lidar=np.ones(len(ids), bool), side=side)


def dr_centre(dr):
    """Degenerate DR at the centre of a DR spec: ranges -> (mid, mid); scalar std entries (cg_off, wheel_yaw_err) -> 0."""
    out = {}
    for k, v in dr.items():
        if isinstance(v, (tuple, list)):
            m = 0.5 * (float(v[0]) + float(v[1]))
            out[k] = (m, m)
        else:
            out[k] = 0.0
    return out


def hypotheses(n, cap=1024, seed=0):
    """List of (mode (n,), g (n,), prior)."""
    H = [(np.zeros(n, np.int64), np.ones(n, np.float32), 0.25)]
    singles = []
    for i in range(n):
        for m, pm in MODE_PRIOR.items():
            gl = G_GRID if m == 1 else (1.0,)
            for gv in gl:
                mo = np.zeros(n, np.int64)
                gg = np.ones(n, np.float32)
                mo[i], gg[i] = m, gv
                singles.append((mo, gg, 0.55 * pm / len(gl) / n))
    H += singles
    dbl = []
    for i, j in itertools.combinations(range(n), 2):
        for (mi, pi), (mj, pj) in itertools.product(MODE_PRIOR.items(), repeat=2):
            dbl.append((i, j, mi, mj, pi * pj))
    room = max(0, cap - len(H))
    rng = np.random.default_rng(seed)
    if len(dbl) > room:
        sel = rng.choice(len(dbl), size=room, replace=False)
        dbl = [dbl[k] for k in sel]
    tot = sum(d[4] for d in dbl) or 1.0
    for i, j, mi, mj, pp in dbl:
        mo = np.zeros(n, np.int64)
        gg = np.ones(n, np.float32)
        mo[i], mo[j] = mi, mj
        if mi == 1:
            gg[i] = 0.5
        if mj == 1:
            gg[j] = 0.5
        H.append((mo, gg, 0.2 * pp / tot))
    return H


class AnalyticBayes(Predictor):
    name = "analytic_bayes"

    def __init__(self, cfg=None):
        cfg = {"cap": 1024, "sigma_scale": 2.0, "sigma_y": 0.03, "sim_chunk": 512, "table_dr": None, **(cfg or {})}
        super().__init__(cfg)
        self.cache = {}
        self.params = _stub.load_params()

    def extra_state(self):
        return dict(sigma_scale=self.cfg["sigma_scale"], sigma_y=self.cfg["sigma_y"], table_dr=self.cfg["table_dr"])

    def load_extra(self, st):
        self.cfg.update(st)

    def _hyp_table(self, geom):
        key = hashlib.md5(np.round(np.asarray(geom), 4).tobytes() + repr(self.cfg.get("table_dr")).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        t0 = time.time()
        n = geom.shape[0]
        asm = assembly_from_geom(geom, self.params)
        H = hypotheses(n, self.cfg["cap"])
        mode = torch.tensor(np.stack([h[0] for h in H]))
        g = torch.tensor(np.stack([h[1] for h in H]))
        prior = torch.tensor([h[2] for h in H], dtype=torch.float32)
        rng = np.random.default_rng(0)
        sw, sb, ys, dr = [], [], [], []
        for s in range(0, len(H), self.cfg["sim_chunk"]):
            F = _stub.simulate_feat(asm, self.params, mode[s:s + 512].shape[0], rng, dr=self.table_dr(),
                                    mode=mode[s:s + 512], g=g[s:s + 512], device=str(self.dev), noise=False)
            v = D.apply_view(F, "E3", PIDS)
            a, _, b, _ = D.summaries(v)
            sw.append(a)
            sb.append(b)
            ys.append(F["y"].float())
            dr.append(torch.nan_to_num(F["drift"].float()))
        tab = dict(mode=mode.to(self.dev), g=g.to(self.dev), logprior=prior.log().to(self.dev),
                   sw=torch.cat(sw).to(self.dev), sb=torch.cat(sb).to(self.dev), y=torch.cat(ys).to(self.dev), drift=torch.cat(dr).to(self.dev),
                   build_s=time.time() - t0)
        self.cache[key] = tab
        return tab

    def table_dr(self):
        td = self.cfg.get("table_dr")
        return {k: (tuple(v) if isinstance(v, list) else v) for k, v in td.items()} if td else _stub.NOMINAL_DR

    def _loglik(self, f, tab, scale):
        sw, mw, sb, mb = D.summaries(f)
        n = int(f["wheel_valid"][0].sum())
        sw = sw[:, :n]
        sig_w = torch.tensor(SIG_W, device=self.dev).repeat(D.N_SUM_W // 2) * scale
        sig_b = torch.tensor(SIG_B, device=self.dev).repeat(D.N_SUM_B // 4) * scale
        out = []
        Hn = tab["sw"].shape[0]
        for s in range(0, Hn, 128):
            dw = (sw[:, None] - tab["sw"][None, s:s + 128]) / sig_w                     # (B,h,n,28)
            lw = (-0.5 * dw ** 2 * mw[:, None, None].float()).sum((-1, -2))
            db = (sb[:, None] - tab["sb"][None, s:s + 128]) / sig_b
            lb = (-0.5 * db ** 2 * mb[:, None].float()).sum(-1)
            out.append(lw + lb)
        ll = torch.cat(out, 1)
        norm = (mw.float().sum(-1) * n * 0 + (mw.float() * torch.log(sig_w)).sum(-1) * n
                + (mb.float() * torch.log(sig_b)).sum(-1) + 0.9189 * (mw.float().sum(-1) * n + mb.float().sum(-1)))
        cnt = mw.float().sum(-1) * n + mb.float().sum(-1)
        return ll - norm[:, None], cnt

    def fit(self, train_iter, val_iter, cfg=None):
        """Grid-search likelihood scale and model-error sigma on validation batches (one assembly each)."""
        if val_iter is None:
            return {}
        t0 = time.time()
        batches = [next(val_iter) for _ in range((cfg or {}).get("ab_val_batches", 3))]
        best = None
        for sc in (0.5, 1.0, 2.0, 4.0, 8.0):
            for sy in (0.01, 0.03, 0.06):
                self.cfg["sigma_scale"], self.cfg["sigma_y"] = sc, sy
                tot = 0.0
                for b in batches:
                    p = self.predict(b, 128)
                    lab = b["labels"]["mode"]
                    v = b["wheel_valid"]
                    pm = p["mode_prob"].clamp(min=1e-6)
                    tot += float(-(pm.gather(-1, lab[..., None])[..., 0].log() * v).sum() / v.sum())
                    yq = p["y_samples"]
                    tot += float(((b["y"][:, None] - yq) ** 2).mean().sqrt()) * 0  # mode NLL drives sigma_scale
                    s = yq.std(1) + 1e-3
                    tot += float((0.5 * ((b["y"] - yq.mean(1)) / s) ** 2 + s.log()).mean())
                if best is None or tot < best[0]:
                    best = (tot, sc, sy)
        self.cfg["sigma_scale"], self.cfg["sigma_y"] = best[1], best[2]
        return dict(sigma_scale=best[1], sigma_y=best[2], fit_s=time.time() - t0)

    @torch.no_grad()
    def predict(self, feat, n_samples=256):
        f = to_dev(D.ensure_batch_fields(feat), self.dev)
        B, nmax = f["wheel_valid"].shape
        n = int(f["wheel_valid"][0].sum())
        tab = self._hyp_table(f["wheel_geom"][0, :n].cpu().numpy())
        ll, cnt = self._loglik(f, tab, self.cfg["sigma_scale"])
        lp = ll + tab["logprior"][None]
        lml = torch.logsumexp(lp, 1)
        post = (lp - lml[:, None]).exp()                                                 # (B,H)
        oh = torch.nn.functional.one_hot(tab["mode"], 5).float()                         # (H,n,5)
        mode_prob = torch.zeros(B, nmax, 5, device=self.dev)
        mode_prob[:, :n] = torch.einsum("bh,hnk->bnk", post, oh)
        weak = (tab["mode"] == 1).float()
        wsum = post @ weak
        gm = (post @ (weak * tab["g"])) / wsum.clamp(min=1e-9)
        g2 = (post @ (weak * tab["g"] ** 2)) / wsum.clamp(min=1e-9)
        gm = torch.where(wsum > 1e-9, gm, torch.full_like(gm, 0.6))
        gsd = torch.sqrt((g2 - gm ** 2).clamp(min=0) + 1e-4)
        g_mean = torch.full((B, nmax), float("nan"), device=self.dev)
        g_std = g_mean.clone()
        g_mean[:, :n], g_std[:, :n] = gm, gsd
        gen = torch.Generator(device=self.dev).manual_seed(int(self.cfg["seed"]) + 5)
        idx = torch.multinomial(post, n_samples, replacement=True, generator=gen)       # (B,S)
        ys = tab["y"][idx] + self.cfg["sigma_y"] * torch.randn(B, n_samples, 14, device=self.dev, generator=gen)
        dsm = (tab["drift"][idx] * torch.exp(0.3 * torch.randn(B, n_samples, 14, device=self.dev, generator=gen))
               + self.cfg["sigma_y"] * torch.randn(B, n_samples, 14, device=self.dev, generator=gen).abs())
        return dict(mode_prob=mode_prob.cpu(), g_mean=g_mean.cpu(), g_std=g_std.cpu(), y_samples=ys.cpu(),
                    y_q=quantiles_from_samples(ys, self.cfg["q_levels"]).cpu(), drift_samples=dsm.cpu(),
                    drift_q=quantiles_from_samples(dsm, self.cfg["q_levels"]).cpu(),
                    ppc_score=(-lml / cnt.clamp(min=1)).cpu())

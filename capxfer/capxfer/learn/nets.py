"""Networks: shared permutation-equivariant wheel-set trunk, heads, physics-structured capability decoder, forward model.

All modules take a *viewed* Feat batch (data.apply_view / collate) already on the device.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from capxfer.probes import LIBRARY, TARGET_KEYS
from . import data as D

from ._stub import load_params as _lp
R_NOM = float(_lp()["wheel"]["radius_m"])   # firmware IK radius (params file)
W_SCALE = torch.tensor([1 / 6.0, 1.0, 1 / 3.0, 1 / 6.0])                   # speed, duty, current, ref
B_SCALE = torch.tensor([1 / 0.3, 1 / 0.3, 1.0, 1.0, 0.5, 0.5, 1 / 0.3, 1 / 0.3, 1.0])
N_MODES = 5
DRIFT_EPS = 0.01


def char_length(geom, valid):
    """Lc from wheel_geom: col6 = |p|/Lc, so Lc = max |p| / max col6."""
    p = geom[..., :2].norm(dim=-1)
    big = torch.where(valid, p, torch.zeros_like(p)).amax(1)
    c6 = torch.where(valid, geom[..., 6], torch.zeros_like(p)).amax(1).clamp(min=1e-3)
    return (big / c6).clamp(min=1e-3)


def wheel_kin(geom, twist):
    """Ideal rolling wheel speed for body twist (..., 3) at each wheel. geom (B,n,8), twist (B,L,3) -> (B,n,L)."""
    x, y = geom[..., 0, None], geom[..., 1, None]
    wx, wy, rx, ry = (geom[..., k, None] for k in (2, 3, 4, 5))
    vx, vy, wz = (twist[:, None, :, k] for k in range(3))
    vcx = vx - wz * y
    vcy = vy + wz * x
    den = R_NOM * (wx * rx + wy * ry)
    den = torch.where(den.abs() < 1e-6, torch.ones_like(den), den)        # padded wheels (zero geometry)
    return (vcx * rx + vcy * ry) / den


def masked_pool(h, m, dim):
    """mean and max over `dim` with bool mask m (broadcastable)."""
    mf = m.float()
    s = (h * mf).sum(dim) / mf.sum(dim).clamp(min=1)
    mx = torch.where(m, h, torch.full_like(h, -1e4)).amax(dim)
    mx = torch.where(mf.sum(dim) > 0, mx, torch.zeros_like(mx))
    return torch.cat([s, mx], -1)


def rel_features(geom, Lc):
    """Pairwise relative features (B,n,n,8)."""
    p = geom[..., :2] / Lc[:, None, None]
    d = p[:, None, :, :] - p[:, :, None, :]
    r, w = geom[..., 4:6], geom[..., 2:4]
    rr = (r[:, :, None] * r[:, None, :]).sum(-1)
    ww = (w[:, :, None] * w[:, None, :]).sum(-1)
    rc = r[:, :, None, 0] * r[:, None, :, 1] - r[:, :, None, 1] * r[:, None, :, 0]
    same = (geom[:, :, None, 7] == geom[:, None, :, 7]).float()
    return torch.stack([d[..., 0], d[..., 1], d.norm(dim=-1), rr, ww, rc, same,
                        torch.exp(-d.norm(dim=-1) * 3)], -1)


class RelBlock(nn.Module):
    """Pre-LN transformer block over wheels with a learned relative-geometry attention bias."""

    def __init__(self, d, heads=4, n_rel=8):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.bias = nn.Sequential(nn.Linear(n_rel, 32), nn.GELU(), nn.Linear(32, heads))
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, rel, valid):
        B, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, n, 3, self.h, self.dk).unbind(2)
        logits = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.dk) + self.bias(rel).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~valid[:, None, None, :], -1e4)
        a = logits.softmax(-1)
        x = x + self.o(torch.einsum("bhij,bjhd->bihd", a, v).reshape(B, n, d))
        return x + self.ff(self.ln2(x))


class TConv(nn.Module):
    def __init__(self, cin, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, d, 5, padding=2), nn.GELU(),
            nn.Conv1d(d, d, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(d, d, 5, padding=8, dilation=4), nn.GELU(),
            nn.Conv1d(d, d, 5, padding=16, dilation=8))

    def forward(self, x):  # (N,C,L)
        return self.net(x)


class Trunk(nn.Module):
    """Per-wheel temporal conv on [wheel channels, kinematic projections, body channels] + geometry embedding,
    then relative-geometry set attention over wheels. Returns per-wheel tokens (B,n,d), body context (B,d)."""

    def __init__(self, d=128, layers=3, dt=64):
        super().__init__()
        self.cw = 4 + 4 + 9          # wheel (4) + derived (4) + body broadcast (9)
        self.cm = 4 + 9 + 7          # channel masks + pid mask
        self.tconv = TConv(self.cw, dt)
        self.bconv = TConv(9 + 9, dt)
        self.geom = nn.Sequential(nn.Linear(8 + self.cm, d), nn.GELU(), nn.Linear(d, d))
        self.wproj = nn.Sequential(nn.Linear(2 * dt, d), nn.GELU(), nn.Linear(d, d))
        self.bproj = nn.Sequential(nn.Linear(2 * dt + self.cm, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([RelBlock(d) for _ in range(layers)])
        self.d = d

    def forward(self, f):
        wx, bx, geom, valid = f["wheel_x"], f["body_x"], f["wheel_geom"], f["wheel_valid"]
        B, n, L, _ = wx.shape
        dev = wx.device
        wm, bm, pm = f["wheel_mask"].float(), f["body_mask"].float(), f["pid_mask"].float()
        tmask = (bx.abs().sum(-1) + wx.abs().sum((1, 3))) > 0                       # (B,L)
        wxs = wx * W_SCALE.to(dev)
        bxs = bx * B_SCALE.to(dev)
        # measured body twist where visible (lidar for vx,vy; gyro for wz), else the command
        lid = bm[:, 6:7, None].transpose(1, 2)
        gy = bm[:, 3:4, None].transpose(1, 2)
        meas = torch.stack([bx[..., 6] * lid[..., 0] + bx[..., 0] * (1 - lid[..., 0]),
                            bx[..., 7] * lid[..., 0] + bx[..., 1] * (1 - lid[..., 0]),
                            bx[..., 3] * gy[..., 0] + bx[..., 2] * (1 - gy[..., 0])], -1)
        kin = wheel_kin(geom, meas) * tmask[:, None] * valid[..., None]                               # (B,n,L)
        spd, ref = wx[..., 0], wx[..., 3]
        derived = torch.stack([(spd - ref) / 3, kin / 6, (kin - spd) / 3, (kin - ref) / 3], -1)
        x = torch.cat([wxs, derived, bxs[:, None].expand(B, n, L, 9)], -1)           # (B,n,L,cw)
        h = self.tconv(x.reshape(B * n, L, self.cw).transpose(1, 2)).transpose(1, 2)  # (Bn,L,dt)
        tm = tmask[:, None].expand(B, n, L).reshape(B * n, L, 1)
        hw = self.wproj(masked_pool(h, tm, 1)).view(B, n, self.d)
        masks = torch.cat([wm, bm, pm], -1)
        hb = self.bconv(torch.cat([bxs, bm[:, None].expand(B, L, 9)], -1).transpose(1, 2)).transpose(1, 2)
        ctx = self.bproj(torch.cat([masked_pool(hb, tmask[..., None], 1), masks], -1))
        tok = hw + self.geom(torch.cat([geom, masks[:, None].expand(B, n, self.cm)], -1)) + ctx[:, None]
        Lc = char_length(geom, valid)
        rel = rel_features(geom, Lc)
        for blk in self.blocks:
            tok = blk(tok, rel, valid)
        return tok, ctx


class HealthHead(nn.Module):
    """Per-wheel mode logits (5) and logit-normal weak gain (mu, log sigma of logit g)."""

    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_MODES + 2))

    def forward(self, tok):
        o = self.net(tok)
        return o[..., :N_MODES], o[..., N_MODES], o[..., N_MODES + 1].clamp(-4, 2)


class QuantileHead(nn.Module):
    """Pooled trunk -> non-crossing quantiles of y (B,14,K)."""

    def __init__(self, d=128, K=19, base=1.0, step=0.05):
        super().__init__()
        self.K, self.base, self.step = K, base, step
        self.net = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d), nn.GELU(),
                                 nn.Linear(2 * d, 14 * K))

    def forward(self, tok, ctx, valid):
        pooled = masked_pool(tok, valid[..., None], 1)
        o = self.net(torch.cat([pooled, ctx], -1)).view(-1, 14, self.K)
        return torch.cat([o[..., :1] * self.step * 4 + self.base, Fn.softplus(o[..., 1:]) * self.step], -1).cumsum(-1)


# ----------------------------------------------------------------------------- physics-structured decoder
def probe_twists(Lc):
    """Commanded hold twists for the 14 targets (B,14,3), with the schedule's yaw scaling."""
    from capxfer.probes import WHEEL_SPEED_CAP, W_NOM
    tw = []
    for p in LIBRARY:
        for leg in (1, -1):
            tw.append([leg * c for c in p.twist])
    tw = torch.tensor(tw, dtype=torch.float32, device=Lc.device)[None].repeat(Lc.shape[0], 1, 1)
    sc = torch.clamp(WHEEL_SPEED_CAP / (Lc * W_NOM), max=1.0)
    tw[..., 2] = tw[..., 2] * sc[:, None]
    return tw


def health_tokens(mode, g):
    """mode (...,n) long, g (...,n) -> (...,n,6) one-hot + gain."""
    return torch.cat([Fn.one_hot(mode, N_MODES).float(), torch.where(mode == 1, g, torch.ones_like(g))[..., None]],
                     -1)


class KinematicDecoder(nn.Module):
    """y = h(health, geometry): each wheel contributes a learned 2x2 stiffness (roller-axis / free-axis) and a target
    contact velocity in its (r, f) frame, both functions of ONLY its own health, local commanded contact velocity and
    static-load share; the body twist is the exact rigid-body weighted-least-squares composition of the wheels
    (two fixed-point passes, the second re-evaluating local inputs at the achieved twist). Morphology enters only
    through exact kinematics, so the decoder is defined for any assembly."""

    def __init__(self, d=96, iters=2):
        super().__init__()
        self.iters = iters
        cin = 6 + 4 + 2 + 2
        self.phi = nn.Sequential(nn.Linear(cin, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 8))
        self.log_s0 = nn.Parameter(torch.tensor(-3.5))
        self.log_sd0 = nn.Parameter(torch.tensor(-1.0))
        self.drift_bias = nn.Parameter(torch.zeros(14))

    def forward(self, htok, geom, valid, Lc=None, return_drift=False):
        """htok (B,S,n,6), geom (B,n,8), valid (B,n) -> y mean (B,S,14), y std (B,S,14)
        [, log(drift + DRIFT_EPS) mean and std (B,S,14): drift from the composed twist, log-normal noise]."""
        B, S, n, _ = htok.shape
        Lc = char_length(geom, valid) if Lc is None else Lc
        xc = probe_twists(Lc)                                                         # (B,14,3)
        p = geom[..., :2]
        r = geom[..., 4:6]
        f = torch.stack([-r[..., 1], r[..., 0]], -1)
        # A_i: contact velocity = v + wz x p  -> (B,n,2,3)
        Ai = torch.zeros(B, n, 2, 3, device=geom.device)
        Ai[..., 0, 0] = 1
        Ai[..., 1, 1] = 1
        Ai[..., 0, 2] = -p[..., 1]
        Ai[..., 1, 2] = p[..., 0]
        Rf = torch.stack([r, f], -2)                                                  # (B,n,2,2) rows r, f
        G = Rf @ Ai                                                                   # (B,n,2,3): (r,f)-components
        nv = valid.float().sum(1).clamp(min=1)
        load = (valid.float() / nv[:, None])[..., None] * 4                           # static load share
        ext = torch.cat([geom[..., 6:7], load], -1)                                   # (B,n,2)
        Wt = torch.tensor([1.0, 1.0, 1.0], device=geom.device)
        Wt = Wt[None].repeat(B, 1)
        Wt[:, 2] = Lc ** 2
        uc = torch.einsum("bnij,bkj->bkni", G, xc)                  # (B,14,n,2) commanded
        xi = xc[:, None].expand(B, S, 14, 3)
        vm = valid[:, None, None, :, None].float()
        for it in range(self.iters):
            ua = torch.einsum("bnij,bskj->bskni", G, xi)                              # achieved (B,S,14,n,2)
            inp = torch.cat([htok[:, :, None].expand(B, S, 14, n, 6),
                             uc[:, None].expand(B, S, 14, n, 2) / 0.3, ua / 0.3,
                             ext[:, None, None].expand(B, S, 14, n, 2),
                             (ua - uc[:, None]) / 0.3], -1)
            o = self.phi(inp)
            k = Fn.softplus(o[..., 0:2]) + 1e-3                                       # stiffness along r, f
            Mt = o[..., 2:6].unflatten(-1, (2, 2)) + torch.eye(2, device=geom.device)
            tgt = (Mt @ uc[:, None].expand(B, S, 14, n, 2)[..., None])[..., 0] * vm   # target (r,f) velocity
            kk = k * vm
            # normal equations: sum_i G_i^T K_i G_i xi = sum_i G_i^T K_i tgt_i
            M = torch.einsum("bnai,bskna,bnaj->bskij", G, kk, G) + 1e-4 * torch.eye(3, device=geom.device)
            rhs = torch.einsum("bnai,bskna->bski", G, kk * tgt)
            xi = torch.linalg.solve(M, rhs[..., None])[..., 0]
            logv = o[..., 6]
            logvd = o[..., 7]
        num = (xi * xc[:, None] * Wt[:, None, None]).sum(-1)
        den = (xc * xc * Wt[:, None]).sum(-1)[:, None]
        y = num / den
        std = torch.sqrt(torch.exp(2 * self.log_s0) +
                         (Fn.softplus(logv) * 0.01 * vm[..., 0]).sum(-1) / nv[:, None, None])
        res = xi - y[..., None] * xc[:, None]
        drift = torch.sqrt((res * res * Wt[:, None, None]).sum(-1) / den + 1e-8)
        ld_mu = torch.log(drift + DRIFT_EPS) + self.drift_bias
        ld_std = torch.sqrt(torch.exp(2 * self.log_sd0) +
                            (Fn.softplus(logvd) * 0.1 * vm[..., 0]).sum(-1) / nv[:, None, None])
        if not return_drift:
            return y, std
        return y, std, ld_mu, ld_std


class ForwardModel(nn.Module):
    """p(observed hold summaries | health, geometry): per-wheel and body Gaussians, for posterior-predictive checks
    and as the learned likelihood surrogate."""

    def __init__(self, d=96, layers=2):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(6 + 8, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([RelBlock(d) for _ in range(layers)])
        self.hw = nn.Linear(d, 2 * D.N_SUM_W)
        self.hb = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 2 * D.N_SUM_B))

    def forward(self, htok, geom, valid):
        """htok (B,S,n,6) -> (mu_w, logs_w) (B,S,n,28), (mu_b, logs_b) (B,S,56)."""
        B, S, n, _ = htok.shape
        g = geom[:, None].expand(B, S, n, 8).reshape(B * S, n, 8)
        v = valid[:, None].expand(B, S, n).reshape(B * S, n)
        x = self.emb(torch.cat([htok.reshape(B * S, n, 6), g], -1))
        rel = rel_features(g, char_length(g, v))
        for blk in self.blocks:
            x = blk(x, rel, v)
        ow = self.hw(x).view(B, S, n, 2, D.N_SUM_W)
        ob = self.hb(masked_pool(x, v[..., None], 1)).view(B, S, 2, D.N_SUM_B)
        return (ow[..., 0, :], ow[..., 1, :].clamp(-6, 2)), (ob[..., 0, :], ob[..., 1, :].clamp(-6, 2))

    @staticmethod
    def loglik(out, sw, mw, sb, mb, valid):
        """Per-sample log-likelihood of observed summaries (B,S) and the observed-dimension count (B,)."""
        (mu_w, ls_w), (mu_b, ls_b) = out
        lw = -0.5 * ((sw[:, None] - mu_w) / ls_w.exp()) ** 2 - ls_w - 0.9189
        mwf = (mw[:, None, :] & valid[:, :, None]).float()                          # (B,n,28)
        lw = (lw * mwf[:, None]).sum((-1, -2))
        lb = -0.5 * ((sb[:, None] - mu_b) / ls_b.exp()) ** 2 - ls_b - 0.9189
        lb = (lb * mb[:, None].float()).sum(-1)
        cnt = mwf.sum((-1, -2)) + mb.float().sum(-1)
        return lw + lb, cnt

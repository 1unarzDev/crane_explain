"""RawTrial -> Feat (CONTRACTS C2). The same function runs on simulated and real RawTrials.

Robot-visible features use only the allowed probe primitives (`pids`, default P1, P5) and the sensing rung. The
evaluator-only target `y` / `drift` uses only `truth_pose` and `cmd` (via the schedule's hold twists) over every
(primitive, leg) of the full schedule; keys whose primitive is absent from the trial are NaN.

Layout (20 Hz): each allowed primitive contributes its two legs, each leg resampled over [seg.t0, seg.t1) into
round(20 * leg_duration) bins (40 for 1 s holds, 44 for P6). Primitives are concatenated in PIDS order.
`seg_pid (Lw,)` gives the PIDS index of every step, `seg_leg (Lw,)` the leg (+1/-1).

Channels
  wheel_x (B,n,Lw,4): [ref_speed (firmware IK of the published cmd, nominal radius, rad/s),
                       enc_speed (count difference over 50 ms, rad/s), duty, current (E3, A)]
  wheel_mask (B,n,4): 1 where the channel is present (current can be wired on a subset of wheels)
  body_x (B,Lb,9): [cmd vx, vy, wz, gyro_z (E1), acc x, y (E1), lidar vx, vy, wz (E2)]; Lb == Lw
  body_mask (9,)
Absent or NaN values are zero; the masks say which channels exist.

Target (evaluator-only): for each TARGET_KEY window (t_a, t_b) from `Schedule.windows()` shifted by the truth latency,
the body-frame displacement integral s(t) = int R(-theta) dp is built from truth_pose increments; a robust (Huber IRLS)
polynomial fit of s over the new-frame samples (centred window; degree TARGET_FIT_DEG) gives xi_real = (vx, vy, wz). Then
  rho   = <xi_real, xi_cmd>_W / |xi_cmd|_W^2,  W = diag(1, 1, Lc^2)
  drift = |xi_real - rho xi_cmd|_W / |xi_cmd|_W
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .probes import PIDS, TARGET_KEYS, LIBRARY

RUNGS = ("E0", "E1", "E2", "E3")
FEAT_HZ = 20.0
WHEEL_CHANNELS = ("ref_speed", "enc_speed", "duty", "current")
BODY_CHANNELS = ("cmd_vx", "cmd_vy", "cmd_wz", "gyro_z", "acc_x", "acc_y", "lidar_vx", "lidar_vy", "lidar_wz")
GEOM_CHANNELS = ("x", "y", "w_hat_x", "w_hat_y", "r_hat_x", "r_hat_y", "norm_p_over_Lc", "module_index_over_4")
# Target fit degree. 1 (default) = robust line fit: lowest variance under 30 Hz sample-hold + noise (RMS ~0.009 in rho on
# the stub sensor model) but it returns a parabola-weighted average of the speed over the window. 3 = smoothed endpoint
# difference: the exact window mean (bias < 0.005) but ~2x the noise. Either way the SAME definition is used for sim and real.
TARGET_FIT_DEG = 1
DEFAULT_PARAMS = Path(__file__).resolve().parents[1] / "params" / "module_measured.yaml"
_PRIM = {p.pid: p for p in LIBRARY}


def _defaults():
    import yaml
    with open(DEFAULT_PARAMS) as fh:
        p = yaml.safe_load(fh)
    return float(p["wheel"]["radius_m"]), float(p["truth"]["apriltag_latency_s"])


def as_assembly(a):
    """meta['assembly'] may be an Assembly or its dict."""
    from .assembly import Assembly
    if isinstance(a, Assembly):
        return a
    if isinstance(a, dict):
        d = dict(a)
        for k in ("wheel_pos", "w_hat", "r_hat", "wheel_module", "module_mass", "module_pos", "lidar"):
            d[k] = np.asarray(d[k])
        d.setdefault("meta", {})
        return Assembly(**{k: d[k] for k in Assembly.__dataclass_fields__ if k in d})
    raise TypeError(f"unsupported assembly type {type(a)}")


def wheel_ref_speed(cmd, asm, radius):
    """Firmware IK in torch, identical to Assembly.ik: cmd (...,3) -> (...,n) rad/s."""
    f = dict(dtype=cmd.dtype, device=cmd.device)
    pos = torch.as_tensor(asm.wheel_pos, **f)
    r = torch.as_tensor(asm.r_hat, **f)
    w = torch.as_tensor(asm.w_hat, **f)
    vx = cmd[..., 0:1] - cmd[..., 2:3] * pos[:, 1]
    vy = cmd[..., 1:2] + cmd[..., 2:3] * pos[:, 0]
    return (vx * r[:, 0] + vy * r[:, 1]) / (radius * (w * r).sum(-1))


def _segments(sched, pids):
    return [s for s in sched.segments if s.pid in pids]


def _bin_index(segs, t, hz=FEAT_HZ):
    """For each 20 Hz step: the 100 Hz indices of its bin (L, k) and the bin-end index (L,)."""
    dt = float(t[1] - t[0])
    T = len(t)
    per = max(1, int(round(1.0 / (hz * dt))))
    idx, pid_i, leg = [], [], []
    for s in segs:
        L = int(round((s.t1 - s.t0) * hz))
        i0 = int(round(s.t0 / dt))
        for k in range(L):
            end = i0 + (k + 1) * per
            idx.append([end - per + 1 + j for j in range(per)])
            pid_i.append(PIDS.index(s.pid))
            leg.append(s.leg)
    idx = torch.clamp(torch.tensor(idx, dtype=torch.long).reshape(-1, per), 0, T - 1)
    return idx, per, torch.tensor(pid_i, dtype=torch.long), torch.tensor(leg, dtype=torch.long)


def _pool(x, idx):
    """x (B,T,...) -> mean over each bin -> (B,L,...); NaN-aware (all-NaN bin -> NaN)."""
    g = x[:, idx]                                                             # (B,L,k,...)
    return g.nanmean(dim=2) if torch.is_floating_point(g) else g.float().mean(2)


def _nan0(x):
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _robust_slope(tt, s, m, half=None, iters=4, k_huber=1.345, deg=1):
    """Window-mean rate of s over [-half, half] (tt centred at 0): robust (Huber IRLS) polynomial fit
    s ~ sum_k c_k t^k per (B,K,C), returning (p(half) - p(-half)) / (2 half) = c1 + c3 half^2 (+ ...). This is the
    smoothed endpoint difference, i.e. the window-average rate even when the speed is not constant (settling, yaw drift),
    using every frame instead of two noisy endpoints. A plain line fit would instead return a parabola-weighted average.
    tt (K,L), s (B,K,L,C), m (B,K,L) bool, half (K,). Returns (B,K,C); NaN where the fit is degenerate."""
    tt = tt[None, :, :, None].expand_as(s)
    X = torch.stack([tt ** k for k in range(deg + 1)], -1)                   # (B,K,L,C,P)
    w = m.float()[..., None].expand_as(s).clone()
    P = deg + 1
    eye = torch.eye(P, dtype=s.dtype, device=s.device)
    coef = None
    for it in range(iters + 1):
        XtW = X * w[..., None]
        A = torch.einsum("bklcp,bklcq->bkcpq", XtW, X) + 1e-10 * eye
        rhs = torch.einsum("bklcp,bklc->bkcp", XtW, s)
        coef = torch.linalg.solve(A, rhs[..., None])[..., 0]                  # (B,K,C,P)
        if it == iters:
            break
        res = s - torch.einsum("bklcp,bkcp->bklc", X, coef)
        absr = torch.where(m[..., None], res.abs(), torch.full_like(res, float("nan")))
        mad = absr.nanmedian(dim=2).values
        scale = (1.4826 * mad).clamp_min(1e-6)[:, :, None]
        u = torch.nan_to_num(absr, nan=0.0) / scale
        hub = torch.where(u <= k_huber, torch.ones_like(u), k_huber / u.clamp_min(1e-12))
        w = m.float()[..., None] * hub
    if half is None:
        half = tt[0, :, :, 0].abs().amax(-1)
    h = half.to(s)[None, :, None]
    rate = sum(coef[..., k] * h ** (k - 1) for k in range(1, P, 2))           # odd terms of (p(h) - p(-h)) / 2h
    nfr = m.sum(2)[..., None].expand(coef.shape[:3])
    return torch.where(nfr > P + 1, rate, torch.full_like(rate, float("nan")))


def body_displacement(truth_pose):
    """Cumulative body-frame displacement s (B,T,3) from a world pose (B,T,3); NaN frames are held."""
    p = truth_pose.clone()
    # hold the last finite value over NaN gaps (lost tag) so increments there are zero
    bad = ~torch.isfinite(p).all(-1)
    if bad.any():
        T = p.shape[1]
        ar = torch.arange(T, device=p.device).expand(p.shape[0], T)
        last = torch.where(~bad, ar, torch.zeros_like(ar)).cummax(1).values
        p = torch.gather(p, 1, last[..., None].expand_as(p))
        p = _nan0(p)
    dp = p[:, 1:] - p[:, :-1]
    dth = torch.atan2(torch.sin(dp[..., 2]), torch.cos(dp[..., 2]))
    th = p[:, :-1, 2]
    c, s_ = torch.cos(th), torch.sin(th)
    bx = c * dp[..., 0] + s_ * dp[..., 1]
    by = -s_ * dp[..., 0] + c * dp[..., 1]
    inc = torch.stack([bx, by, dth], -1)
    s = torch.cat([torch.zeros_like(inc[:, :1]), inc.cumsum(1)], 1)
    new = torch.cat([torch.ones_like(bad[:, :1]), (dp.abs().sum(-1) > 0)], 1) & ~bad
    return s, new


def targets(raw, asm, truth_latency, min_frames=4, fit_deg=None):
    """Evaluator-only capability ratios y (B,14) and drift (B,14). Uses only truth_pose and the schedule's cmd twists."""
    t = raw["t"]
    tp = raw["truth_pose"].float()
    B, T = tp.shape[:2]
    dt = float(t[1] - t[0])
    sched = raw["meta"]["schedule"]
    wins = {f"{pid}{'+' if leg > 0 else '-'}": (ta, tb, tw) for pid, leg, ta, tb, tw in sched.windows()}
    s, new = body_displacement(tp)
    keys = [k for k in TARGET_KEYS if k in wins]
    y = torch.full((B, len(TARGET_KEYS)), float("nan"), device=tp.device)
    drift = torch.full_like(y, float("nan"))
    if not keys:
        return y, drift
    idx_l, Lmax = [], 0
    for k in keys:
        ta, tb, _ = wins[k]
        i0 = int(math.ceil((ta + truth_latency) / dt - 1e-6))
        i1 = int(math.floor((tb + truth_latency) / dt + 1e-6))
        idx_l.append((i0, i1))
        Lmax = max(Lmax, i1 - i0 + 1)
    K = len(keys)
    idx = torch.zeros(K, Lmax, dtype=torch.long)
    inwin = torch.zeros(K, Lmax, dtype=torch.bool)
    for j, (i0, i1) in enumerate(idx_l):
        r = torch.arange(i0, i0 + Lmax)
        idx[j] = r.clamp(0, T - 1)
        inwin[j] = (r <= i1) & (r >= 0) & (r < T)
    idx, inwin = idx.to(tp.device), inwin.to(tp.device)
    sw = s[:, idx]                                                            # (B,K,L,3)
    m = new[:, idx] & inwin[None]
    few = m.sum(2) < min_frames                                               # noise-free / stationary: use every sample
    m = torch.where(few[..., None], inwin[None].expand_as(m), m)
    centre = torch.tensor([(i0 + i1) / 2.0 for i0, i1 in idx_l], device=tp.device)
    tt = (idx.float() - centre[:, None]) * dt
    half = torch.tensor([(i1 - i0) * dt / 2.0 for i0, i1 in idx_l], device=tp.device)
    xi = _robust_slope(tt, sw, m, half, deg=TARGET_FIT_DEG if fit_deg is None else fit_deg)                                             # (B,K,3)
    Lc = float(asm.char_length())
    Wd = torch.tensor([1.0, 1.0, Lc * Lc], device=tp.device)
    cmdw = torch.tensor(np.stack([wins[k][2] for k in keys]), dtype=torch.float32, device=tp.device)   # (K,3)
    nc2 = (cmdw * cmdw * Wd).sum(-1)                                          # (K,)
    rho = (xi * cmdw * Wd).sum(-1) / nc2
    res = xi - rho[..., None] * cmdw
    dr = torch.sqrt((res * res * Wd).sum(-1) / nc2)
    cols = torch.tensor([TARGET_KEYS.index(k) for k in keys], device=tp.device)
    y[:, cols] = rho
    drift[:, cols] = dr
    return y, drift


@torch.no_grad()
def featurize(raw, rung="E2", pids=("P1", "P5"), radius=None, truth_latency=None, with_targets=True):
    """RawTrial -> Feat (CONTRACTS C2). Vectorised over the batch (trials of one assembly)."""
    if rung not in RUNGS:
        raise ValueError(f"rung {rung!r} not in {RUNGS}")
    meta = raw["meta"]
    asm = as_assembly(meta["assembly"])
    d_rad, d_lat = _defaults() if (radius is None and "wheel_radius" not in meta) or \
        (truth_latency is None and "truth_latency_s" not in meta) else (None, None)
    radius = float(radius if radius is not None else meta.get("wheel_radius", d_rad))
    truth_latency = float(truth_latency if truth_latency is not None else meta.get("truth_latency_s", d_lat))
    ri = RUNGS.index(rung)
    t = raw["t"]
    dev = raw["duty"].device
    B, T, n = raw["duty"].shape
    sched = meta["schedule"]
    pids = tuple(p for p in PIDS if p in pids)
    segs = _segments(sched, pids)
    if not segs:
        raise ValueError(f"none of the primitives {pids} is in the trial's schedule")
    idx, per, seg_pid, seg_leg = _bin_index(segs, t.cpu())
    idx = idx.to(dev)
    dt = float(t[1] - t[0])
    cpr = float(meta["cpr"])

    # ---- per-wheel channels
    cmd = raw["cmd"].float()
    cmd_b = _pool(cmd, idx)                                                   # (B,L,3)
    ref = wheel_ref_speed(cmd_b, asm, radius)                                 # (B,L,n)
    cnt = raw["counts"]
    end = idx[:, -1]
    start = (end - per).clamp_min(0)
    enc = (cnt[:, end] - cnt[:, start]).float() * (2 * math.pi / cpr) / (per * dt)   # (B,L,n)
    duty = _pool(raw["duty"].float(), idx)
    cur = _pool(raw["current"].float(), idx) if ri >= 3 else torch.zeros_like(duty)
    wheel_x = torch.stack([ref, enc, duty, cur], -1).permute(0, 2, 1, 3).contiguous()   # (B,n,L,4)
    wmask = torch.ones(B, n, 4, device=dev)
    if ri >= 3:
        wmask[..., 3] = torch.isfinite(raw["current"].float()).any(1).float()
    else:
        wmask[..., 3] = 0
    wheel_x = _nan0(wheel_x) * wmask[:, :, None, :]

    # ---- body channels
    L = idx.shape[0]
    body = torch.zeros(B, L, 9, device=dev)
    bmask = torch.zeros(9, device=dev)
    body[..., 0:3] = cmd_b
    bmask[0:3] = 1
    if ri >= 1:
        body[..., 3] = _pool(raw["gyro_z"].float()[..., None], idx)[..., 0]
        body[..., 4:6] = _pool(raw["acc"].float(), idx)
        bmask[3:6] = 1
    if ri >= 2:
        lt = raw["lidar_twist"].float()
        if torch.isfinite(lt).any():
            body[..., 6:9] = _pool(lt, idx)
            bmask[6:9] = 1
    body = _nan0(body) * bmask

    # ---- geometry (assembly frame, origin at the nominal CG)
    cg = asm.cg()
    Lc = asm.char_length()
    p = asm.wheel_pos - cg
    geom = np.concatenate([p, asm.w_hat, asm.r_hat, np.linalg.norm(p, axis=1, keepdims=True) / Lc,
                           np.asarray(asm.wheel_module, float)[:, None] / 4.0], 1)
    geom = torch.tensor(geom, dtype=torch.float32, device=dev).expand(B, n, 8).clone()

    pid_mask = torch.tensor([1.0 if q in pids else 0.0 for q in PIDS], device=dev)
    feat = dict(wheel_x=wheel_x, wheel_mask=wmask, wheel_geom=geom, body_x=body, body_mask=bmask, pid_mask=pid_mask,
                n_wheels=torch.full((B,), n, dtype=torch.long, device=dev), seg_pid=seg_pid.to(dev), seg_leg=seg_leg.to(dev),
                rung=rung, pids=pids, source=meta.get("source", "SIMULATION"), trial_ids=list(meta.get("trial_ids", [])),
                assembly_name=asm.name, char_length=float(Lc))
    if with_targets:
        y, drift = targets(raw, asm, truth_latency)
        feat["y"], feat["drift"] = y.to(dev), drift.to(dev)
        feat["labels"] = meta.get("labels")
    return feat


# ----------------------------------------------------------------------------- normalisation for learners
class Normalizer:
    """Per-channel mean/std over masked entries of wheel_x and body_x (and the y targets). JSON save/load.

    `fit(feats)` takes an iterable of Feat dicts; `apply(feat)` returns a shallow copy with normalised wheel_x / body_x
    (masked channels stay zero)."""

    def __init__(self, wheel_mean=None, wheel_std=None, body_mean=None, body_std=None, y_mean=None, y_std=None):
        self.wheel_mean, self.wheel_std = wheel_mean, wheel_std
        self.body_mean, self.body_std = body_mean, body_std
        self.y_mean, self.y_std = y_mean, y_std

    @staticmethod
    def _acc(x, m, s1, s2, cnt):
        C = x.shape[-1]
        xf = x.reshape(-1, C).double()
        mf = m.reshape(-1, C).double()
        s1 += (xf * mf).sum(0)
        s2 += (xf * xf * mf).sum(0)
        cnt += mf.sum(0)

    def fit(self, feats):
        Cw, Cb = len(WHEEL_CHANNELS), len(BODY_CHANNELS)
        w = [torch.zeros(Cw, dtype=torch.float64) for _ in range(3)]
        b = [torch.zeros(Cb, dtype=torch.float64) for _ in range(3)]
        yv = [torch.zeros(len(TARGET_KEYS), dtype=torch.float64) for _ in range(3)]
        for f in feats:
            wx = f["wheel_x"].detach().cpu()
            wm = f["wheel_mask"].detach().cpu()[:, :, None, :].expand_as(wx)
            self._acc(wx, wm, *w)
            bx = f["body_x"].detach().cpu()
            bm = f["body_mask"].detach().cpu().expand_as(bx)
            self._acc(bx, bm, *b)
            if "y" in f:
                y = f["y"].detach().cpu()
                self._acc(_nan0(y), torch.isfinite(y).float(), *yv)

        def fin(s1, s2, c):
            c = c.clamp_min(1)
            mu = s1 / c
            sd = (s2 / c - mu * mu).clamp_min(0).sqrt()
            sd = torch.where(sd > 1e-8, sd, torch.ones_like(sd))
            return mu.float(), sd.float()
        self.wheel_mean, self.wheel_std = fin(*w)
        self.body_mean, self.body_std = fin(*b)
        self.y_mean, self.y_std = fin(*yv)
        return self

    def apply(self, feat):
        f = dict(feat)
        dev = f["wheel_x"].device
        f["wheel_x"] = (f["wheel_x"] - self.wheel_mean.to(dev)) / self.wheel_std.to(dev) * f["wheel_mask"][:, :, None, :]
        f["body_x"] = (f["body_x"] - self.body_mean.to(dev)) / self.body_std.to(dev) * f["body_mask"]
        return f

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ("wheel_mean", "wheel_std", "body_mean", "body_std", "y_mean", "y_std")} | \
            {"wheel_channels": list(WHEEL_CHANNELS), "body_channels": list(BODY_CHANNELS), "target_keys": list(TARGET_KEYS)}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(**{k: torch.tensor(d[k], dtype=torch.float32)
                      for k in ("wheel_mean", "wheel_std", "body_mean", "body_std", "y_mean", "y_std")})

"""I2: decision-relevant identifiability for modular mecanum assemblies (analytic side).

Notation (assembly frame, origin = assembly origin = the point the firmware IK rotates about):
  wheel i at p_i, rolling dir w_i, roller (traction) axis r_i, lateral n_i = R90 w_i.
  probe twist xi = (vx, vy, wz).  Contact velocity v_i = J_i xi, J_i = [[1,0,-y_i],[0,1,x_i]].
  drive row   g_i = J_i^T r_i = (r_x, r_y, x r_y - y r_x)     a_i(xi) = g_i . xi   (traction-axis speed demand)
  lateral row h_i = J_i^T n_i = (n_x, n_y, p_i . w_i)          b_i(xi) = h_i . xi   (cross-roller speed demand)
Excitation (Prop. 1): a drive fault (weak/coast/brake) on wheel i is excited by xi iff a_i(xi) != 0; a roller-lock on
wheel i is excited iff b_i(xi) != 0. In the quasi-static model an unexcited fault changes neither the signals nor the
capability under xi.
Also: a batched quasi-static (QS) equilibrium solver with the simulator's contact/motor laws, used as the planner's
consequence model and for the masking law.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from capxfer import probes
from capxfer.sim import OK, WEAK, COAST, BRAKE, ROLLER, G_EARTH, nominal as sim_nominal

NOM = dict(mu=0.8, s0=0.03, c_roller=2.25, crr=0.0175, tau_bd=0.75, v_batt=11.85, stall_scale=1.0, noload_scale=1.0,
           radius_scale=1.0)


# ----------------------------------------------------------------------------- geometry / excitation
def rows(asm):
    p, w, r = asm.wheel_pos, asm.w_hat, asm.r_hat
    n_hat = np.stack([-w[:, 1], w[:, 0]], 1)
    g = np.stack([r[:, 0], r[:, 1], p[:, 0] * r[:, 1] - p[:, 1] * r[:, 0]], 1)
    h = np.stack([n_hat[:, 0], n_hat[:, 1], p[:, 0] * n_hat[:, 1] - p[:, 1] * n_hat[:, 0]], 1)
    return g, h


def probe_twists(asm, pids=probes.PIDS):
    """{pid: outbound hold twist} with the schedule's yaw scaling."""
    sch = probes.Schedule(asm.char_length(), pids)
    return {s.pid: s.twist for s in sch.segments if s.leg > 0}


def probe_time(pids):
    return sum(2 * probes.leg_duration(p) for p in probes.LIBRARY if p.pid in pids)


def excitation(asm, pids=probes.PIDS):
    """-> E (n, 5, P): |speed demand| (m/s) of each (wheel, mode) channel under each probe (mode 0 = 0)."""
    g, h = rows(asm)
    tw = probe_twists(asm, pids)
    P = [tw[p] for p in pids]
    E = np.zeros((asm.n, 5, len(P)))
    for k, xi in enumerate(P):
        a, b = np.abs(g @ xi), np.abs(h @ xi)
        E[:, WEAK, k] = E[:, COAST, k] = E[:, BRAKE, k] = a
        E[:, ROLLER, k] = b
    return E


def danger_closed_form(asm, eta=0.03):
    """Closed form for P1+P5 on 90-deg-rotated X-pattern modules (Prop. 3): the only channels P1+P5 can miss are
    roller locks on wheels with w = +-x (module yaw even) and |p . w| * wz < eta."""
    tw = probe_twists(asm, ("P5",))["P5"]
    even = np.abs(asm.w_hat[:, 0]) > 0.5
    return even & (np.abs((asm.wheel_pos * asm.w_hat).sum(1)) * abs(tw[2]) < eta)


# ----------------------------------------------------------------------------- normal loads (as sim._prepare)
def normal_loads(asm, payload=None):
    mass = asm.mass
    cg = asm.cg()
    if payload is not None:
        pm, pp = payload
        cg = (cg * mass + np.asarray(pp) * pm) / (mass + pm)
        mass = mass + pm
    A = np.concatenate([np.ones((1, asm.n)), asm.wheel_pos.T], 0)
    b = np.array([1.0, cg[0], cg[1]]) * mass * G_EARTH
    N = A.T @ np.linalg.solve(A @ A.T, b)
    N = np.clip(N, 0.02 * G_EARTH, None)
    return N * mass * G_EARTH / N.sum()


# ----------------------------------------------------------------------------- quasi-static solver
def qs_solve(asm, params, modes, gs, xis, dom=None, N=None, iters=15, device="cpu", mu_factor=None):
    """Quasi-static equilibrium of the assembly under constant commanded twists.
    modes, gs: (B, n); xis: (K, 3) commanded twists. Returns xi_real (B, K, 3), omega (B, K, n), duty (B, K, n).
    Driven wheels track the IK reference (PI integral) unless the duty saturates; passive wheels obey the
    back-drive / brake torque law. Body inertia and the rotating-frame terms are dropped (hold windows)."""
    d = {**NOM, **(dom or {})}
    nom = sim_nominal(params)
    f64 = dict(dtype=torch.float64, device=device)
    modes = torch.as_tensor(modes, device=device)
    gs = torch.as_tensor(gs, **f64)
    B, n = modes.shape
    K = len(xis)
    xis = torch.as_tensor(np.asarray(xis), **f64)
    pos = torch.as_tensor(asm.wheel_pos, **f64)
    w = torch.as_tensor(asm.w_hat, **f64)
    r = torch.as_tensor(asm.r_hat, **f64)
    f_hat = torch.stack([-r[:, 1], r[:, 0]], -1)
    n_hat = torch.stack([-w[:, 1], w[:, 0]], -1)
    roller = (modes == ROLLER)[..., None]
    d1 = torch.where(roller, w, r)                                       # (B,n,2)
    d2 = torch.where(roller, n_hat, f_hat)
    rad = nom["radius"] * d["radius_scale"]
    c1 = rad * (w * d1).sum(-1)
    c2 = rad * (w * d2).sum(-1)
    lever = lambda dd: torch.stack([dd[..., 0], dd[..., 1], pos[:, 0] * dd[..., 1] - pos[:, 1] * dd[..., 0]], -1)
    a1, a2 = lever(d1), lever(d2)                                        # (B,n,3)
    if N is None:
        N = normal_loads(asm)
    N = torch.as_tensor(np.asarray(N), **f64).expand(B, n)
    mu = d["mu"] * (torch.as_tensor(mu_factor, **f64) if mu_factor is not None else 1.0)
    s0, c_r, crr = d["s0"], d["c_roller"], d["crr"]
    tau_s, om0 = nom["tau_s"] * d["stall_scale"], nom["omega0"] * d["noload_scale"]
    vb = d["v_batt"] / nom["v0"]
    passive = (modes == COAST) | (modes == BRAKE)
    brake = modes == BRAKE
    gw = torch.where(modes == WEAK, gs, torch.ones_like(gs))
    # IK reference with nominal geometry
    denom = nom["radius"] * (w * r).sum(-1)
    vr = xis[:, None, :2] + xis[:, None, 2:3] * torch.stack([-pos[:, 1], pos[:, 0]], -1)   # (K,n,2)
    om_ref = (vr * r).sum(-1) / denom                                   # (K,n)

    # batch over (B,K)
    BB = B * K
    rep = lambda t: t[:, None].expand(B, K, *t.shape[1:]).reshape(BB, *t.shape[1:])
    A1, A2, C1, C2, NN = rep(a1), rep(a2), rep(c1), rep(c2), rep(N)
    MU = rep(mu.expand(B, n)) if torch.is_tensor(mu) and mu.dim() == 2 else mu
    FR2 = rep(roller[..., 0])
    PAS, BRK, GW = rep(passive), rep(brake), rep(gw)
    OMR = om_ref[None].expand(B, K, n).reshape(BB, n)
    XIC = xis[None].expand(B, K, 3).reshape(BB, 3)

    S0 = [s0]

    def forces(xi, om):
        s1 = (A1 * xi[:, None]).sum(-1) - C1 * om
        s2 = (A2 * xi[:, None]).sum(-1) - C2 * om
        f1 = -MU * NN * torch.tanh(s1 / S0[0])
        f2 = torch.where(FR2, -MU * NN * torch.tanh(s2 / S0[0]), -c_r * s2)
        return f1, f2

    def resid(x, sat):
        xi, om = x[:, :3], x[:, 3:]
        f1, f2 = forces(xi, om)
        body = (f1[..., None] * A1 + f2[..., None] * A2).sum(1)
        tau_c = f1 * C1 + f2 * C2 + crr * NN * rad * torch.tanh(om / 0.5)   # load torque the motor must supply
        tau_pas = -d["tau_bd"] * torch.tanh(om / 0.3) - BRK * (tau_s / om0) * om
        tau_sat = tau_s * (sat * GW * vb - om / om0)
        wheel = torch.where(PAS, tau_pas - tau_c, torch.where(sat != 0, tau_sat - tau_c, om - OMR))
        return torch.cat([body, wheel], 1), tau_c

    x = torch.cat([XIC, torch.where(PAS, torch.zeros_like(OMR), OMR)], 1).clone()
    sat = torch.zeros(BB, n, **f64)
    for outer, s0e in enumerate([30 * s0, 10 * s0, 3 * s0, s0, s0, s0, s0]):   # homotopy in the slip scale
        S0[0] = s0e
        for it in range(iters):
            F, _ = resid(x, sat)
            hstep = 1e-6
            J = torch.empty(BB, 3 + n, 3 + n, **f64)
            for j in range(3 + n):
                xp = x.clone()
                xp[:, j] += hstep
                J[:, :, j] = (resid(xp, sat)[0] - F) / hstep
            dx = torch.linalg.solve(J + 1e-10 * torch.eye(3 + n, **f64), -F[..., None])[..., 0]
            dx = torch.nan_to_num(dx)
            nrm = lambda FF: (FF * FF).sum(1)
            f0 = nrm(F)
            best_x, best_f = x, f0
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.03, 0.01):
                xt = x + alpha * dx
                ft = nrm(resid(xt, sat)[0])
                better = ft < best_f
                best_x = torch.where(better[:, None], xt, best_x)
                best_f = torch.where(better, ft, best_f)
            x = best_x
            if F.abs().max() < 1e-9:
                break
        _, tau_c = resid(x, sat)
        om = x[:, 3:]
        u_req = (tau_c / tau_s + om / om0) / (GW * vb)
        new = torch.where(~PAS & (u_req.abs() > 1.0), torch.sign(u_req), torch.zeros_like(u_req))
        new = torch.where(sat != 0, sat, new)                            # once saturated stay saturated
        if torch.equal(new, sat) and s0e == s0 and outer >= 3:
            break
        sat = new
    Fres, tau_c = resid(x, sat)
    qs_solve.last_resid = Fres.abs().amax(1).reshape(B, K).numpy()
    om = x[:, 3:]
    duty = torch.where(PAS, torch.zeros_like(om), torch.clamp((tau_c / tau_s + om / om0) / (GW * vb), -1, 1))
    return (x[:, :3].reshape(B, K, 3).numpy(), om.reshape(B, K, n).numpy(), duty.reshape(B, K, n).numpy())


def rho_of(xi_real, xi_cmd, Lc):
    W = np.array([1.0, 1.0, Lc * Lc])
    xi_cmd = np.asarray(xi_cmd)
    return (xi_real * xi_cmd * W).sum(-1) / (xi_cmd * xi_cmd * W).sum(-1)


def legs(asm):
    """14 commanded twists in TARGET_KEYS order."""
    tw = probe_twists(asm)
    return np.array([s * tw[p.pid] for p in probes.LIBRARY for s in (+1, -1)])


def lc_cg(asm):
    return asm.char_length()


# ----------------------------------------------------------------------------- linear masking law (Prop. 2)
def leverage_deficit(asm, i, xi, N=None, mu=NOM["mu"], s0=NOM["s0"], c_r=NOM["c_roller"]):
    """Linear-regime (small-slip) capability deficit of a single locked (passive) wheel i under twist xi:
    Delta rho = -k_i a_i(xi) <H^-1 g_i, xi>_W / |xi|_W^2,  H = sum_j k_j g_j g_j^T + c_r sum_j l_j l_j^T,
    k_j = mu N_j / s0 (traction stiffness), l_j the free-axis lever. Also returns the absorbed-regime
    (faulty wheel saturated at mu N_i) deficit, s0-scaled: -mu N_i sign(a_i) <H_h^-1 g_i, xi>_W/|xi|_W^2."""
    g, _ = rows(asm)
    p, r = asm.wheel_pos, asm.r_hat
    f = np.stack([-r[:, 1], r[:, 0]], 1)
    l = np.stack([f[:, 0], f[:, 1], p[:, 0] * f[:, 1] - p[:, 1] * f[:, 0]], 1)
    N = normal_loads(asm) if N is None else N
    k = mu * N / s0
    H = (k[:, None, None] * g[:, :, None] * g[:, None, :]).sum(0) + c_r * (l[:, :, None] * l[:, None, :]).sum(0)
    Lc = asm.char_length()
    W = np.diag([1.0, 1.0, Lc * Lc])
    a = g[i] @ xi
    nx = xi @ W @ xi
    lin = -k[i] * a * (xi @ W @ np.linalg.solve(H, g[i])) / nx
    Hh = H - k[i] * np.outer(g[i], g[i])
    absorbed = -mu * N[i] * np.sign(a) * (xi @ W @ np.linalg.solve(Hh, g[i])) / nx
    return lin, absorbed

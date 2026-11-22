"""Batched GPU simulator of a docked mecanum assembly running the probe library.

Model (stated simplifications, all randomised in training):
* Planar rigid body (assembly assumed rigid: magnets + latches), body-frame Newton-Euler with velocity-product terms.
* Static normal loads: least-norm solution of vertical force and roll/pitch moment balance about the CG (plus payload).
* Each wheel has two contact channels. Healthy mecanum: a friction channel along the contact-roller axis (coupled to
  wheel spin) and a viscous roller-bearing channel along the free direction. Roller-locked wheel: friction channels
  along the rolling and the lateral directions (box friction instead of an ellipse). Friction: -mu N tanh(s / s0).
* Motor: linear torque-speed curve at the wheel, tau = tau_s (V/V0 - omega/omega0), effective inertia J_w + J_r N^2.
* Firmware (docs/TEAM_T0_HANDOFF.md §2): per-wheel PI velocity loop on ideal-IK references with actuation latency,
  run at params firmware.vel_loop_hz (MR: discrete ZOH loop with encoder-count speed when slower than 1/dt);
  weak(g) multiplies the clamped duty by g; coast = gearbox back-drive friction only; brake = winding short damping.
* Integration: linearly-implicit Euler on (twist, wheel speeds) with analytic Jacobians (block-structured Schur
  solve), default dt = 5 ms (validated against 1 ms, see docs/coordination/A.md). CUDA-graph replay on GPU.

Fault modes: 0 ok, 1 weak, 2 coast, 3 brake, 4 roller_locked.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .probes import Schedule

OK, WEAK, COAST, BRAKE, ROLLER = 0, 1, 2, 3, 4
MODE_NAMES = ("ok", "weak", "coast", "brake", "roller_locked")
G_EARTH = 9.81


def nominal(params):
    """Nominal scalar physical constants from the parameter file."""
    m = params["motor"]
    fw = params["firmware"]
    w0 = m["no_load_rpm_12v"] * 2 * math.pi / 60.0
    r_nom = float(params["wheel"]["radius_m"])
    # MR: CONFIRMED firmware (team 2026-11-21), per wheel on surface speed v = omega r_nom (m/s):
    #   duty = clamp(v_ref / ff_ref_speed_mps + kp_duty_per_mps e + ki_duty_per_m int(e dt), -1, 1)
    # expressed here per rad/s of wheel speed: k_ff = r_nom / ff_ref_speed_mps, kp = kp_duty_per_mps r_nom, ...
    kp = float(fw["kp_duty_per_mps"]) * r_nom if "kp_duty_per_mps" in fw else float(fw["kp_duty_per_radps"])
    ki = float(fw["ki_duty_per_m"]) * r_nom if "ki_duty_per_m" in fw else float(fw["ki_duty_per_rad"])
    if fw.get("ff_ref_speed_mps"):
        kff = r_nom / float(fw["ff_ref_speed_mps"])
    else:
        kff = float(fw.get("ff_duty_per_radps") or 0.0)
    return dict(
        radius=r_nom,
        tau_s=float(m["stall_torque_nm_12v"]), omega0=w0, v0=12.0, v_batt=float(params["battery"]["nominal_v"]),
        j_eff=float(params["wheel"]["inertia_wheel_kgm2"]) + float(m["rotor_inertia_kgm2"]) * float(m["gear_ratio"]) ** 2,
        kp=kp, ki=ki, latency=float(fw["actuation_latency_s"]),
        cpr=float(m["counts_per_output_rev"]),
        ctrl_hz=float(fw.get("vel_loop_hz") or 0.0),                     # MR: discrete velocity loop (0 = continuous)
        ff_duty_per_radps=kff,                                           # MR: reference feed-forward (duty per rad/s)
    )


# ----------------------------------------------------------------------------- domain randomisation
DR_DEFAULT = dict(
    mu=(0.6, 1.0), s0=(0.01, 0.05), c_roller=(0.5, 4.0), crr=(0.005, 0.03), tau_bd=(0.3, 1.2),
    latency=(0.02, 0.08), stall_scale=(0.8, 1.15), noload_scale=(0.9, 1.1), mass_scale=(0.93, 1.07),
    izz_scale=(0.8, 1.25), radius_scale=(0.97, 1.03), cg_off=0.015, kp_scale=(0.7, 1.4), ki_scale=(0.7, 1.4),
    v_batt=(11.1, 12.6), wheel_yaw_err=0.02,
)

# A2: wide training DR so plausible real floors / drivetrains fall inside it (integration dry run: a mu = 0.55 plant
# was outside DR_DEFAULT and coverage collapsed). Widened: mu, s0 (friction, the mu/s0 ridge is all sys-ID can
# identify), crr (x2 weakly identified), izz (-9 % sys-ID error), c_roller, latency, radius, battery, motor and PI
# gains, back-drive torque. Use as datagen.generate(dr=sim.DR_WIDE) or `--dr wide`.
DR_WIDE = dict(
    DR_DEFAULT, mu=(0.3, 1.2), s0=(0.005, 0.08), c_roller=(0.3, 6.0), crr=(0.002, 0.05), tau_bd=(0.2, 1.5),
    latency=(0.01, 0.12), stall_scale=(0.7, 1.25), noload_scale=(0.85, 1.15), mass_scale=(0.9, 1.1),
    izz_scale=(0.7, 1.4), radius_scale=(0.95, 1.05), kp_scale=(0.5, 2.0), ki_scale=(0.5, 2.0), v_batt=(10.8, 12.6),
    cg_off=0.02, wheel_yaw_err=0.03,
)
# MR (2026-11-21, params MEASURED_V1 dd3f3d7e): DR centred on the measured robot. Every *_scale multiplies the measured
# params value, so the centre is the measurement; measured quantities are narrowed (mass +-5 %, geometry fixed: wheel
# radius only to the measured module spread +-1 %, cg offset as before); friction stays wide (mu 0.3-1.2: the tilt-board
# 0.47-0.52 is one floor, and a healthy single-module sys-ID cannot identify mu). Everything else as DR_WIDE except the
# (confirmed) controller gains, fixed below.
DR_MEASURED = dict(DR_WIDE, mass_scale=(0.95, 1.05), radius_scale=(0.99, 1.01), mu=(0.3, 1.2))
# MR (2026-11-21): the firmware is CONFIRMED by the team (params firmware: ff_ref_speed_mps / kp_duty_per_mps /
# ki_duty_per_m, 50 Hz), so the controller gains are not randomised in the measured DR (kp/ki scale fixed at 1); only the
# physical drivetrain (stall / no-load scale, actuation latency, friction, rolling / roller resistance, back-drive, inertia,
# battery) stays randomised. (An earlier provisional firmware-structure DR, feed-forward on/off and gain scales, was
# dropped and its data/models discarded.) Optional DR key ff_scale (multiplier on the firmware k_ff, p_ff_off share
# switched off) remains available for ablations only; no preset uses it.
DR_MEASURED.update(kp_scale=(1.0, 1.0), ki_scale=(1.0, 1.0))
# D17 (integrator, 2026-11-21): until the team's free-speed (hook check) measurement arrives, the motor no-load speed is
# randomised wide enough that the no-load wheel SURFACE speed at the nominal 11.1 V covers 0.45-0.65 m/s
# (0.0501 m x 10.47 rad/s x 11.1/12 = 0.485 m/s at scale 1 -> scale 0.9-1.35 gives 0.44-0.66 m/s).
DR_MEASURED.update(noload_scale=(0.9, 1.35))
DR_PRESETS = dict(default=DR_DEFAULT, wide=DR_WIDE, measured=DR_MEASURED)


def sample_domain(B, rng, dr=None, device="cpu"):
    dr = {**DR_DEFAULT, **(dr or {})}
    u = lambda k: torch.tensor(rng.uniform(*dr[k], size=B), dtype=torch.float32, device=device)
    d = {k: u(k) for k in ("mu", "s0", "c_roller", "crr", "tau_bd", "latency", "stall_scale", "noload_scale",
                           "mass_scale", "izz_scale", "radius_scale", "kp_scale", "ki_scale", "v_batt")}
    d["cg_dx"] = torch.tensor(rng.normal(0, dr["cg_off"], B), dtype=torch.float32, device=device)
    d["cg_dy"] = torch.tensor(rng.normal(0, dr["cg_off"], B), dtype=torch.float32, device=device)
    d["wheel_yaw_err_std"] = float(dr["wheel_yaw_err"])
    if "ff_scale" in dr:                       # MR: optional ablation DR on the firmware feed-forward gain
        ff = rng.uniform(*dr["ff_scale"], size=B)
        if float(dr.get("p_ff_off", 0.0)) > 0:                # firmware structure: feed-forward absent in this share
            ff = np.where(rng.uniform(size=B) < float(dr["p_ff_off"]), 0.0, ff)
        d["ff_scale"] = torch.tensor(ff, dtype=torch.float32, device=device)
    return d


def sample_faults(B, n, rng, p_healthy=0.25, p_double=0.2, mode_probs=(0.35, 0.15, 0.2, 0.3), g_range=(0.3, 0.9),
                  device="cpu"):
    """Training fault prior: healthy, one faulted wheel, or two faulted wheels."""
    mode = np.zeros((B, n), np.int64)
    g = np.ones((B, n), np.float32)
    r = rng.uniform(size=B)
    for b in range(B):
        if r[b] < p_healthy:
            continue
        k = 2 if (r[b] > 1 - p_double and n >= 2) else 1
        wheels = rng.choice(n, size=k, replace=False)
        for w in wheels:
            mode[b, w] = 1 + rng.choice(4, p=np.asarray(mode_probs) / np.sum(mode_probs))
            if mode[b, w] == WEAK:
                g[b, w] = rng.uniform(*g_range)
    return torch.tensor(mode, device=device), torch.tensor(g, device=device)


# ----------------------------------------------------------------------------- simulation
REC_KEYS = ("wheel_angle", "duty", "torque", "twist", "accel", "pose", "omega", "cmd", "tau_motor")
STATE_KEYS = ("xi", "om", "integ", "theta_w", "pose", "prev_xi", "u_hold", "th_ctrl")


def _prepare(asm, params, dom, mode, g, sched, dt, device, payload, mu_factor, seed, sheet=None):
    """All per-trial constants of the run (tensors on device). Pure function of its inputs."""
    nom = nominal(params)
    B, n = mode.shape
    f32 = dict(dtype=torch.float32, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    mode, g = mode.to(device), g.to(device).float()
    for k in dom:
        if torch.is_tensor(dom[k]):
            dom[k] = dom[k].to(device)

    pos = torch.tensor(asm.wheel_pos, **f32).expand(B, n, 2).clone()
    # small per-wheel mounting yaw error rotates rolling/roller directions
    yerr = torch.randn(B, n, generator=gen, **f32) * dom["wheel_yaw_err_std"]

    def rot(v):
        v = torch.tensor(v, **f32).expand(B, n, 2)
        c, s = torch.cos(yerr), torch.sin(yerr)
        return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)
    w_hat, r_hat = rot(asm.w_hat), rot(asm.r_hat)
    f_hat = torch.stack([-r_hat[..., 1], r_hat[..., 0]], -1)
    n_hat = torch.stack([-w_hat[..., 1], w_hat[..., 0]], -1)
    roller = mode == ROLLER
    d1 = torch.where(roller[..., None], w_hat, r_hat)
    d2 = torch.where(roller[..., None], n_hat, f_hat)
    radius = nom["radius"] * dom["radius_scale"]                      # (B,)
    c1 = radius[:, None] * (w_hat * d1).sum(-1)                       # (B,n)
    c2 = radius[:, None] * (w_hat * d2).sum(-1)

    # mass properties and normal loads
    mass = asm.mass * dom["mass_scale"]
    izz = asm.izz() * dom["izz_scale"] * dom["mass_scale"]
    cg = torch.tensor(asm.cg(), **f32).expand(B, 2).clone()
    cg[:, 0] += dom["cg_dx"]
    cg[:, 1] += dom["cg_dy"]
    if payload is not None:
        pm = payload["mass"].to(device).float()
        pp = torch.stack([payload["x"].to(device), payload["y"].to(device)], -1).float()
        izz = izz + pm * ((pp - cg) ** 2).sum(-1) * mass / (mass + pm)  # parallel-axis (point payload)
        cg = (cg * mass[:, None] + pp * pm[:, None]) / (mass + pm)[:, None]
        mass = mass + pm
    A = torch.cat([torch.ones(B, 1, n, **f32), pos.transpose(1, 2)], 1)     # (B,3,n)
    b = torch.stack([torch.ones(B, **f32), cg[:, 0], cg[:, 1]], 1)[..., None] * (mass * G_EARTH)[:, None, None]
    N = (A.transpose(1, 2) @ torch.linalg.solve(A @ A.transpose(1, 2), b))[..., 0]
    N = torch.clamp(N, min=0.02 * G_EARTH)
    N = N * (mass * G_EARTH / N.sum(1))[:, None]

    mu = dom["mu"][:, None]                                           # (B,1), or (B,n) with a per-wheel factor
    if mu_factor is not None:
        mf = mu_factor.to(device).float()
        mu = mu * (mf[:, None] if mf.dim() == 1 else mf)               # (B,) global F7 sheet or (B,n) patch

    def lever(d):
        return torch.stack([d[..., 0], d[..., 1], pos[..., 0] * d[..., 1] - pos[..., 1] * d[..., 0]], -1)

    steps = int(math.ceil(sched.T / dt))
    tgrid = np.arange(steps + 2) * dt
    pos_nom = torch.tensor(asm.wheel_pos, **f32)
    r_nom = torch.tensor(asm.r_hat, **f32)
    w_nom = torch.tensor(asm.w_hat, **f32)
    weak, coast, brake = mode == WEAK, mode == COAST, mode == BRAKE
    driven = ~(coast | brake)
    C = dict(
        pos=pos, d1=d1, d2=d2, c1=c1, c2=c2, a1=lever(d1), a2=lever(d2), N=N, mu=mu.expand(B, n).contiguous(),
        s0=dom["s0"][:, None], c_r=dom["c_roller"][:, None], crr=dom["crr"][:, None], fric2=roller,
        tau_s=(nom["tau_s"] * dom["stall_scale"])[:, None], omega0=(nom["omega0"] * dom["noload_scale"])[:, None],
        vb=(dom["v_batt"] / nom["v0"])[:, None], kp=(nom["kp"] * dom["kp_scale"])[:, None],
        ki=(nom["ki"] * dom["ki_scale"])[:, None], tau_bd=dom["tau_bd"][:, None], radius=radius[:, None],
        mass=mass, izz=izz, j_eff=torch.full((B, n), nom["j_eff"], **f32),
        tw_table=torch.tensor(sched.twist(tgrid), **f32), pos_nom=pos_nom, r_nom=r_nom,
        denom=nom["radius"] * (w_nom * r_nom).sum(-1),                 # firmware IK uses nominal geometry
        lat_idx=dom["latency"] / dt,
        weak=weak, coast=coast, brake=brake, driven=driven, passive=coast | brake,
        gw=torch.where(weak, g, torch.ones_like(g)),
        steps_f=torch.tensor(float(steps), **f32), dt=torch.tensor(float(dt), **f32),
    )
    # MR: discrete firmware velocity loop at params firmware.vel_loop_hz (MEASURED 50 Hz). Between loop ticks the duty is
    # held (ZOH); at a tick the wheel speed is the encoder count difference since the previous tick (cpr counts/rev).
    # A loop rate >= 1/dt (e.g. the old 200 Hz estimate at dt = 5 ms) keeps the original continuous-PI step exactly.
    ce = int(round(1.0 / (nom["ctrl_hz"] * dt))) if nom["ctrl_hz"] > 0 else 1
    C["zoh"] = ce > 1
    if C["zoh"]:
        C["ctrl_every"] = torch.tensor(float(ce), **f32)
        C["t_ctrl"] = torch.tensor(float(ce * dt), **f32)
        C["cnt_per_rad"] = torch.tensor(nom["cpr"] / (2 * math.pi), **f32)
    ffs = dom.get("ff_scale")                                          # optional ablation multiplier (default 1)
    kff = nom["ff_duty_per_radps"] * (ffs if ffs is not None else torch.ones(B, **f32))
    C["kff"] = (kff * torch.ones(B, **f32))[:, None]
    if sheet is not None:                                              # world-fixed low-friction sheet (A2)
        sv = lambda k: torch.as_tensor(sheet[k]).to(device).float()
        C["sheet_n"] = torch.stack([sv("nx"), sv("ny")], -1)            # (B,2) unit normal, initial body frame
        C["sheet_c"] = sv("c")[:, None]                                # (B,1) offset: covered where n.p > c
        C["sheet_k"] = (1.0 - sv("factor"))[:, None]                   # (B,1) 1 - mu factor on the sheet
    return C, steps, nom, mode, g


def _init_state(B, n, device):
    z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=device)
    return dict(xi=z(B, 3), om=z(B, n), integ=z(B, n), theta_w=z(B, n), pose=z(B, 3), prev_xi=z(B, 3),
                u_hold=z(B, n), th_ctrl=z(B, n))


def _step(S, C, it):
    """One linearly-implicit Euler step. Pure tensor function (CUDA-graph and torch.compile friendly).
    `it` is a 0-dim float tensor step index. Returns (new_state, per-step outputs)."""
    xi, om, integ = S["xi"], S["om"], S["integ"]
    dt = C["dt"]
    t_idx = it - C["lat_idx"]                                          # (B,)
    i0 = torch.clamp(torch.floor(t_idx), torch.zeros_like(t_idx), C["steps_f"]).long()
    fr = (t_idx - i0.float()).clamp(0, 1)[:, None]
    pos_ok = (t_idx >= 0)[:, None]
    tw = C["tw_table"]
    if tw.dim() == 2:                                                  # one schedule for the batch
        tw0, tw1 = tw[i0], tw[i0 + 1]
    else:                                                              # per-trial schedules (simulate_multi)
        tw0 = torch.gather(tw, 1, i0[:, None, None].expand(-1, 1, 3))[:, 0]
        tw1 = torch.gather(tw, 1, (i0 + 1)[:, None, None].expand(-1, 1, 3))[:, 0]
    cmd = tw0 * (1 - fr) + tw1 * fr * pos_ok.float()
    cmd = torch.where(pos_ok, cmd, torch.zeros_like(cmd))
    pn = C["pos_nom"]
    vrx = cmd[:, 0:1] - cmd[:, 2:3] * pn[..., 1]
    vry = cmd[:, 1:2] + cmd[:, 2:3] * pn[..., 0]
    om_ref = (vrx * C["r_nom"][..., 0] + vry * C["r_nom"][..., 1]) / C["denom"]

    # firmware PI (anti-windup on the unscaled clamp)
    kp, ki, tau_s, omega0, vb = C["kp"], C["ki"], C["tau_s"], C["omega0"], C["vb"]
    driven, passive, brake = C["driven"], C["passive"], C["brake"]
    zero = torch.zeros_like(om)
    if C["zoh"]:                                                       # MR: discrete loop, encoder speed, ZOH duty
        upd = torch.remainder(it, C["ctrl_every"]) < 0.5               # 0-dim bool: a loop tick at this step
        k = C["cnt_per_rad"]
        om_meas = (torch.floor(S["theta_w"] * k) - torch.floor(S["th_ctrl"] * k)) / k / C["t_ctrl"]
        e = om_ref - om_meas
        u_raw = C["kff"] * om_ref + kp * e + ki * integ
        unsat = (u_raw.abs() < 1) | (torch.sign(e) != torch.sign(u_raw))
        integ = torch.where(upd & unsat & driven, integ + e * C["t_ctrl"], integ)
        u = torch.where(upd, torch.clamp(u_raw, -1, 1), S["u_hold"])
        th_ctrl = torch.where(upd, S["theta_w"], S["th_ctrl"])
        u_hold = u
        dfb = zero                                                     # held duty: no feedback within the step
    else:
        e = om_ref - om
        u_raw = C["kff"] * om_ref + kp * e + ki * integ
        u = torch.clamp(u_raw, -1, 1)
        unsat = (u_raw.abs() < 1) | (torch.sign(e) != torch.sign(u_raw))
        integ = torch.where(unsat & driven, integ + e * dt, integ)
        th_ctrl, u_hold = S["th_ctrl"], S["u_hold"]
        dfb = tau_s * vb * kp * C["gw"] * unsat.float()
    volt = u * C["gw"] * vb
    tau_em = torch.where(driven, tau_s * (volt - om / omega0), zero)
    dtau_m = torch.where(driven, -tau_s / omega0 - dfb, zero)
    tau_em = tau_em - torch.where(brake, tau_s / omega0 * om, zero)    # winding short damping
    dtau_m = dtau_m - torch.where(brake, tau_s / omega0, zero)
    thb = torch.tanh(om / 0.3)
    tau_m = tau_em - torch.where(passive, C["tau_bd"] * thb, zero)
    dtau_m = dtau_m - torch.where(passive, C["tau_bd"] / 0.3 * (1 - thb * thb), zero)
    # rolling resistance at the wheel
    N, radius, crr = C["N"], C["radius"], C["crr"]
    thr = torch.tanh(om / 0.5)
    rr = crr * N * radius * thr
    drr = crr * N * radius / 0.5 * (1 - thr * thr)

    # contact channels
    pos, d1, d2, c1, c2, a1, a2 = C["pos"], C["d1"], C["d2"], C["c1"], C["c2"], C["a1"], C["a2"]
    mu, s0, c_r, fric2 = C["mu"], C["s0"], C["c_r"], C["fric2"]
    if "sheet_n" in C:                                                 # wheel contact on the sheet (5 mm soft edge)
        pz = S["pose"]
        cth, sth = torch.cos(pz[:, 2:3]), torch.sin(pz[:, 2:3])
        wx = pz[:, 0:1] + cth * pos[..., 0] - sth * pos[..., 1]
        wy = pz[:, 1:2] + sth * pos[..., 0] + cth * pos[..., 1]
        on = torch.sigmoid((C["sheet_n"][:, 0:1] * wx + C["sheet_n"][:, 1:2] * wy - C["sheet_c"]) / 0.005)
        mu = mu * (1.0 - C["sheet_k"] * on)
    vcx = xi[:, 0:1] - xi[:, 2:3] * pos[..., 1]
    vcy = xi[:, 1:2] + xi[:, 2:3] * pos[..., 0]
    s1 = vcx * d1[..., 0] + vcy * d1[..., 1] - c1 * om
    s2 = vcx * d2[..., 0] + vcy * d2[..., 1] - c2 * om
    th1 = torch.tanh(s1 / s0)
    f1 = -mu * N * th1
    df1 = -mu * N * (1 - th1 * th1) / s0
    th2 = torch.tanh(s2 / s0)
    f2 = torch.where(fric2, -mu * N * th2, -c_r * s2)
    df2 = torch.where(fric2, -mu * N * (1 - th2 * th2) / s0, -c_r.expand_as(s2))

    mass, izz = C["mass"], C["izz"]
    W = (f1[..., None] * a1 + f2[..., None] * a2).sum(1)               # (B,3)
    Fb = torch.stack([W[:, 0] + mass * xi[:, 2] * xi[:, 1], W[:, 1] - mass * xi[:, 2] * xi[:, 0], W[:, 2]], 1)
    Fw = tau_m - (f1 * c1 + f2 * c2) - rr                              # (B,n)

    # Structured solve of (M - dt J) dx = dt F. J = [[J11, Q^T], [Q, diag(dg)]], Schur complement on the wheels.
    Q = -(df1 * c1)[..., None] * a1 - (df2 * c2)[..., None] * a2       # (B,n,3)
    dg = c1 * c1 * df1 + c2 * c2 * df2 + dtau_m - drr                  # (B,n) (<= 0)
    D = C["j_eff"] - dt * dg                                           # (B,n) > 0
    J11 = (df1[..., None, None] * a1[..., :, None] * a1[..., None, :]).sum(1) + \
          (df2[..., None, None] * a2[..., :, None] * a2[..., None, :]).sum(1)
    Qd = Q / D[..., None]
    Sm = -dt * J11 - dt * dt * (Qd[..., :, None] * Q[..., None, :]).sum(1)
    Mb = torch.stack([mass, mass, izz], 1)
    b2 = dt * Fw
    r = dt * Fb + dt * (Qd * b2[..., None]).sum(1)
    # explicit symmetric 3x3 solve (Cramer)
    s00 = Sm[:, 0, 0] + Mb[:, 0]
    s11 = Sm[:, 1, 1] + Mb[:, 1]
    s22 = Sm[:, 2, 2] + Mb[:, 2]
    s01, s02, s12 = Sm[:, 0, 1], Sm[:, 0, 2], Sm[:, 1, 2]
    k00 = s11 * s22 - s12 * s12
    k01 = s02 * s12 - s01 * s22
    k02 = s01 * s12 - s02 * s11
    k11 = s00 * s22 - s02 * s02
    k12 = s01 * s02 - s00 * s12
    k22 = s00 * s11 - s01 * s01
    det = s00 * k00 + s01 * k01 + s02 * k02
    x1 = torch.stack([k00 * r[:, 0] + k01 * r[:, 1] + k02 * r[:, 2],
                      k01 * r[:, 0] + k11 * r[:, 1] + k12 * r[:, 2],
                      k02 * r[:, 0] + k12 * r[:, 1] + k22 * r[:, 2]], 1) / det[:, None]
    x2 = (b2 + dt * (Q * x1[:, None, :]).sum(-1)) / D
    xi_new = xi + x1
    om_new = om + x2
    pose = S["pose"]
    th = pose[:, 2]
    pose = pose + dt * torch.stack([xi_new[:, 0] * torch.cos(th) - xi_new[:, 1] * torch.sin(th),
                                    xi_new[:, 0] * torch.sin(th) + xi_new[:, 1] * torch.cos(th), xi_new[:, 2]], 1)
    S_new = dict(xi=xi_new, om=om_new, integ=integ, theta_w=S["theta_w"] + om_new * dt, pose=pose,
                 prev_xi=S["prev_xi"], u_hold=u_hold, th_ctrl=th_ctrl)
    out = dict(u=u, tau_m=tau_m, tau_em=tau_em, cmd=cmd)
    return S_new, out


def _record(S, out, rec_dt):
    """Recorded quantities at a record instant; also returns the updated prev_xi."""
    xi = S["xi"]
    acc = (xi[:, :2] - S["prev_xi"][:, :2]) / rec_dt
    accel = torch.stack([acc[:, 0] - xi[:, 2] * xi[:, 1], acc[:, 1] + xi[:, 2] * xi[:, 0]], 1)
    return dict(wheel_angle=S["theta_w"], duty=out["u"], torque=out["tau_m"], twist=xi, accel=accel,
                pose=S["pose"], omega=S["om"], cmd=out["cmd"], tau_motor=out["tau_em"])


_COMPILED = {}


def _get_step(compile_):
    if not compile_:
        return _step
    if "step" not in _COMPILED:
        _COMPILED["step"] = torch.compile(_step, dynamic=False, fullgraph=True)
    return _COMPILED["step"]


def _run_block(S, C, it0, step_fn, rec_every, K, stage):
    """K records x rec_every steps from step index it0 (0-dim tensor); writes records into stage[key][:, j]."""
    rec_dt = C["dt"] * rec_every
    for j in range(K):
        for r in range(rec_every):
            S_new, out = step_fn(S, C, it0 + float(j * rec_every + r))
            for k in STATE_KEYS:
                S[k].copy_(S_new[k])
        rec = _record(S, out, rec_dt)
        for k in REC_KEYS:
            stage[k][:, j].copy_(rec[k])
        S["prev_xi"].copy_(S["xi"])


def _integrate(C, steps, B, n, dt, record_hz, device, engine, block_records, store_device, store_dtype, xi0):
    """Run the integrator over the full schedule for prepared constants C; returns (R, n_rec, rec_every)."""
    rec_every = int(round(1.0 / (record_hz * dt)))
    if abs(rec_every * dt * record_hz - 1.0) > 1e-6:
        raise ValueError("1/record_hz must be a multiple of dt")
    n_rec = steps // rec_every
    store_device = torch.device(store_device) if store_device is not None else device
    shapes = dict(wheel_angle=n, duty=n, torque=n, twist=3, accel=2, pose=3, omega=n, cmd=3, tau_motor=n)
    pin = store_device.type == "cpu" and device.type == "cuda"
    R = {k: torch.empty(B, n_rec, c, dtype=store_dtype, device=store_device, pin_memory=pin)
         for k, c in shapes.items()}
    K = max(1, min(block_records, n_rec))
    stage = {k: torch.zeros(B, K, c, dtype=torch.float32, device=device) for k, c in shapes.items()}
    S = _init_state(B, n, device)
    if xi0 is not None:                                                # optional initial body twist (B,3) for tests
        S["xi"].copy_(torch.as_tensor(xi0, dtype=torch.float32).to(device).expand(B, 3))
        S["prev_xi"].copy_(S["xi"])
    it0 = torch.zeros((), dtype=torch.float32, device=device)
    step_fn = _get_step(engine == "compile")
    n_blocks = -(-n_rec // K)

    if engine == "eager":
        replay = lambda: _run_block(S, C, it0, step_fn, rec_every, K, stage)
    else:
        S0 = {k: v.clone() for k, v in S.items()}
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):                                  # warm-up (also triggers compilation)
            _run_block(S, C, it0, step_fn, rec_every, 1, stage)
        torch.cuda.current_stream(device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_block(S, C, it0, step_fn, rec_every, K, stage)
        for k in S:
            S[k].copy_(S0[k])
        it0.zero_()
        replay = graph.replay
    for blk in range(n_blocks):
        replay()
        it0 += float(K * rec_every)
        k0 = blk * K
        k1 = min(n_rec, k0 + K)
        for k in REC_KEYS:
            R[k][:, k0:k1].copy_(stage[k][:, :k1 - k0], non_blocking=pin)
    if pin:
        torch.cuda.synchronize(device)
    return R, n_rec, rec_every


@torch.no_grad()
def simulate(asm, params, dom, mode, g, schedule=None, dt=0.005, record_hz=100, device="cuda", payload=None,
             mu_factor=None, seed=0, engine="auto", block_records=50, store_device=None, store_dtype=torch.float32, xi0=None,
             sheet=None):
    """Run the probe library for B environments of one assembly.

    payload: optional dict(mass=(B,), x=(B,), y=(B,)) placed on the assembly (out-of-model fault F6).
    mu_factor: optional multiplier on friction (out-of-model low-friction floor F7): (B,) for the whole floor, or
            (B,n) per wheel (a low-friction patch under some wheels; A2 addition).
    engine: "eager" (plain loop), "graph" (CUDA-graph replay of blocks of `block_records` records),
            "compile" (torch.compile'd step inside the CUDA graph), "auto" (graph on CUDA, eager on CPU).
    store_device / store_dtype: where/how the recorded streams are kept (default: device, float32); e.g. "cpu" to
            run very large B without holding the streams on the GPU.
    xi0: optional initial body twist (B,3) or (3,) (default zero; used by the energy test).
    sheet: optional world-fixed low-friction sheet (F7 patch, A2): dict(nx, ny, c, factor), each (B,). The floor
            half-plane nx*x + ny*y > c (world frame = assembly frame at t=0, origin at the assembly origin) has
            friction mu*factor; each wheel's mu follows its contact position over time (5 mm soft edge).
    Returns a dict of raw ground-truth streams at record_hz. Keys (shapes): t (T,), wheel_angle/duty/torque/omega/
    tau_motor (B,T,n), twist/pose/cmd (B,T,3), accel (B,T,2), mode (B,n), g (B,n), meta. `torque` is the net drive
    torque at the wheel (motor minus back-drive), `tau_motor` the electromagnetic part only (for current models)."""
    sched = schedule or Schedule(asm.char_length())
    B, n = mode.shape
    device = torch.device(device)
    if engine == "auto":
        engine = "graph" if device.type == "cuda" else "eager"
    if device.type != "cuda" and engine != "eager":
        raise ValueError("graph/compile engines need CUDA")
    C, steps, nom, mode, g = _prepare(asm, params, dom, mode, g, sched, dt, device, payload, mu_factor, seed, sheet)
    R, n_rec, rec_every = _integrate(C, steps, B, n, dt, record_hz, device, engine, block_records, store_device,
                                     store_dtype, xi0)
    store_device = R["wheel_angle"].device
    R["t"] = torch.arange(n_rec, dtype=torch.float32, device=store_device) * rec_every * dt
    R["mode"] = mode.to(store_device)
    R["g"] = g.to(store_device)
    R["meta"] = dict(n=n, B=B, record_hz=record_hz, cpr=nom["cpr"], tau_s=C["tau_s"][:, 0], schedule=sched,
                     asm=asm.name, assembly=asm, char_length=asm.char_length(), dt=dt, engine=engine, seed=seed,
                     ood="payload" if payload is not None else (
                         "low_mu" if (mu_factor is not None or sheet is not None) else "none"))
    return R


_SHARED = ("pos_nom", "r_nom", "denom", "tw_table")


@torch.no_grad()
def simulate_multi(asms, params, doms, modes, gs, dt=0.005, record_hz=100, device="cuda", payloads=None,
                   mu_factors=None, seed=0, engine="auto", block_records=50, store_device=None, sheets=None,
                   store_dtype=torch.float32):
    """Several assemblies with the SAME wheel count in one batched run (much faster for many small groups).
    asms: list of Assembly; doms/modes/gs (and optional payloads/mu_factors): lists aligned with asms, each for that
    assembly's b_i trials. Returns a list of per-assembly result dicts identical in format to simulate()."""
    n = asms[0].n
    assert all(a.n == n for a in asms), "simulate_multi needs equal wheel counts"
    device = torch.device(device)
    if engine == "auto":
        engine = "graph" if device.type == "cuda" else "eager"
    Cs, sizes, info = [], [], []
    steps = None
    for i, asm in enumerate(asms):
        sched = Schedule(asm.char_length())
        C, st, nom, mode, g = _prepare(asm, params, doms[i], modes[i], gs[i], sched, dt, device,
                                       payloads[i] if payloads else None, mu_factors[i] if mu_factors else None,
                                       seed + 7919 * i, sheet=sheets[i] if sheets else None)
        assert steps is None or st == steps
        steps = st
        b = modes[i].shape[0]
        for k in _SHARED:
            C[k] = C[k].expand(b, *C[k].shape)
        Cs.append(C)
        sizes.append(b)
        info.append((asm, sched, nom, mode, g))
    C = {}
    for k in Cs[0]:
        v = Cs[0][k]
        if torch.is_tensor(v) and v.dim() >= 1 and k not in ("steps_f", "dt"):
            C[k] = torch.cat([c[k] for c in Cs], 0).contiguous()
        else:
            C[k] = v
    B = sum(sizes)
    R, n_rec, rec_every = _integrate(C, steps, B, n, dt, record_hz, device, engine, block_records, store_device,
                                     store_dtype, None)
    sd = R["wheel_angle"].device
    t = torch.arange(n_rec, dtype=torch.float32, device=sd) * rec_every * dt
    out, o = [], 0
    for i, b in enumerate(sizes):
        asm, sched, nom, mode, g = info[i]
        Ri = {k: R[k][o:o + b] for k in REC_KEYS}
        Ri["t"] = t
        Ri["mode"], Ri["g"] = mode.to(sd), g.to(sd)
        ood = "payload" if payloads and payloads[i] is not None else (
            "low_mu" if (mu_factors and mu_factors[i] is not None) or (sheets and sheets[i] is not None) else "none")
        Ri["meta"] = dict(n=n, B=b, record_hz=record_hz, cpr=nom["cpr"], tau_s=C["tau_s"][o:o + b, 0],
                          schedule=sched, asm=asm.name, assembly=asm, char_length=asm.char_length(), dt=dt,
                          engine=engine, seed=seed + 7919 * i, ood=ood)
        out.append(Ri)
        o += b
    return out


def targets_from_twist(R, schedule=None, char_length=None):
    """Ground-truth capability ratios (B,14) in probes.TARGET_KEYS order, from the simulator's true body twist
    (rho = <xi, xi_cmd>_W / |xi_cmd|_W^2 over each hold window, W = diag(1, 1, Lc^2)), plus drift (B,14).
    WS-B's featurizer computes the evaluator target from truth_pose; this is the noise-free reference."""
    sched = schedule or R["meta"]["schedule"]
    Lc = char_length if char_length is not None else R["meta"]["char_length"]
    t = R["t"]
    w = torch.tensor([1.0, 1.0, Lc * Lc], dtype=torch.float32, device=R["twist"].device)
    rho, drift = [], []
    for pid, leg, ta, tb, tw in sched.windows():
        m = (t >= ta) & (t <= tb)
        xr = R["twist"][:, m].float().mean(1)                          # (B,3)
        xc = torch.tensor(tw, dtype=torch.float32, device=xr.device)
        nc = (w * xc * xc).sum()
        r = (w * xr * xc).sum(-1) / nc
        rho.append(r)
        res = xr - r[:, None] * xc
        drift.append(torch.sqrt((w * res * res).sum(-1) / nc))
    return torch.stack(rho, 1), torch.stack(drift, 1)

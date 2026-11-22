"""WS-A simulator tests: structured solve equivalence, determinism, physics sanity, dt convergence."""
import numpy as np
import pytest
import torch

from capxfer import assembly, sim

P = assembly.load_params("params/module_measured.yaml")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _run(asm, B=64, mode_wheel0=0, g0=1.0, dt=0.005, seed=0, engine="auto", **kw):
    rng = np.random.default_rng(seed)
    dom = sim.sample_domain(B, rng)
    mode = torch.zeros(B, asm.n, dtype=torch.long)
    mode[:, 0] = mode_wheel0
    g = torch.ones(B, asm.n)
    g[:, 0] = g0
    return sim.simulate(asm, P, dom, mode, g, dt=dt, device=DEV, engine=engine, seed=seed, **kw)


def test_output_contract():
    asm = assembly.physical("single", P)
    R = _run(asm, B=8)
    T = R["t"].shape[0]
    for k, c in dict(wheel_angle=4, duty=4, torque=4, omega=4, tau_motor=4, twist=3, pose=3, cmd=3, accel=2).items():
        assert R[k].shape == (8, T, c), k
        assert torch.isfinite(R[k]).all(), k
    assert R["mode"].shape == (8, 4) and R["g"].shape == (8, 4)
    assert abs(float(R["t"][1] - R["t"][0]) - 0.01) < 1e-6
    assert R["duty"].abs().max() <= 1.0


def test_schur_solve_matches_dense():
    """The block (Schur) solve equals a dense solve of (M - dt J) dx = dt F built from the same step."""
    asm = assembly.physical("pair", P)
    B, n = 16, asm.n
    rng = np.random.default_rng(1)
    dom = sim.sample_domain(B, rng)
    mode, g = sim.sample_faults(B, n, rng)
    sched = sim.Schedule(asm.char_length())
    C, steps, nom, mode, g = sim._prepare(asm, P, dom, mode, g, sched, 0.005, torch.device("cpu"), None, None, 0)
    S = sim._init_state(B, n, "cpu")
    # xi = 0 so the explicitly-treated velocity-product terms (m wz vy, -m wz vx) have zero state derivative
    S["xi"] = torch.zeros(B, 3)
    S["om"] = torch.randn(B, n) * 3
    S64 = {k: v.double() for k, v in S.items()}
    C64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in C.items()}
    it = torch.tensor(300.0, dtype=torch.float64)
    Sn, _ = sim._step(S64, C64, it)
    dx = torch.cat([Sn["xi"] - S64["xi"], Sn["om"] - S64["om"]], 1)
    # residual check: (M - dt J) dx - dt F == 0 with J the state Jacobian of F (central differences)
    def F_of(x):
        xi, om = x[:, :3], x[:, 3:]
        S2 = dict(S64, xi=xi, om=om)
        # F = M (x(dt) - x) / dt for a vanishing step (float64)
        C2 = dict(C64, dt=torch.tensor(1e-9, dtype=torch.float64))
        Sx, _ = sim._step(S2, C2, it)
        M = torch.cat([C64["mass"][:, None].expand(B, 2), C64["izz"][:, None], C64["j_eff"]], 1)
        return torch.cat([Sx["xi"] - xi, Sx["om"] - om], 1) * M / 1e-9
    x0 = torch.cat([S64["xi"], S64["om"]], 1)
    F0 = F_of(x0)
    assert torch.isfinite(F0).all()
    # linearly-implicit Euler: M dx = dt (F0 + J dx). Check with J by central differences (smooth regime).
    eps = 1e-6
    J = torch.zeros(B, 3 + n, 3 + n, dtype=torch.float64)
    for j in range(3 + n):
        e = torch.zeros(3 + n, dtype=torch.float64)
        e[j] = eps
        J[:, :, j] = (F_of(x0 + e) - F_of(x0 - e)) / (2 * eps)
    M = torch.cat([C64["mass"][:, None].expand(B, 2), C64["izz"][:, None], C64["j_eff"]], 1)
    lhs = M * dx - 0.005 * (J @ dx[..., None])[..., 0]
    rhs = 0.005 * F0
    rel = (lhs - rhs).abs().max() / rhs.abs().max()
    assert rel < 1e-3, float(rel)


def test_determinism_and_engines():
    asm = assembly.physical("single", P)
    a = _run(asm, B=32, mode_wheel0=1, g0=0.5, seed=4)
    b = _run(asm, B=32, mode_wheel0=1, g0=0.5, seed=4)
    assert torch.equal(a["twist"], b["twist"]) and torch.equal(a["omega"], b["omega"])
    if DEV == "cuda":
        c = _run(asm, B=32, mode_wheel0=1, g0=0.5, seed=4, engine="eager")
        assert (c["twist"] - a["twist"]).abs().max() < 1e-5


def test_healthy_tracking():
    """WS-MR (2026-11-21), measured robot with the CONFIRMED firmware (feed-forward v_ref/0.65 + PI 1.0/0.8 per m/s at
    50 Hz): the feed-forward under-drives the ~0.49 m/s (11.1 V) motor and the slow PI closes only part of the gap in a
    1 s hold, so a healthy module ends the holds at ~0.85-0.90 of command, the same for every direction. Capability is
    therefore RELATIVE to the healthy baseline (D17). Trial spread now also comes from battery and gain DR (the shortfall
    depends on them), hence 0.2 instead of the old 0.1 around the per-direction median."""
    asm = assembly.physical("single", P)
    rho, drift = sim.targets_from_twist(_run(asm, B=64))
    med = rho.median(0).values
    assert 0.8 < float(med.min()) and float(med.max()) < 0.95          # measured-robot absolute healthy level
    assert float(med.max() - med.min()) < 0.07                         # uniform over directions (relative ~1)
    r, p6 = rho[:, :10], rho[:, 10:]
    assert (r / med[:10] - 1).abs().max() < 0.2
    assert p6.min() > 0.7 and p6.max() < 1.05                          # P6 near top speed: loosely
    assert drift[:, :10].max() < 0.08


@pytest.mark.parametrize("mode,expect", [
    (sim.WEAK, "P4"), (sim.COAST, "P4"), (sim.BRAKE, "P4"), (sim.ROLLER, "P2")])
def test_fault_effects(mode, expect):
    """Faulting the fl wheel ('/' roller axis) leaves the (+,+) diagonal P3 (which needs no fl spin) intact and
    degrades P4, which loads it. Roller lock keeps forward drive (P1) and mainly costs strafe. WS-MR (D17): checked on
    RELATIVE capability = faulted / healthy twin (same domain draws and seed), since the healthy measured robot itself
    ends the holds at ~0.87 of command."""
    asm = assembly.physical("single", P)
    rho, _ = sim.targets_from_twist(_run(asm, B=64, mode_wheel0=mode, g0=0.5))
    base, _ = sim.targets_from_twist(_run(asm, B=64))
    from capxfer.probes import TARGET_KEYS
    r = dict(zip(TARGET_KEYS, (rho.mean(0) / base.mean(0)).tolist()))
    intact = "P1+" if mode == sim.ROLLER else "P3+"
    assert abs(r[intact] - 1) < 0.05
    assert r[expect + "+"] < 0.9
    if mode in (sim.COAST, sim.BRAKE):
        assert r["P4+"] < 0.6


def test_passive_energy_dissipates():
    asm = assembly.physical("single", P)
    B = 16
    rng = np.random.default_rng(2)
    dom = sim.sample_domain(B, rng)
    mode = torch.full((B, 4), sim.COAST, dtype=torch.long)
    R = sim.simulate(asm, P, dom, mode, torch.ones(B, 4), device=DEV, xi0=torch.tensor([0.3, 0.2, 0.5]))
    m = asm.mass * dom["mass_scale"].to(DEV)
    izz = asm.izz() * dom["izz_scale"].to(DEV) * dom["mass_scale"].to(DEV)
    tw = R["twist"]
    j = sim.nominal(P)["j_eff"]
    E = 0.5 * m[:, None] * (tw[..., 0] ** 2 + tw[..., 1] ** 2) + 0.5 * izz[:, None] * tw[..., 2] ** 2 + \
        0.5 * j * (R["omega"] ** 2).sum(-1)
    assert (E[:, 1:] - E[:, :-1]).max() < 1e-6
    assert E[:, -1].max() < 1e-3 * E[:, 0].min()


def test_dt_convergence():
    """dt = 5 ms vs 1 ms: capability ratios within 0.01, wheel speeds within 2 % of peak (full study in A.md)."""
    asm = assembly.physical("single", P)
    rng = np.random.default_rng(5)
    B = 64
    dom = sim.sample_domain(B, rng)
    mode, g = sim.sample_faults(B, 4, rng)
    out = {}
    for dt in (0.001, 0.005):
        R = sim.simulate(asm, P, dict(dom), mode, g, dt=dt, device=DEV, seed=1)
        out[dt] = (sim.targets_from_twist(R)[0], R["omega"])
    assert (out[0.001][0] - out[0.005][0]).abs().max() < 0.01
    om_err = (out[0.001][1] - out[0.005][1]).abs().amax((1, 2)) / out[0.001][1].abs().amax((1, 2)).clamp(min=1)
    assert om_err.max() < 0.02


def test_ood_args_and_cpu():
    asm = assembly.physical("single", P)
    B = 8
    pay = dict(mass=torch.full((B,), 2.0), x=torch.full((B,), 0.08), y=torch.zeros(B))
    R = _run(asm, B=B, payload=pay)
    assert R["meta"]["ood"] == "payload" and torch.isfinite(R["twist"]).all()
    R = _run(asm, B=B, mu_factor=torch.full((B,), 0.25))
    rho, _ = sim.targets_from_twist(R)
    assert R["meta"]["ood"] == "low_mu" and rho.min() < 0.95       # P6 max effort saturates traction
    rng = np.random.default_rng(0)
    dom = sim.sample_domain(4, rng)
    Rc = sim.simulate(asm, P, dom, torch.zeros(4, 4, dtype=torch.long), torch.ones(4, 4), device="cpu")
    assert torch.isfinite(Rc["twist"]).all()


def test_simulate_multi_matches_single():
    rng = np.random.default_rng(3)
    asms = [assembly.physical("single", P), assembly.random_assembly(rng, P, sizes=(1,))]
    doms, modes, gs = [], [], []
    for a in asms:
        doms.append(sim.sample_domain(8, rng))
        m, g = sim.sample_faults(8, 4, rng)
        modes.append(m)
        gs.append(g)
    outs = sim.simulate_multi(asms, P, [dict(d) for d in doms], modes, gs, device=DEV, seed=11)
    for i, a in enumerate(asms):
        R = sim.simulate(a, P, dict(doms[i]), modes[i], gs[i], device=DEV, seed=11 + 7919 * i)
        assert (outs[i]["twist"] - R["twist"]).abs().max() < 1e-5
        assert outs[i]["meta"]["assembly"] is a

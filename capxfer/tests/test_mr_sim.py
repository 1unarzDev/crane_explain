"""WS-MR: measured-robot simulator settings (params MEASURED_V1): 50 Hz ZOH velocity loop with encoder-count speed,
measured-centred DR, sensor DR centred on the measured AprilTag latency, and the controller-gain diagnosis."""
import copy

import numpy as np
import torch

from capxfer import assembly, datagen, sim
from capxfer.probes import Schedule

P = assembly.load_params("params/module_measured.yaml")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _healthy(params, B=16, dr=None, seed=0, typ="single"):
    asm = assembly.physical(typ, params)
    rng = np.random.default_rng(seed)
    dom = sim.sample_domain(B, rng, dr)
    mode = torch.zeros(B, asm.n, dtype=torch.long)
    return sim.simulate(asm, params, dom, mode, torch.ones(B, asm.n), device=DEV, seed=seed)


def test_measured_params_are_used():
    nom = sim.nominal(P)
    assert nom["cpr"] == 2048 and nom["ctrl_hz"] == 50
    fw = P["firmware"]
    r = P["wheel"]["radius_m"]
    assert abs(nom["kp"] - fw["kp_duty_per_mps"] * r) < 1e-12 and abs(nom["ki"] - fw["ki_duty_per_m"] * r) < 1e-12
    assert abs(nom["ff_duty_per_radps"] - r / fw["ff_ref_speed_mps"]) < 1e-12     # duty = v_ref / 0.65 + ...
    assert abs(nom["radius"] - 0.0501) < 1e-9
    asm = assembly.physical("single", P)
    assert 12.5 < asm.mass < 13.1
    assert np.allclose(np.abs(asm.wheel_pos - asm.wheel_pos.mean(0)), 0.100, atol=1e-6)


def test_zoh_duty_changes_only_on_loop_ticks():
    R = _healthy(P, B=4)
    # records at 100 Hz, loop at 50 Hz: the duty is held for two records (record k at t = (k+1)*10 ms - 5 ms step)
    d = R["duty"][0, :, 0].cpu().numpy()
    changes = np.flatnonzero(np.abs(np.diff(d)) > 0)
    assert len(changes) > 50
    assert len(set(changes % 2)) == 1                                  # every change on the same record parity
    # encoder-quantised speed: duty steps are multiples of kp * (2 pi / cpr) / T_ctrl plus ki/ref terms -> finite
    assert torch.isfinite(R["twist"]).all() and R["duty"].abs().max() <= 1


def test_continuous_loop_when_rate_exceeds_dt():
    p = copy.deepcopy(P)
    p["firmware"]["vel_loop_hz"] = 200                                  # = 1/dt at dt 5 ms: original continuous PI
    asm = assembly.physical("single", p)
    rng = np.random.default_rng(0)
    dom = sim.sample_domain(2, rng)
    C, *_ = sim._prepare(asm, p, dom, torch.zeros(2, 4, dtype=torch.long), torch.ones(2, 4),
                         Schedule(asm.char_length()), 0.005,
                         torch.device("cpu"), None, None, 0)
    assert C["zoh"] is False
    C5, *_ = sim._prepare(asm, P, dict(dom), torch.zeros(2, 4, dtype=torch.long), torch.ones(2, 4),
                          Schedule(asm.char_length()), 0.005,
                          torch.device("cpu"), None, None, 0)
    assert C5["zoh"] is True and float(C5["ctrl_every"]) == 4


def test_dr_measured_centred():
    d = sim.DR_MEASURED
    assert d["mass_scale"] == (0.95, 1.05) and d["mu"] == (0.3, 1.2) and d["radius_scale"] == (0.99, 1.01)
    assert d["kp_scale"] == (1.0, 1.0) and d["ki_scale"] == (1.0, 1.0)     # confirmed firmware: gains not randomised
    assert d["noload_scale"] == (0.9, 1.35)                                # D17: no-load surface speed 0.44-0.66 m/s
    for k in ("latency", "stall_scale", "s0", "c_roller", "crr", "tau_bd", "izz_scale", "v_batt"):
        assert d[k] == sim.DR_WIDE[k]
    dom = sim.sample_domain(4096, np.random.default_rng(1), d)
    assert 0.95 <= float(dom["mass_scale"].min()) and float(dom["mass_scale"].max()) <= 1.05
    assert abs(float(dom["mass_scale"].mean()) - 1) < 0.005
    assert "ff_scale" not in dom                                        # optional key only drawn on request
    sdr = datagen.measured_sensor_dr(P)
    lat = P["truth"]["apriltag_latency_s"]
    assert abs(np.mean(sdr["truth_latency"]) - lat) < 1e-9 and abs(lat - 0.078) < 1e-9


def _nominal_dom(B, **over):
    v = dict(mu=0.5, s0=0.03, c_roller=2.0, crr=0.015, tau_bd=0.75, latency=0.04, stall_scale=1.0, noload_scale=1.0,
             mass_scale=1.0, izz_scale=1.0, radius_scale=1.0, kp_scale=1.0, ki_scale=1.0, v_batt=11.1, cg_dx=0.0,
             cg_dy=0.0)
    v.update(over)
    d = {k: torch.full((B,), float(x)) for k, x in v.items()}
    d["wheel_yaw_err_std"] = 0.0
    return d


def _nominal_rho(typ, **over):
    asm = assembly.physical(typ, P)
    z = torch.zeros(4, asm.n, dtype=torch.long)
    return sim.targets_from_twist(sim.simulate(asm, P, _nominal_dom(4, **over), z, torch.ones(4, asm.n), device=DEV))[0]


def test_confirmed_firmware_tracking_level():
    """Integrator gate 'healthy tracks 1.0 +- 0.06' is NOT met by the confirmed controller (docs/coordination/MR.md):
    the feed-forward v_ref/0.65 drives the motor to only ~0.75 of the reference (no-load surface speed ~0.485 m/s at
    11.1 V), and the PI (kp 1.0, ki 0.8 per m/s; closed-loop time constant ~4 s) recovers only part of the gap within the
    1 s holds: nominal healthy rho ~0.80-0.86 for one and four modules. If the drivetrain really reached 0.65 m/s at
    full duty (feed-forward exact), the same loop tracks at 1.0 +- 0.06. This test pins both facts."""
    for typ in ("single", "square4"):
        rho = _nominal_rho(typ)
        assert 0.78 < float(rho.min()) and float(rho.max()) < 0.88, (typ, rho.mean(0))
    exact = _nominal_rho("single", ff_scale=0.65 / (0.0501 * sim.nominal(P)["omega0"] * 11.1 / 12.0))
    assert (exact - 1).abs().max() < 0.06


def test_feedforward_is_what_makes_it_track():
    """Without the 1/0.65 feed-forward (ablation ff_scale 0) the same PI ends the 1 s holds far below the command."""
    assert float(_nominal_rho("single", ff_scale=0.0).max()) < 0.6

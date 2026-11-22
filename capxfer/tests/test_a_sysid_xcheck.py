"""WS-A sys-ID hook (CMA-ES) and L1 cross-check smoke tests."""
import numpy as np
import pytest
import torch

from capxfer import assembly, sim, sysid, xcheck_l1

P = assembly.load_params("params/module_measured.yaml")


def test_sysid_recovers_controller_and_motor():
    """WS-MR (2026-11-21): the controller is CONFIRMED (feed-forward v_ref/0.65 + PI, params firmware), so kp/ki are no
    longer fitted. With the feed-forward carrying most of the duty the wheels run lightly loaded and the motor's no-load
    speed (what sets rho, and what the team's free-speed hook check measures) is identified, while the stall torque is
    only weakly identified (wheel load torques are small): recovered within 25 %, not 10 % as under the old PI-only
    estimate. Same pseudo-real recipe otherwise."""
    true = dict(mu=0.8, s0=0.03, c_roller=2.0, crr=0.015, latency=0.05, stall_scale=0.9, noload_scale=1.15,
                izz_scale=1.0, kp_scale=1.0, ki_scale=1.0)
    real = sysid.make_pseudo_real(true, P, n_trials=4, seed=1)
    keys = ("latency", "stall_scale", "noload_scale")
    fixed = {k: v for k, v in true.items() if k not in keys}
    fit = sysid.fit_params(real, P, keys=keys, iters=12, popsize=16, fixed=fixed, pids=("P1", "P2", "P3"), restarts=1,
                           select="loss")
    assert fit["history"][-1] <= fit["history"][0]
    assert abs(fit["best"]["noload_scale"] - true["noload_scale"]) / true["noload_scale"] < 0.05, fit["best"]
    assert abs(fit["best"]["stall_scale"] - true["stall_scale"]) / true["stall_scale"] < 0.25, fit["best"]
    assert abs(fit["best"]["latency"] - true["latency"]) < 0.02
    dr = sysid.dr_around(fit)
    assert dr["noload_scale"][0] < fit["best"]["noload_scale"] < dr["noload_scale"][1]


def test_sysid_refuses_faulted():
    real = sysid.make_pseudo_real(dict(mu=0.8), P, n_trials=2)
    real[0]["meta"]["labels"]["mode"][0, 1] = 3
    with pytest.raises(AssertionError):
        sysid.fit_params(real, P, iters=1, popsize=4)


@pytest.mark.skipif(not xcheck_l1.FLEET.exists(), reason="Fleet worktree absent")
def test_l1_healthy_agreement():
    case = xcheck_l1.CASES[0]
    l1, conv = xcheck_l1.l1_ratios(P, case, n_worlds=4)
    ours = xcheck_l1.capxfer_ratios(P, case, B=32)
    assert conv.all()
    # WS-MR (2026-11-21): L1 is quasi-static (wheels at their commanded speed), while the measured robot's confirmed
    # firmware (feed-forward v_ref/0.65 > no-load ~0.49 m/s, slow PI) ends the 1 s holds at ~0.8-0.9 of command for
    # every direction alike. The cross-check is therefore on RELATIVE capability (D17): each model's ratios divided by
    # its own mean over the 14 directions must agree within 0.04, and capxfer's absolute healthy level is the
    # measured-robot shortfall (0.75-0.98), not ~1.
    l1m, om = np.nanmean(l1, 0), ours.mean(0)
    assert np.abs(l1m / l1m.mean() - om / om.mean()).max() < 0.04
    assert 0.75 < om.mean() < 0.98


def test_sysid_a2_defaults_fallback_gate_and_dr():
    """A2: the fit returns the better of (restart optima, nominal) on an internal split; the comparative gate and the
    sys-ID-centred DR are well formed. Pseudo-real plant with mu below the default DR and radius 0.97."""
    true = dict(mu=0.5, s0=0.03, c_roller=2.0, crr=0.015, latency=0.05, stall_scale=0.9, noload_scale=1.0,
                izz_scale=1.0, kp_scale=1.0, ki_scale=1.0, radius_scale=0.97)
    real = sysid.make_pseudo_real(true, P, n_trials=6, seed=2)[0]
    fit_raw, ho_raw = sysid.subset_trials(real, range(4)), sysid.subset_trials(real, range(4, 6))
    keys = ("radius_scale", "latency", "stall_scale")
    fixed = {k: v for k, v in true.items() if k not in keys}
    fit = sysid.fit_params([fit_raw], P, keys=keys, iters=4, popsize=6, restarts=2, fixed=fixed, pids=("P1",))
    assert fit["use"] in ("fitted", "nominal") and set(fit["val"]) == {"fitted_r0", "fitted_r1", "nominal"}
    assert (fit["use"] == "nominal") == (min(fit["val"], key=fit["val"].get) == "nominal")
    assert fit["n_fit"] == 2 and fit["n_val"] == 2
    g = sysid.gate(fit, [ho_raw], P, fixed=fixed)
    assert g["gate_pass"] == (g["e_fit"] <= g["e_nom"] - g["margin"] and g["e_fit"] <= g["cap"])
    assert g["e_nom"] > 0.015                                      # radius 0.97 is visible to the rho error
    dr = sysid.dr_from_sysid(fit, gate_result=g)
    assert dr["mu"] == sim.DR_WIDE["mu"] and dr["s0"] == sim.DR_WIDE["s0"]   # keep_wide keys
    if g["use"] == "fitted":
        lo, hi = dr["radius_scale"]
        assert sysid.FIT_BOUNDS["radius_scale"][0] <= lo <= fit["best"]["radius_scale"] <= hi <= sysid.FIT_BOUNDS["radius_scale"][1]
    d = sysid.nominal_domain(3, P, dict(mu=0.4))
    assert torch.allclose(d["mu"], torch.full((3,), 0.4)) and d["radius_scale"][0] == 1.0

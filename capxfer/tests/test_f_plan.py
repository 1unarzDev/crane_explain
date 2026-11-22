"""WS-F: capability adapter, docking scene/planner and the sim executor (short)."""
import math
import os

import numpy as np
import pytest

from capxfer import assembly as A
from capxfer.probes import TARGET_KEYS
from capxfer.plan import capability as CAP
from capxfer.plan import dock as D

P = A.load_params(os.path.join(os.path.dirname(__file__), "..", "params", "module_measured.yaml"))
NOSTRAFE = np.array([0.1 if k[:2] in ("P2", "P3", "P4") or k.startswith("P6y") else 1.0 for k in TARGET_KEYS])


def test_from_intervals_gating_and_model():
    asm = A.physical("pair", P)
    lo = np.full(14, 0.9)
    lo[TARGET_KEYS.index("P2+")] = 0.3
    cs = CAP.from_intervals(lo, lo + 0.1, asm)
    assert not cs.usable["P2+"] and cs.usable["P2-"] and cs.usable["P1+"]
    assert not cs.allowed((0.0, 0.2, 0.0)) and cs.allowed((0.0, -0.2, 0.0))
    e = cs.expected((0.2, 0.0, 0.0))
    assert abs(e[0] - 0.19) < 1e-9 and abs(e[1]) < 1e-12
    for xi in ((0.1, 0.05, 0.2), (-0.2, -0.1, -0.3), (0.0, -0.2, 0.1)):
        assert np.allclose(cs.command_for(cs.expected(xi)), xi)
    with pytest.raises(ValueError):
        CAP.from_intervals(np.ones(13), np.ones(13), asm)


def test_omni_and_drift_gate():
    asm = A.physical("square4", P)
    om = CAP.omni(asm)
    assert all(om.usable.values()) and om.vmax["y+"] == CAP.V_FAST
    d = np.zeros(14)
    d[TARGET_KEYS.index("P2+")] = 0.4
    cs = CAP.from_point(np.ones(14), asm, d)
    assert not cs.usable["P2+"]
    d[:] = 0.0
    d[TARGET_KEYS.index("P5+")] = 0.4           # yaw tolerates more drift
    assert CAP.from_point(np.ones(14), asm, d).usable["P5+"]


@pytest.mark.parametrize("name", ["single", "pair"])
def test_planner_routes(name):
    asm = A.physical(name, P)
    sc = D.make_scene(asm, 1, "wide")
    p1 = D.plan_dock(CAP.omni(asm), asm, sc)
    assert p1.status == "ok" and p1.route() == "sideways"
    p2 = D.plan_dock(CAP.from_point(NOSTRAFE, asm, name="nostrafe"), asm, sc)
    assert p2.status == "ok" and p2.route() == "yaw+forward" and p2.duration > p1.duration
    assert all(abs(c[1]) < 1e-9 for c, _ in p2.cmd)


def test_command_schedule_preserves_integral():
    cmds = [((0.2, 0.0, 0.0), 1.0), ((0.0, -0.1, 0.3), 0.5)]
    s = D.CommandSchedule(cmds)
    t = np.arange(0, s.T, 1e-4)
    I = s.twist(t).sum(0) * 1e-4
    assert np.allclose(I, [0.2, -0.05, 0.15], atol=2e-3)


def test_execute_healthy_single_closed_loop():
    import torch
    torch.set_num_threads(2)
    from capxfer import sim
    asm = A.physical("single", P)
    sc = D.make_scene(asm, 2, "wide")
    dom = sim.sample_domain(1, np.random.default_rng(0))
    r = D.simulate_arm("omni", asm, D.fault_spec(asm, "F0"), dom, P, sc, loop="closed", seed=3)
    assert r["success"], r
    assert r["route"] == "sideways"

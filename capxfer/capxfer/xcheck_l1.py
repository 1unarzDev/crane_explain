"""Cross-check of the capxfer batched simulator against the independent Fleet L1 truth model.

    python -m capxfer.xcheck_l1 [--worlds 48] [--trials 256] [--out ~/.cache/capxfer/A/xcheck_l1.json]

The L1 model (~/worktrees/crane-fleet-orch/Tools/Fleet/capplan/truth, imported read-only via sys.path; nothing there
is modified) is solved quasi-statically (steady twist under ideal-IK wheel-speed commands) for every probe direction
of the library. capxfer runs the full dynamic probe schedule and averages each hold window. Both give the capability
ratio rho = <xi, xi_cmd>_W / |xi_cmd|_W^2 (W = diag(1, 1, Lc^2)) per TARGET_KEYS.

Geometry/parameter matching. A custom L1 module type `capxfer_xcheck` is registered IN MEMORY with the capxfer
single-module geometry (wheel positions +-lx, +-ly from params/module_measured.yaml, radius, mass, the same X roller
pattern: L1 hand -1/+1/+1/-1 = capxfer "/", "\\", "\\", "/"). Motor: L1 stall torque and no-load speed set to
capxfer's tau_s * V/12 and omega0 * V/12 with V ~ U(11.1, 12.6). Regime "isotropic" (mu 0.6-1.0 in both).

Remaining model differences (stated, not removed):
* L1 velocity loop is P-only (K_v 0.3-3 N m s/rad, steady-state error under load); capxfer is PI (zero steady-state
  error). The "matched" variant overrides K_v = 30 (approximates PI steady state); "native" keeps L1's K_v.
* L1 slip curve: tanh of a speed-normalised slip ratio (s_x0 0.02-0.08) with a friction circle; capxfer: tanh of slip
  velocity (s0 0.01-0.05 m/s), no coupling between channels.
* Roller drag: L1 Coulomb-like mu_rb N tanh(v/0.01) (mu_rb 0.01-0.06); capxfer viscous c_roller v (0.5-4 N s/m).
* Normal loads: L1 vertical statics with load transfer at COM height; capxfer static least-norm loads.
* L1 wheels are massless (quasi-static); capxfer integrates wheel and body dynamics over a 1 s hold.
* Faults: L1 dead_locked = wheel speed exactly 0  <->  capxfer BRAKE (winding-short damping + back-drive).
  L1 dead_free with tau_bd overridden to capxfer's range (0.3-1.2 N m)  <->  capxfer COAST.
  L1 weak(g) scales stall torque and no-load speed by g  <->  capxfer WEAK duty x g (same effect).
  Roller-locked: L1 conventional wheel at that corner (lateral friction kappa_lat mu)  <->  capxfer ROLLER.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

import numpy as np
import torch

from . import assembly as A
from . import sim
from .probes import TARGET_KEYS, Schedule

FLEET = pathlib.Path(os.path.expanduser("~/worktrees/crane-fleet-orch/Tools/Fleet"))
PARAMS = pathlib.Path(__file__).resolve().parents[1] / "params" / "module_measured.yaml"
CASES = (("healthy", sim.OK, None, "ok"), ("weak0.5", sim.WEAK, 0.5, ("weak", 0.5)),
         ("locked(brake)", sim.BRAKE, None, "dead_locked"), ("coast(dead_free)", sim.COAST, None, "dead_free"),
         ("roller_locked", sim.ROLLER, None, "roller"))


def _l1():
    if str(FLEET) not in sys.path:
        sys.path.insert(0, str(FLEET))
    from capplan import library, modgraph, truth          # noqa: F401
    from capplan.truth import model as M
    return library, modgraph, truth, M


def _register(params, roller_fl=False):
    library, modgraph, truth, M = _l1()
    mp, wp = params["module"], params["wheel"]
    lx, ly, r = float(mp["half_wheelbase_m"]), float(mp["half_track_m"]), float(wp["radius_m"])
    kw = dict(radius=r, phi_max=50.0, tau_max=float(params["motor"]["stall_torque_nm_12v"]))
    wheels = library.mecanum_set(lx=lx, ly=ly, **kw)
    name = "capxfer_xcheck"
    if roller_fl:
        wheels[0] = library.wheel("wheel_conventional", lx, ly, 0.0, 0, **kw)
        name += "_rl"
    side = float(mp["side_x_m"])
    spec = library._t(name, library.body(side, side, 0.14, float(mp["mass_kg"]["M1"])), wheels)
    if name not in library.TYPE_REGISTRY:
        library.register(name, spec)
    return modgraph.single(name)


def l1_ratios(params, case, n_worlds=48, seed=0, matched=True):
    """(n_worlds, 14) L1 quasi-static rho for the single module with wheel fl in `case`."""
    library, modgraph, truth, M = _l1()
    name, _, g, h = case
    graph = _register(params, roller_fl=(h == "roller"))
    asm = A.physical("single", params)
    Lc = asm.char_length()
    sched = Schedule(Lc)
    xc = np.array([w[4] for w in sched.windows()])          # (14,3) commanded hold twists
    nom = sim.nominal(params)
    rng = np.random.default_rng(seed)
    health = None if h in ("ok", "roller") else {("m0", 0): h}
    out, conv = [], []
    for k in range(n_worlds):
        vb = rng.uniform(11.1, 12.6) / 12.0
        ov = dict(tau_stall=nom["tau_s"] * vb, w_nl=nom["omega0"] * vb)
        if matched:
            ov["K_v"] = 30.0
            ov["tau_bd"] = float(rng.uniform(0.3, 1.2))
        w = truth.World(seed * 1000 + k, "isotropic", overrides=ov)
        Am = M.build(graph, health, w)
        assert np.allclose(np.sort(Am.pos, 0), np.sort(asm.wheel_pos, 0), atol=1e-9)
        cmd = asm.ik(xc, nom["radius"])                     # firmware IK with nominal radius (same as capxfer)
        r = M.solve_qs(Am, cmd)
        xi = r["xi"]
        Wt = np.array([1.0, 1.0, Lc * Lc])
        rho = (Wt * xi * xc).sum(1) / (Wt * xc * xc).sum(1)
        out.append(np.where(r["converged"], rho, np.nan))            # non-converged QS solves are excluded
        conv.append(r["converged"])
    return np.array(out), np.array(conv)


def capxfer_ratios(params, case, B=256, seed=0, device=None):
    name, mode_id, g, _ = case
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    asm = A.physical("single", params)
    rng = np.random.default_rng(seed)
    dom = sim.sample_domain(B, rng)
    mode = torch.zeros(B, 4, dtype=torch.long)
    mode[:, 0] = mode_id
    gg = torch.ones(B, 4)
    if g is not None:
        gg[:, 0] = g
    R = sim.simulate(asm, params, dom, mode, gg, device=device, seed=seed)
    return sim.targets_from_twist(R)[0].cpu().numpy()


def run(n_worlds=48, B=256, seed=0, out=None, log=print):
    params = A.load_params(PARAMS)
    rows = {}
    for case in CASES:
        ours = capxfer_ratios(params, case, B=B, seed=seed)
        res = {}
        for variant, matched in (("matched", True), ("native", False)):
            l1, conv = l1_ratios(params, case, n_worlds=n_worlds, seed=seed, matched=matched)
            res[variant] = dict(mean=np.nanmean(l1, 0).tolist(), std=np.nanstd(l1, 0).tolist(),
                                converged=float(conv.mean()))
        rows[case[0]] = dict(capxfer_mean=ours.mean(0).tolist(), capxfer_std=ours.std(0).tolist(), l1=res)
        d = np.abs(np.array(res["matched"]["mean"]) - ours.mean(0))
        rows[case[0]]["max_abs_diff_matched"] = float(d.max())
        rows[case[0]]["median_abs_diff_matched"] = float(np.median(d))
        log(f"{case[0]:18s} max|d mean rho| matched={d.max():.3f} (key {TARGET_KEYS[int(d.argmax())]}) "
            f"median={np.median(d):.3f}")
    summary = dict(keys=list(TARGET_KEYS), rows=rows, n_worlds=n_worlds, B=B, seed=seed,
                   fleet_commit=_fleet_commit(), params_sha256=_sha(PARAMS))
    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out).write_text(json.dumps(summary, indent=1))
    return summary


def _sha(p):
    import hashlib
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _fleet_commit():
    head = FLEET.parents[1] / ".git"
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", str(FLEET), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def markdown(summary):
    """Agreement table (mean rho per probe key) for docs/coordination/A.md."""
    keys = summary["keys"]
    lines = ["| case | source | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for case, r in summary["rows"].items():
        lines.append(f"| {case} | capxfer | " + " | ".join(f"{v:.2f}" for v in r["capxfer_mean"]) + " |")
        lines.append(f"| | L1 matched | " + " | ".join(f"{v:.2f}" for v in r["l1"]["matched"]["mean"]) + " |")
        lines.append(f"| | L1 native | " + " | ".join(f"{v:.2f}" for v in r["l1"]["native"]["mean"]) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=48)
    ap.add_argument("--trials", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/capxfer/A/xcheck_l1.json"))
    a = ap.parse_args(argv)
    s = run(a.worlds, a.trials, a.seed, a.out)
    print(markdown(s))


if __name__ == "__main__":
    main()

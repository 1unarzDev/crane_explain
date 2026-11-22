"""capxfer cluster tasks, run by runner.py in a torch interpreter with cwd = the capxfer project root.

stdin {"task", "args", "seed", "out"}; stdout last line {"metrics": {name: number}, "env": {...}} (files go to `out`,
and the worker uploads each as an artifact). Add a task: a function TASK(seed, out, **args) -> metrics dict, then
register it in TASKS. Everything here is SIMULATION.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time


def _env():
    import numpy
    import torch
    torch.set_num_threads(int(os.environ.get("CAPXFER_THREADS", "4")))
    return dict(os=platform.system(), os_release=platform.release(), machine=platform.machine(),
                node_hash=__import__("hashlib").sha256(platform.node().encode()).hexdigest()[:8],
                cpu_count=os.cpu_count(), python=sys.version.split()[0], torch=torch.__version__,
                numpy=numpy.__version__, torch_threads=torch.get_num_threads(),
                cuda_built=bool(torch.version.cuda), cuda_visible=torch.cuda.is_available())


def smoke(seed, out, n_trials=64, asm="pair"):
    """Import capxfer and run an n_trials-trial batch (random faults, DR domain) of one physical assembly on CPU."""
    import numpy as np
    import torch
    import capxfer
    from capxfer import assembly as A, sim
    P = A.load_params("params/module_measured.yaml")
    a = A.physical(asm, P)
    rng = np.random.default_rng(seed)
    dom = sim.sample_domain(n_trials, rng)
    mode, g = sim.sample_faults(n_trials, a.n, rng)
    t0 = time.time()
    R = sim.simulate(a, P, dom, torch.as_tensor(mode), torch.as_tensor(g, dtype=torch.float32), device="cpu", seed=seed)
    wall = time.time() - t0
    rho, drift = sim.targets_from_twist(R)
    rho = rho.cpu().numpy()
    rec = dict(task="smoke", asm=asm, n_wheels=a.n, n_trials=n_trials, sim_wall_s=wall, T=int(R["t"].shape[0]),
               rho_mean=float(np.nanmean(rho)), rho_finite_frac=float(np.isfinite(rho).mean()),
               capxfer_file=os.path.relpath(capxfer.__file__))
    with open(os.path.join(out, "smoke.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    return dict(n_trials=n_trials, n_wheels=a.n, sim_wall_s=wall, trials_per_s=n_trials / wall,
                rho_mean=rec["rho_mean"], rho_finite_frac=rec["rho_finite_frac"])


def _sweep_assemblies(P, n_asm, rng_seed):
    """Exactly exp_sweep.main's morphology list: 4 physical types, then n_asm random ones of size sizes[j % 6]."""
    import numpy as np
    from capxfer import assembly as A
    rng = np.random.default_rng(rng_seed)
    asms = [A.physical(k, P) for k in ("single", "pair", "L3", "square4")]
    for k in asms:
        k.meta["family"] = "phys_" + k.name
    sizes = [1, 2, 3, 4, 5, 6]
    for j in range(n_asm):
        asms.append(A.random_assembly(rng, P, sizes=(sizes[j % 6],)))
    return asms


def i2_sweep_cell(seed, out, n_asm=336, rng_seed=0, expect_n=None):
    """One morphology of the I2 masking sweep (exp_sweep.py, unchanged physics and configs): the job seed is the
    morphology index in exp_sweep's list (--n-asm n_asm --seed rng_seed). Writes i2_asm<idx>.pt with exp_sweep's
    per-assembly result dict (merge into a sweep file with cluster/merge_i2.py)."""
    import numpy as np
    import torch
    from capxfer import assembly as A, sim
    from capxfer.ideas.i2 import exp_sweep as X, simrun as S, theory as T
    P = A.load_params("params/module_measured.yaml")
    asms = _sweep_assemblies(P, n_asm, rng_seed)
    idx = int(seed)
    asm = asms[idx]
    n = asm.n
    if expect_n is not None and n not in (expect_n if isinstance(expect_n, list) else [expect_n]):
        raise ValueError("morphology %d has n=%d, expected %s" % (idx, n, expect_n))
    modes, gs, tags = X.configs(n)
    B = len(modes) + X.N_DR
    dom = S.centred_domain(B)
    dr = sim.sample_domain(X.N_DR, np.random.default_rng(1000 + idx))
    for k in dom:
        if torch.is_tensor(dom[k]) and k in dr:
            dom[k][len(modes):] = dr[k]
    M = torch.as_tensor(np.concatenate([modes, np.zeros((X.N_DR, n), np.int64)]))
    G = torch.as_tensor(np.concatenate([gs, np.ones((X.N_DR, n), np.float32)]))
    t0 = time.time()
    R = sim.simulate_multi([asm], P, [dom], [M], [G], device="cpu", seed=idx, store_device="cpu")[0]
    t_sim = time.time() - t0
    w = S.windows_reduce(R)
    del R
    t1 = time.time()
    xi, om, du = T.qs_solve(asm, P, modes, gs, T.legs(asm))
    t_qs = time.time() - t1
    qs_rho = T.rho_of(xi, T.legs(asm), asm.char_length()).astype(np.float32)
    res = dict(name=asm.name, family=asm.meta.get("family"), n=n, modules=asm.modules, wheel_pos=asm.wheel_pos,
               w_hat=asm.w_hat, r_hat=asm.r_hat, tags=tags, nc=len(modes),
               sim={k: v.astype(np.float32) for k, v in w.items()},
               qs=dict(rho=qs_rho, om=om.astype(np.float32), duty=du.astype(np.float32), twist=xi.astype(np.float32),
                       resid=T.qs_solve.last_resid.astype(np.float32)),
               E=T.excitation(asm), idx=idx, n_asm=n_asm, rng_seed=rng_seed, g_weak=X.G_WEAK, n_dr=X.N_DR,
               sim_seed=idx, source="SIMULATION")
    torch.save(res, os.path.join(out, "i2_asm%04d.pt" % idx))
    rho = w["rho"][:len(modes)]
    loss = (rho[0:1] - rho[1:]).max(1)  # worst-leg loss per single fault
    passive = np.array([m in (sim.COAST, sim.BRAKE) for (_, m) in tags[1:]])
    return dict(idx=idx, n=n, nc=len(modes), sim_s=t_sim, qs_s=t_qs, qs_rho_mae=float(np.abs(qs_rho - rho).mean()),
                worst_loss_passive_median=float(np.median(loss[passive])), worst_loss_max=float(loss.max()))


TASKS = dict(smoke=smoke, i2_sweep_cell=i2_sweep_cell)


def main():
    req = json.loads(sys.stdin.read())
    env = _env()
    t0 = time.time()
    metrics = TASKS[req["task"]](int(req["seed"]), req["out"], **(req.get("args") or {}))
    metrics["task_s"] = time.time() - t0
    with open(os.path.join(req["out"], "env.json"), "w") as fh:
        json.dump(env, fh, indent=1)
    sys.stdout.write("\n" + json.dumps({"metrics": metrics, "env": env}) + "\n")


if __name__ == "__main__":
    main()

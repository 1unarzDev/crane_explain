"""Task 3(b): sim-to-sim stress tests. Models trained on the W simulator configuration (DR_WIDE) are tested on
deliberately MISSPECIFIED plants: effects that are outside the training DR or absent from the model class.

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.i4.stress gen      # -> ~/.cache/capxfer/I4/stress_v1_feats.pt
  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.i4.stress eval     # -> ~/.cache/capxfer/I4/stress_v1_results.json

The plant changes are injected without editing capxfer/sim.py: `sim._prepare` is wrapped (per-trial constants) and
`sim._step` is replaced by a source-patched copy (asserted replacements, so it stays in sync with sim.py). The firmware
IK and the featurizer keep the NOMINAL assembly geometry, exactly as on the robot.

Conditions (every condition reuses the SAME domain draw, faults, sim seed and sensor-noise seed as `nominal`, so
d_rho = rho(stress) - rho(nominal) is an exact paired consequence):
  nominal        DR_WIDE as in training (control)
  yaw_mis        per-wheel mounting yaw error sd 0.10 rad (5.7 deg); training sd 0.03 rad
  latency_hi     actuation latency U(0.15, 0.25) s; training range 0.01-0.12 s
  roller_aniso   per-wheel roller-bearing drag x logU(1/4, 4), direction-asymmetric (x U(1, 4) for one rolling
                 direction), per-wheel friction x U(0.7, 1.3). Training: one isotropic viscous c_roller per trial.
  compliant      static proxy of compliant couplings / docking play: per-module yaw N(0, 2 deg) and offset N(0, 4 mm)
                 of the real wheel geometry against the nominal grid, and normal loads redistributed x LogN(0, 0.25)
                 per wheel (flexing chassis / uneven floor). Training: rigid exact grid, least-norm static loads.
  friction_form  slip-curve MODEL FORM change: Pacejka-like -mu N sin(1.5 atan(s / 1.5 s0)) (peak at 2.6 s0, falls
                 to 0.71 mu) instead of tanh, plus a friction circle on roller-locked wheels (combined slip).
  combined       all of the above at once
"""
from __future__ import annotations

import inspect
import json
import math
import sys
import time

import numpy as np
import torch

from capxfer import assembly as A
from capxfer import datagen, sensors, sim

from .common import I4, TYPES, feat_from_raw, json_dump

CONDS = ("nominal", "yaw_mis", "latency_hi", "roller_aniso", "compliant", "friction_form", "combined")
PARTS = dict(nominal=(), yaw_mis=("yaw",), latency_hi=("lat",), roller_aniso=("roller",), compliant=("compl",),
             friction_form=("pac",), combined=("yaw", "lat", "roller", "compl", "pac"))
_STATE = dict(parts=(), rng=None, asm=None)


def _pac(s, s0, mu, N, Cp):
    x = s / s0
    a = torch.atan(x / Cp)
    f = -mu * N * torch.sin(Cp * a)
    df = -mu * N * torch.cos(Cp * a) / (1 + (x / Cp) ** 2) / s0
    return f, df


_OLD = """    th1 = torch.tanh(s1 / s0)
    f1 = -mu * N * th1
    df1 = -mu * N * (1 - th1 * th1) / s0
    th2 = torch.tanh(s2 / s0)
    f2 = torch.where(fric2, -mu * N * th2, -c_r * s2)
    df2 = torch.where(fric2, -mu * N * (1 - th2 * th2) / s0, -c_r.expand_as(s2))
"""
_NEW = """    if "pac_C" in C:
        f1, df1 = _pac(s1, s0, mu, N, C["pac_C"])
        f2r, df2r = _pac(s2, s0, mu, N, C["pac_C"])
    else:
        th1 = torch.tanh(s1 / s0)
        f1 = -mu * N * th1
        df1 = -mu * N * (1 - th1 * th1) / s0
        th2 = torch.tanh(s2 / s0)
        f2r = -mu * N * th2
        df2r = -mu * N * (1 - th2 * th2) / s0
    c_r_eff = c_r * torch.where(s2 > 0, C["asym_pos"], C["asym_neg"]) if "asym_pos" in C else c_r
    f2 = torch.where(fric2, f2r, -c_r_eff * s2)
    df2 = torch.where(fric2, df2r, -c_r_eff.expand_as(s2))
    if "pac_C" in C:
        fm = torch.sqrt(f1 * f1 + f2 * f2 + 1e-12)
        sc = torch.where(fric2, torch.clamp(mu * N / fm, max=1.0), torch.ones_like(fm))
        f1, df1 = f1 * sc, df1 * sc
        f2, df2 = f2 * sc, torch.where(fric2, df2 * sc, df2)
"""


def _patched_step():
    src = inspect.getsource(sim._step)
    assert src.count(_OLD) == 1, "sim._step changed; update stress._OLD"
    ns = dict(sim.__dict__)
    ns["_pac"] = _pac
    exec(compile(src.replace(_OLD, _NEW), "stress_step", "exec"), ns)
    return ns["_step"]


_ORIG_PREPARE, _ORIG_STEP = sim._prepare, sim._step
_STEP = _patched_step()


def _rot(v, d):
    c, s = torch.cos(d)[..., None], torch.sin(d)[..., None]
    return torch.cat([c * v[..., :1] - s * v[..., 1:], s * v[..., :1] + c * v[..., 1:]], -1)


def _prepare(asm, params, dom, mode, g, sched, dt, device, payload, mu_factor, seed, sheet=None):
    parts, rng = _STATE["parts"], _STATE["rng"]
    C, steps, nom, mode, g = _ORIG_PREPARE(asm, params, dom, mode, g, sched, dt, device, payload, mu_factor, seed,
                                           sheet)
    B, n = mode.shape
    f32 = dict(dtype=torch.float32, device=device)
    T = lambda x: torch.tensor(np.asarray(x), **f32)
    if "roller" in parts:
        C["c_r"] = C["c_r"] * T(np.exp(rng.uniform(np.log(0.25), np.log(4.0), (B, n))))
        asym = T(rng.uniform(1.0, 4.0, (B, n)))
        flip = T(rng.uniform(size=(B, n)) < 0.5)
        C["asym_pos"] = flip * asym + (1 - flip)
        C["asym_neg"] = (1 - flip) * asym + flip
        C["mu"] = C["mu"] * T(rng.uniform(0.7, 1.3, (B, n)))
    if "compl" in parts:
        m = len(asm.modules)
        dl = T(rng.normal(0, math.radians(2.0), (B, m)))
        off = T(rng.normal(0, 0.004, (B, m, 2)))
        wm = torch.as_tensor(asm.wheel_module, device=device).long()
        cm = T(asm.module_pos)[wm][None].expand(B, n, 2)
        d_w, o_w = dl[:, wm], off[:, wm]
        pos = cm + o_w + _rot(C["pos"] - cm, d_w)
        C["pos"] = pos
        for k in ("d1", "d2"):
            C[k] = _rot(C[k], d_w)
        lever = lambda d: torch.stack([d[..., 0], d[..., 1], pos[..., 0] * d[..., 1] - pos[..., 1] * d[..., 0]], -1)
        C["a1"], C["a2"] = lever(C["d1"]), lever(C["d2"])
        Nn = C["N"] * T(np.exp(rng.normal(0, 0.25, (B, n))))
        C["N"] = Nn * (C["N"].sum(1, keepdim=True) / Nn.sum(1, keepdim=True))
    if "pac" in parts:
        C["pac_C"] = 1.5
    return C, steps, nom, mode, g


def simulate_stressed(asm, params, dom, mode, g, parts, rng, seed):
    _STATE.update(parts=parts, rng=rng)
    dom = dict(dom)
    if "yaw" in parts:
        dom["wheel_yaw_err_std"] = 0.10
    if "lat" in parts:
        dom["latency"] = torch.tensor(rng.uniform(0.15, 0.25, mode.shape[0]), dtype=torch.float32)
    sim._prepare, sim._step = _prepare, _STEP
    try:
        return sim.simulate(asm, params, dom, mode, g, device="cuda", engine="graph", seed=seed), dom
    finally:
        sim._prepare, sim._step = _ORIG_PREPARE, _ORIG_STEP


def gen(B=256, seed=77, types=TYPES, conds=CONDS, out=I4 / "stress_v1_feats.pt", extra_asms=(), gpu_frac=0.1):
    torch.cuda.set_per_process_memory_fraction(gpu_frac)
    params = A.load_params(datagen.PARAMS)
    feats, summ = [], {}
    asms = [(t, A.physical(t, params)) for t in types] + list(extra_asms)
    for ti, (t, asm) in enumerate(asms):
        rng0 = np.random.default_rng(seed + ti)
        dom = sim.sample_domain(B, rng0, sim.DR_WIDE)
        mode, g = sim.sample_faults(B, asm.n, rng0)
        s_seed = int(rng0.integers(2 ** 31))
        rho_nom = None
        part = out.with_name(out.stem + f"_{t}.pt")
        if part.exists():
            fs, sm = torch.load(part, weights_only=False)
            feats += fs
            summ.update(sm)
            continue
        fs, sm = [], {}
        for ci, c in enumerate(conds):
            t0 = time.time()
            for attempt in range(30):                     # the GPU is shared: wait out transient OOM
                try:
                    R, dom_c = simulate_stressed(asm, params, dom, mode, g, PARTS[c],
                                                 np.random.default_rng(seed * 31 + ci), s_seed)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    time.sleep(20)
            rho, drift = sim.targets_from_twist(R)
            if c == "nominal":
                rho_nom = rho.cpu()
            raw = sensors.to_raw(R, params, dom_c, np.random.default_rng(seed + 1000 + ti),
                                 trial_ids=[f"stress_{c}_{t}_{i}" for i in range(B)], ood="none")
            lab = raw["meta"]["labels"]
            lab.update(rho_twist=rho.cpu(), rho_twist_nominal=rho_nom, family=f"phys_{t}" if t in TYPES else t,
                       asm_type=t if t in TYPES else "random", stress=c)
            F = feat_from_raw(raw)
            F["labels"]["stress"] = c
            F["labels"]["rho_twist_nominal"] = rho_nom
            fs.append(F)
            d = (rho.cpu() - rho_nom).abs()
            ok = torch.isfinite(rho).all(1)
            sm[f"{c}/{t}"] = dict(median_abs_drho=float(d.median()), p90_trial_max_drho=float(d.amax(1).quantile(0.9)),
                                    frac_trial_drho_gt_0p1=float((d.amax(1) > 0.1).float().mean()),
                                    finite=float(ok.float().mean()), s=time.time() - t0)
            print(c, t, json.dumps(sm[f"{c}/{t}"]), flush=True)
            del R, raw
        torch.save((fs, sm), part)
        feats += fs
        summ.update(sm)
    torch.save(feats, out)
    json_dump(summ, I4 / "stress_v1_drho.json")
    return feats, summ


METHODS = ("W:structured_ensemble", "W:structured_npe", "W:ensemble", "W:blackbox", "W:analytic_bayes",
           "W:nominal_sim")


def evaluate(methods=METHODS, out=I4 / "stress_v1_results.json"):
    """Calibrate (split, split-joint-10, APS) on devW random in-model sim groups (what a sim-calibrated deployment
    would do), test on every stress condition x physical type. Also scores the stressed trials against the paired
    nominal target to separate 'prediction moved' from 'truth moved'."""
    from . import eval_core as E
    feats = torch.load(I4 / "stress_v1_feats.pt", weights_only=False)
    import os
    E.MEM_FRAC = float(os.environ.get("I4_MEM_FRAC", 0.06))
    dev = E.devw_feats()
    res = json.loads(out.read_text()) if out.exists() else {}
    for spec in methods:
        if spec in res:
            continue
        t0 = time.time()
        cal_f = dev["random"][::6] if spec.endswith("analytic_bayes") else dev["random"][::2]
        m = E.load_model(spec)
        nmax = max(F["wheel_x"].shape[1] for F in cal_f + feats)
        cal = E.predict(m, cal_f, "cal", nmax)
        st = E.predict(m, feats, "stress", nmax)
        tests = {}
        conds = sorted(set(st["stress"]), key=CONDS.index)
        fam = np.asarray(st["family"])
        for c in conds:
            selc = np.asarray(st["stress"]) == c
            tests[f"{c}/all"] = E.subset(st, selc)
            for t in np.unique(fam[selc]):
                tests[f"{c}/{t}"] = E.subset(st, selc & (fam == t))
        r = E.conformal(cal, tests)
        ap = E.aps(cal, tests)
        for k, t in tests.items():
            r[k]["diag"] = E.diag_stats(t)
            r[k]["aps"] = ap.get(k)
        res[spec] = r
        print(spec, f"{time.time() - t0:.0f}s", {c: (round(r[f"{c}/all"]["split_joint"]["joint"], 3),
                                                     round(r[f"{c}/all"]["raw"]["marg"], 3)) for c in conds}, flush=True)
        json_dump(res, out)
        del m
    return res


if __name__ == "__main__":
    {"gen": gen, "eval": evaluate}[sys.argv[1]]()

"""I1 decisive test: compositional sets with the stage-2 plant box centred on a SYS-ID FIT of pseudo-real plants.

For each plant P (fixed physical point + small per-trial jitter; differs from the training nominal):
  1. 16 healthy single-module trials -> sysid.fit_params on 8 (internal val split) + sysid.gate on the other 8.
     Plant box = sysid.dr_from_sysid(fit, gate_result) (the project's rule: unidentifiable keys stay DR_WIDE),
     plus an "aggressive" box (+-10 % of the DR width around theta_hat for every fitted key).
  2. Pseudo-real calibration set (random sizes 1-3 + single + pair) and test set (random sizes 4-6 + square4 + L3),
     generated with datagen.generate under the plant -> featurize -> W-model predictions (structured_npe, ensemble).
  3. Stage 1 calibrated on the pseudo-real cal wheels; stage 2 draws theta from the sys-ID box.
     Baselines: split CQR of ensemble / structured_npe calibrated on the pseudo-real cal set (what a real LOAO study
     would do), target = noise-free rho.
Usage: PYTHONPATH=. python capxfer/ideas/i1/plants.py <plant> [n_stage2]
"""
from __future__ import annotations

import json
import math
import pathlib
import pickle
import sys
import time

import numpy as np
import torch

from capxfer import assembly as A, datagen, featurize, sensors, sim, sysid
from capxfer.ideas.i1 import stage1 as S1
from capxfer.ideas.i1.stage2 import band, configs
from capxfer.learn import c2_eval, data as LD
from capxfer.pipeline.dryrun import TRUE_DOMAIN

OUT = pathlib.Path.home() / ".cache/capxfer/I1/plants"
LADDER = {"P1P5": ("P1", "P5"), "P1P5P2": ("P1", "P5", "P2")}
UNTESTED = {"P1P5P2": [4, 5, 6, 7, 10, 11, 12, 13], "P1P5": [2, 3, 4, 5, 6, 7, 10, 11, 12, 13]}
FIXED_KEYS = {"radius_scale", "mass_scale", "izz_scale", "stall_scale", "noload_scale", "kp_scale", "ki_scale"}

PLANTS = {
    # the integration dry-run plant (mu 0.55 etc.; partly outside DR_DEFAULT, inside DR_WIDE)
    "dryrun_true": dict(base=TRUE_DOMAIN, yaw=0.01),
    # actuation latency OUTSIDE the training DR (0.16 s > DR_WIDE max 0.12) plus a different drivetrain/floor point
    "latency_out": dict(base=dict(TRUE_DOMAIN, latency=0.16, mu=0.9, c_roller=1.2, noload_scale=1.08, v_batt=12.3,
                                  kp_scale=1.6, stall_scale=1.1), yaw=0.01),
    # model-form-like: wheel mounting yaw error 0.10 rad std (3.3x the DR max 0.03; sys-ID assumes 0)
    "yaw_misalign": dict(base=dict(TRUE_DOMAIN, mu=0.75, c_roller=4.5, s0=0.03, latency=0.05, noload_scale=1.0,
                                   v_batt=12.0), yaw=0.10),
}


def plant_domain(name, B, rng, device="cpu", jitter=0.02):
    P = PLANTS[name]
    d = {k: torch.tensor(v * (1 + (0.0 if k in FIXED_KEYS else jitter) * rng.standard_normal(B)),
                         dtype=torch.float32, device=device) for k, v in P["base"].items()}
    d["cg_dx"] = torch.tensor(rng.normal(0, 0.005, B), dtype=torch.float32, device=device)
    d["cg_dy"] = torch.tensor(rng.normal(0, 0.005, B), dtype=torch.float32, device=device)
    d["wheel_yaw_err_std"] = float(P["yaw"])
    return d


def generate(name, split, n_trials, seed):
    out = OUT / "data"
    if (out / f"{name}_{split}" / "manifest.json").exists() and \
            json.loads((out / f"{name}_{split}" / "manifest.json").read_text()).get("complete"):
        return out / f"{name}_{split}"
    orig = sim.sample_domain
    sim.sample_domain = lambda B, rng, dr=None, device="cpu": plant_domain(name, B, rng, device)
    try:
        if split == "cal":
            mix, sizes = {"random": 0.8, "single": 0.1, "pair": 0.1}, (1, 2, 3)
        else:
            mix, sizes = {"random": 0.75, "square4": 0.125, "L3": 0.125}, (4, 5, 6)
        datagen.generate(f"{name}_{split}", n_trials, mix, seed=seed, sizes=sizes, trials_per_asm=64,
                         phys_trials_per_asm=64, wheel_budget=4096, engine="graph", out_root=out, log=lambda *a: None)
    finally:
        sim.sample_domain = orig
    return out / f"{name}_{split}"


def load_feats(d):
    """Like learn.data._load_shard_A but keeps rho_twist (noise-free target)."""
    feats = []
    for p in sorted(pathlib.Path(d).glob("shard_*.pt")):
        for raw in datagen.load_shard(p):
            F = LD.from_featurize(featurize.featurize(raw, "E3", tuple(LD.PIDS)))
            lab = raw["labels"]
            F = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in F.items()}
            F["labels"] = dict(mode=lab["mode"].cpu().long(), g=lab["g"].cpu().float(), ood="none",
                               family=(f"phys_{lab['asm_type']}" if lab.get("asm_type") not in (None, "random")
                                       else lab.get("family")), asm=raw["meta"]["assembly"].name)
            F["rho"] = lab["rho_twist"].float().numpy()
            F["asm_dict"] = datagen.asm_to_dict(raw["meta"]["assembly"])
            feats.append(F)
    return feats


def run_sysid(name, params, seed=0):
    f = OUT / f"{name}_sysid.json"
    if f.exists():
        return json.loads(f.read_text())
    rng = np.random.default_rng(seed + 5)
    asm = A.physical("single", params)
    dom = plant_domain(name, 16, rng, "cuda")
    R = sim.simulate(asm, params, dom, torch.zeros(16, 4, dtype=torch.long), torch.ones(16, 4), device="cuda",
                     seed=seed + 7)
    raw = sensors.to_raw(R, params, dom, rng)
    raw["meta"]["labels"]["mode"] = torch.zeros(16, 4, dtype=torch.long)
    raw["meta"]["labels"]["ood"] = "none"
    fit_raw, ho_raw = sysid.subset_trials(raw, range(8)), sysid.subset_trials(raw, range(8, 16))
    t0 = time.time()
    fit = sysid.fit_params([fit_raw], params, iters=30, popsize=24, restarts=3, seed=seed, device="cuda")
    g = sysid.gate(fit, [ho_raw], params, device="cuda")
    res = dict(fit={k: fit[k] for k in ("best", "std", "use", "val", "loss")}, gate=g, secs=time.time() - t0,
               true={k: float(v) for k, v in PLANTS[name]["base"].items()}, yaw=PLANTS[name]["yaw"])
    f.write_text(json.dumps(res, indent=1, default=float))
    return res


def boxes(sres):
    fit = dict(best=sres["fit"]["best"], std=sres["fit"]["std"], use=sres["fit"]["use"])
    honest = sysid.dr_from_sysid(fit, gate_result=sres["gate"])
    out = dict(sysid=honest, prior=dict(sim.DR_WIDE))
    for f in (0.1, 0.2):                       # aggr20 declared after seeing dryrun_true only (see I1.md)
        aggr = dict(sim.DR_WIDE)
        if sres["gate"]["use"] == "fitted":
            for k, v in fit["best"].items():
                if k in sim.DR_WIDE:
                    lo, hi = sim.DR_WIDE[k]
                    h = f * (hi - lo)
                    aggr[k] = (max(sysid.FIT_BOUNDS[k][0], v - h), min(sysid.FIT_BOUNDS[k][1], v + h))
        out[f"aggr{int(round(f * 100))}"] = aggr
    return out


def predict(feats, pids, tag):
    recs = {}
    for meth in ("structured_npe", "ensemble"):
        model = c2_eval.load_model(meth, device="cuda", tag="W")
        r, _ = c2_eval.predict_set(model, feats, "E1", LADDER[pids], tag, 24, n_samples=128)
        recs[meth] = r
        del model
        torch.cuda.empty_cache()
    return recs


def as_D(r, feats):
    D = dict(P=np.nan_to_num(r["mode_prob"].numpy().astype(float)), mt=r["mode_true"].numpy(),
             n=r["n_wheels"].numpy(), yq=r["y_q"].numpy().astype(float), fam=np.array(r["family"]),
             gid=np.array(r["group_id"]), g_true=r["g_true"].numpy())
    D["rho"] = np.concatenate([F["rho"] for F in feats])
    D["grp"] = np.concatenate([np.full(F["rho"].shape[0], i) for i, F in enumerate(feats)])
    return D


def stage2(jobs, params, rng, M1, M2):
    M = M1 + M2
    Zs, by_n = {}, {}
    for ji, jb in enumerate(jobs):
        by_n.setdefault(jb[0].n, []).append(ji)
    for n, jis in sorted(by_n.items()):
        budget = 60000 // n
        chunk, cnt = [], 0
        for ji in jis + [None]:
            if ji is not None:
                chunk.append(ji)
                cnt += len(jobs[ji][1]) * M
            if chunk and (ji is None or cnt >= budget):
                asms, doms, modes, gs = [], [], [], []
                for c in chunk:
                    asm, cf, dom = jobs[c]
                    asms.append(asm)
                    idx = np.tile(np.arange(M), len(cf))
                    doms.append({k: (v[idx] if torch.is_tensor(v) else v) for k, v in dom.items()})
                    modes.append(torch.tensor(np.repeat(np.stack([m for m, _ in cf]), M, 0)))
                    gs.append(torch.tensor(np.repeat(np.stack([g for _, g in cf]), M, 0)))
                Rs = sim.simulate_multi(asms, params, doms, modes, gs, device="cuda", engine="graph", record_hz=50,
                                        store_device="cpu", seed=int(rng.integers(1 << 30)))
                for c, R in zip(chunk, Rs):
                    Zs[c] = sim.targets_from_twist(R)[0].cpu().numpy().reshape(len(jobs[c][1]), M, 14)
                del Rs
                chunk, cnt = [], 0
    return Zs


def main(name, n2=100, M1=30, M2=39, alpha1=0.05, alpha2=0.05, K=2, seed=0, box_list=("sysid", "aggr10", "aggr20")):
    torch.cuda.set_per_process_memory_fraction(0.12)
    torch.set_num_threads(4)                                           # shared-machine policy
    OUT.mkdir(parents=True, exist_ok=True)
    params = A.load_params(datagen.PARAMS)
    t0 = time.time()
    sres = run_sysid(name, params, seed)
    print(f"[{time.time() - t0:.0f}s] sysid {name}: gate {sres['gate']['gate_pass']} use={sres['gate']['use']} "
          f"e_fit={sres['gate']['e_fit']:.4f} e_nom={sres['gate']['e_nom']:.4f}; best "
          + " ".join(f"{k}={v:.3g}" for k, v in sres['fit']['best'].items()), flush=True)
    BX = boxes(sres)
    cal_dir = generate(name, "cal", 1280, seed + 101)
    te_dir = generate(name, "test", 768, seed + 202)
    print(f"[{time.time() - t0:.0f}s] data generated", flush=True)
    cache = OUT / f"{name}_preds.pkl"
    if cache.exists():
        P = pickle.load(open(cache, "rb"))
    else:
        fc, ft = load_feats(cal_dir), load_feats(te_dir)
        P = dict(asm_t=[F["asm_dict"] for F in ft])
        for pids in LADDER:
            rc, rt = predict(fc, pids, "cal"), predict(ft, pids, "test")
            P[pids] = dict(cal={m: as_D(r, fc) for m, r in rc.items()}, test={m: as_D(r, ft) for m, r in rt.items()})
        pickle.dump(P, open(cache, "wb"))
    print(f"[{time.time() - t0:.0f}s] predictions ready", flush=True)
    rng = np.random.default_rng(seed)
    results = dict(plant=name, sysid=sres, boxes={k: {kk: list(vv) if isinstance(vv, tuple) else vv
                                                      for kk, vv in v.items()} for k, v in BX.items()})
    alpha = alpha1 + alpha2
    for pids in LADDER:
        Dc, Dt = P[pids]["cal"]["structured_npe"], P[pids]["test"]["structured_npe"]
        Ec, Et = P[pids]["cal"]["ensemble"], P[pids]["test"]["ensemble"]
        ix = UNTESTED[pids]
        T = S1.calibrate(Dc, np.arange(len(Dc["n"])), alpha1, K=K, seed=seed)
        # stage-1 coverage on all test trials, by pool
        fam = Dt["fam"]
        pools = {"unseen (4-6 + square4)": np.flatnonzero(fam != "phys_L3"), "L3 (held-out type)": np.flatnonzero(fam == "phys_L3")}
        pools = {k: v for k, v in pools.items() if len(v)}
        R = {}
        for pn, idx in pools.items():
            r1 = S1.evaluate(Dt, idx, T, "sparse", K)
            R[pn] = dict(stage1=r1)
        # baselines on all test trials
        for bn, (Cc, Ct) in (("ENS-split", (Ec, Et)), ("SNPE-split", (Dc, Dt))):
            ql, qh = Cc["yq"][..., 0], Cc["yq"][..., 2]
            s = np.maximum(ql - Cc["rho"], Cc["rho"] - qh)
            w = np.maximum(qh - ql, 0.02)
            tpt = np.array([S1.cq(s[:, k], alpha) for k in range(14)])
            tj = S1.cq((s[:, ix] / w[:, ix]).max(1), alpha)
            for pn, idx in pools.items():
                tl, th = Ct["yq"][idx, :, 0], Ct["yq"][idx, :, 2]
                tw = np.maximum(th - tl, 0.02)
                rho = Ct["rho"][idx]
                R[pn][bn] = dict(pt_cov=float(((rho >= tl - tpt) & (rho <= th + tpt)).mean()),
                                 pt_w=float(np.median(th - tl + 2 * tpt)),
                                 joint_cov=float(((rho >= tl - tj * tw) & (rho <= th + tj * tw))[:, ix].all(1).mean()),
                                 joint_w=float(np.median((2 * tj * tw + th - tl)[:, ix])), n=len(idx))
        # stage 2 on a subsample
        sub = {pn: rng.choice(idx, size=min(n2 if pn.startswith("unseen") else n2 // 2, len(idx)), replace=False)
               for pn, idx in pools.items()}
        tsets = {"jun": ix}
        for bxn in box_list:
            jobs, meta = [], []
            for pn, idx in sub.items():
                for b in idx:
                    asm = datagen.asm_from_dict(P["asm_t"][Dt["grp"][b]])
                    n = asm.n
                    cf = configs(S1.sets_for(Dt, b, T, "sparse"), K)
                    dom = sim.sample_domain(M1 + M2, rng, BX[bxn])
                    jobs.append((asm, cf if cf else [(np.zeros(n, np.int64), np.ones(n, np.float32))], dom))
                    meta.append((pn, b, len(cf)))
            t1 = time.time()
            Zs = stage2(jobs, params, rng, M1, M2)
            for pn in sub:
                rows = [(ji, b, nc) for ji, (p_, b, nc) in enumerate(meta) if p_ == pn]
                lo, hi, jlo, jhi, ab = [], [], [], [], 0
                for ji, b, nc in rows:
                    cf = jobs[ji][1]
                    if nc == 0:
                        ab += 1
                        lo.append(np.full(14, -9.)); hi.append(np.full(14, 9.))
                        jlo.append(np.full(14, -9.)); jhi.append(np.full(14, 9.))
                        continue
                    Z = Zs[ji]
                    keys = [cf[i][0].tobytes() for i in range(nc)]
                    parts = [band(Z[[i for i in range(nc) if keys[i] == u]], M1, alpha2, tsets)
                             for u in dict.fromkeys(keys)]
                    lo.append(np.min([p["pt"][0] for p in parts], 0)); hi.append(np.max([p["pt"][1] for p in parts], 0))
                    jlo.append(np.min([p["jun"][0] for p in parts], 0)); jhi.append(np.max([p["jun"][1] for p in parts], 0))
                lo, hi, jlo, jhi = map(np.array, (lo, hi, jlo, jhi))
                bs = np.array([b for _, b, _ in rows])
                rho = Dt["rho"][bs]
                jc = ((rho >= jlo) & (rho <= jhi))[:, ix].all(1)
                R[pn][f"COMP-{bxn}"] = dict(pt_cov=float(((rho >= lo) & (rho <= hi)).mean()),
                                            pt_w=float(np.median(hi - lo)), joint_cov=float(jc.mean()),
                                            joint_w=float(np.median((jhi - jlo)[:, ix])), n=len(bs),
                                            abstain=ab / max(1, len(bs)),
                                            ncfg_med=float(np.median([nc for *_, nc in rows])))
                # baselines on the same subsample, for a matched comparison
                for bn, (Cc, Ct) in (("ENS-split@sub", (Ec, Et)),):
                    ql, qh = Cc["yq"][..., 0], Cc["yq"][..., 2]
                    s = np.maximum(ql - Cc["rho"], Cc["rho"] - qh)
                    w = np.maximum(qh - ql, 0.02)
                    tpt = np.array([S1.cq(s[:, k], alpha) for k in range(14)])
                    tj = S1.cq((s[:, ix] / w[:, ix]).max(1), alpha)
                    tl, th = Ct["yq"][bs, :, 0], Ct["yq"][bs, :, 2]
                    tw = np.maximum(th - tl, 0.02)
                    R[pn][bn] = dict(pt_cov=float(((rho >= tl - tpt) & (rho <= th + tpt)).mean()),
                                     joint_cov=float(((rho >= tl - tj * tw) & (rho <= th + tj * tw))[:, ix].all(1).mean()),
                                     joint_w=float(np.median((2 * tj * tw + th - tl)[:, ix])), n=len(bs))
            print(f"[{time.time() - t0:.0f}s] {pids} stage2 {bxn}: {len(jobs)} trials in {time.time() - t1:.0f}s",
                  flush=True)
        results[pids] = R
        for pn, rr in R.items():
            print(f"## {name} {pids} {pn}: stage1 all-wheel cov {rr['stage1']['cov']:.3f} (n={rr['stage1']['n']})")
            for k, v in rr.items():
                if k != "stage1":
                    print(f"   {k}: " + ", ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}"
                                               for a, b in v.items()))
    (OUT / f"{name}_results{'' if len(box_list) == 3 else '_' + '_'.join(box_list)}.json").write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100,
         **({"box_list": tuple(sys.argv[3].split(","))} if len(sys.argv) > 3 else {}))

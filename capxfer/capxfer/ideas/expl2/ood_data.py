"""EXPL2 OOD subsets (SIMULATION ONLY): pre-split evidence + post-split truth for the WS-A OOD sets.

For every multi-module trial of dev_ood_payload_v3 (2-6 kg payload on a lifter module) and dev_ood_lowmu_v3
(world-fixed low-mu sheet under one side), the robot probes the ASSEMBLY (E1, P1+P5+P2) exactly as in the SPLIT
benchmark: the W-generation structured_ensemble gives per-wheel mode posteriors and the W ensemble gives the
current-body capability quantiles. The assembly then splits into singles. Post-split truth is simulated per module at
the trial's true plant and health:
  payload : the payload stays on its lifter module (same mass, offset rotated into the module frame); other modules
            carry nothing.
  low_mu  : each module keeps the per-wheel friction factor its wheels had on the sheet at the start of the trial
            (sheet frozen under the module; modules entirely off the sheet are unaffected).
A paired twin without the OOD condition (same plant, health, seed) gives the in-model truth, so every module gets
`ood_consequential` = max_k |rho_ood - rho_twin| > 0.1 (the WS-A definition).

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.ideas.expl2.ood_data
Writes ~/.cache/capxfer/EXPL2/ood_{payload,low_mu}.npz.
"""
from __future__ import annotations

import json
import math
import pathlib
import time

import numpy as np
import torch

from capxfer import datagen, featurize, sim
from capxfer.ideas.split import common as C, sims as SM
from capxfer.learn import c2_eval, data as LD
from capxfer.probes import PIDS

OUT = pathlib.Path.home() / ".cache/capxfer/EXPL2"
SETS = {"payload": "dev_ood_payload_v3", "low_mu": "dev_ood_lowmu_v3"}
VIEW = ("P1", "P5", "P2")
CONSEQUENTIAL = 0.1


def predict(models, raw):
    F = LD.from_featurize(featurize.featurize(raw, "E3", PIDS))
    F = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in F.items()}
    v = LD.apply_view(F, "E1", VIEW)
    with torch.no_grad():
        ps = models["structured_ensemble"].predict(v, 64)
        pe = models["ensemble"].predict(v, 256)
    ys = pe["y_samples"].float().cpu().numpy()
    zs = np.minimum(ys[..., 2], ys[..., 3])
    return dict(P=np.nan_to_num(ps["mode_prob"].float().cpu().numpy()), g_mean=ps["g_mean"].float().cpu().numpy(),
                bb_q05z=np.quantile(zs, 0.05, 1), bb_q50z=np.quantile(zs, 0.5, 1),
                bb_yq=pe["y_q"].float().cpu().numpy())


def main():
    SM.setup()
    torch.cuda.set_per_process_memory_fraction(0.08)
    c2_eval.MEM_FRAC = 0.08
    prm = C.params()
    models = {m: c2_eval.load_model(m, device="cuda", tag="W") for m in ("structured_ensemble", "ensemble")}
    t0 = time.time()
    for kind, name in SETS.items():
        mp = pathlib.Path.home() / f".cache/capxfer/A/{name}/manifest.json"
        man = json.loads(mp.read_text())
        trials, mods = [], []          # per-trial and per-module records
        jobs = {}                      # (mid, has_payload) -> list of per-module sim inputs
        gi = 0
        for s in man["shards"]:
            for raw in datagen.load_shard(mp.parent / s["file"]):
                asm = raw["meta"]["assembly"]
                nmod = len(asm.modules)
                if nmod < 2:
                    continue
                lab = raw["labels"]
                B = lab["mode"].shape[0]
                pr = predict(models, raw)
                dom = lab["domain"]
                for b in range(B):
                    t = dict(gi=gi, b=b, nmod=nmod, n=4 * nmod, modules=[tuple(m) for m in asm.modules],
                             family=str(lab.get("family")), asm_type=str(lab.get("asm_type")),
                             mode=lab["mode"][b].numpy().astype(np.int64), g=lab["g"][b].numpy().astype(np.float32),
                             rho=lab["rho_twist"][b].numpy().astype(float),
                             rho_noood=lab["rho_twist_noood"][b].numpy().astype(float),
                             theta={k: float(dom[k][b]) for k in C.PLANT_KEYS},
                             yaw_std=float(dom["wheel_yaw_err_std"][b]),
                             P=pr["P"][b], g_mean=pr["g_mean"][b], bb_q05z=float(pr["bb_q05z"][b]),
                             bb_q50z=float(pr["bb_q50z"][b]), bb_yq=pr["bb_yq"][b])
                    ti = len(trials)
                    trials.append(t)
                    for k, (mid, _, _, yk) in enumerate(asm.modules):
                        pay = None
                        mf = np.ones(4, np.float32)
                        if kind == "payload" and int(dom["payload_module"][b]) == k:
                            th = -yk * math.pi / 2
                            dx, dy = float(dom["payload_dx"][b]), float(dom["payload_dy"][b])
                            pay = (float(dom["payload_mass"][b]), math.cos(th) * dx - math.sin(th) * dy,
                                   math.sin(th) * dx + math.cos(th) * dy)
                        if kind == "low_mu":
                            mf = dom["mu_factor"][b, 4 * k:4 * k + 4].numpy().astype(np.float32)
                        mi = len(mods)
                        mods.append(dict(t=ti, mod=k, mid=mid, payload=pay is not None, mu_min=float(mf.min())))
                        jobs.setdefault((mid, pay is not None), []).append(
                            (mi, t["theta"], t["yaw_std"], t["mode"][4 * k:4 * k + 4], t["g"][4 * k:4 * k + 4], pay, mf))
                gi += 1
                print(f"[{time.time() - t0:.0f}s] {kind} group {gi} nmod={nmod} B={B}", flush=True)
        rho = np.zeros((len(mods), 14))
        rho_tw = np.zeros((len(mods), 14))
        for (mid, hp), js in sorted(jobs.items()):
            for s0 in range(0, len(js), 4096):
                jj = js[s0:s0 + 4096]
                dom = SM.dom_from([j[1] for j in jj], yaw_std=float(np.mean([j[2] for j in jj])))
                mode = torch.tensor(np.stack([j[3] for j in jj]))
                g = torch.tensor(np.stack([j[4] for j in jj]))
                payload = None
                if hp:
                    payload = {k: torch.tensor([j[5][i] for j in jj], dtype=torch.float32)
                               for i, k in enumerate(("mass", "x", "y"))}
                mfac = torch.tensor(np.stack([j[6] for j in jj]))
                seed = 777 + C.MIDS.index(mid) * 10 + int(hp) + s0
                asm1 = C.single_asm(mid, prm)
                kw = dict(device="cuda", engine="graph", record_hz=50, store_device="cpu", seed=seed)
                R1 = sim.simulate_multi([asm1], prm, [dict(dom)], [mode], [g], payloads=[payload] if hp else None,
                                        mu_factors=[mfac] if kind == "low_mu" else None, **kw)[0]
                R0 = sim.simulate_multi([asm1], prm, [dict(dom)], [mode], [g], **kw)[0]
                idx = np.array([j[0] for j in jj])
                rho[idx] = sim.targets_from_twist(R1)[0].cpu().numpy()
                rho_tw[idx] = sim.targets_from_twist(R0)[0].cpu().numpy()
                del R1, R0
                print(f"[{time.time() - t0:.0f}s] {kind} truth {mid} payload={hp} {s0 + len(jj)}/{len(js)}", flush=True)
        nmax = 24
        pad = lambda a, fill: np.stack([np.concatenate([x, np.full((nmax - len(x),) + x.shape[1:], fill)]) for x in a])
        np.savez_compressed(
            OUT / f"ood_{kind}.npz", kind=kind, dataset=name, source="SIMULATION",
            t_gi=np.array([t["gi"] for t in trials]), t_nmod=np.array([t["nmod"] for t in trials]),
            t_modules=np.array([json.dumps(t["modules"]) for t in trials]),
            t_family=np.array([t["family"] for t in trials]), t_asm_type=np.array([t["asm_type"] for t in trials]),
            t_mode=pad([t["mode"] for t in trials], -1), t_g=pad([t["g"] for t in trials], np.nan),
            t_P=pad([t["P"] for t in trials], 0.0), t_gmean=pad([t["g_mean"] for t in trials], np.nan),
            t_rho=np.stack([t["rho"] for t in trials]), t_rho_noood=np.stack([t["rho_noood"] for t in trials]),
            t_bb_q05z=np.array([t["bb_q05z"] for t in trials]), t_bb_q50z=np.array([t["bb_q50z"] for t in trials]),
            t_bb_yq=np.stack([t["bb_yq"] for t in trials]),
            m_t=np.array([m["t"] for m in mods]), m_mod=np.array([m["mod"] for m in mods]),
            m_mid=np.array([m["mid"] for m in mods]), m_payload=np.array([m["payload"] for m in mods]),
            m_mu_min=np.array([m["mu_min"] for m in mods]), m_rho=rho, m_rho_twin=rho_tw,
            m_consequential=np.abs(rho - rho_tw).max(1) > CONSEQUENTIAL)
        print(f"wrote ood_{kind}.npz trials={len(trials)} modules={len(mods)}", flush=True)


if __name__ == "__main__":
    main()

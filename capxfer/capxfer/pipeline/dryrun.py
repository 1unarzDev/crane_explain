"""SIMULATION-labelled pseudo-real collection for the integration dry run (never field data).

  python -m capxfer.pipeline.dryrun OUT_DIR [--scale small|tiny] [--seed 1]

1. WS-B `field.schedule.write` builds a reduced sealed schedule (4 assembly types, F0-F5 automated + F4/F6/F7 operator
   blocks); the seed goes to OUT_DIR/sealed/ (separate from the schedule, as in the real protocol).
2. WS-B `field.synth_session.make_session` writes one FieldLog v2 session per assembly through FieldLog's own writer;
   the verifier labels them SIMULATION. The simulated "true" domain is a fixed, deliberately perturbed point (TRUE_DOMAIN,
   partly outside the training DR) plus small per-trial jitter, to mimic a sim-to-real gap.
3. A separate healthy single-module sys-ID session (as the real protocol collects before the main collection).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from capxfer import sim as S
from capxfer.field import schedule as SCH
from capxfer.field import synth_session as SS

from . import adapters as AD

# "true" plant for the dry run: different from the training DR centre; mu below and s0/latency at the upper edge of DR
TRUE_DOMAIN = dict(mu=0.55, s0=0.045, c_roller=3.2, crr=0.022, tau_bd=0.9, latency=0.075, stall_scale=0.86,
                   noload_scale=0.94, mass_scale=1.05, izz_scale=1.18, radius_scale=0.985, kp_scale=0.85,
                   ki_scale=0.8, v_batt=11.6)
JITTER = 0.02          # relative per-trial jitter around TRUE_DOMAIN
CG_STD = 0.005

SCALES = {
    # per assembly: automated counts and operator (blocks, trials per block); ~54 trials per assembly type.
    # D3: one 10-trial F6 and one 10-trial F7 block (5 F6w/F7w each); L3/square4 run their F6 block as F7 (schedule).
    # D3b/c: single gets 2+2 OOD blocks with 9 of 10 F6w/F7w and 2 fewer F1 (the full design's per-type override,
    # scaled); 8 field sessions, each with blocks of every type (auto blocks of 5 so every type has >= 8 blocks).
    "small": dict(counts={"F0": 8, "F1": 8, "F2": 4, "F3": 4, "F5": 6}, operator={"F4": (1, 4), "F6": (1, 10), "F7": (1, 10)},
                  auto_block=5, sysid_trials=8, n_sessions=8,
                  type_overrides={"single": {"counts": {"F1": 6}, "operator": {"F6": [2, 10], "F7": [2, 10]},
                                             "ood_w_per_10": 9}}),
    "small_v0": dict(counts={"F0": 8, "F1": 8, "F2": 4, "F3": 4, "F5": 6}, operator={"F4": (1, 4), "F6": (1, 3), "F7": (1, 3)},
                     auto_block=10, sysid_trials=8),
    "tiny": dict(counts={"F0": 2, "F1": 2, "F2": 1, "F3": 1, "F5": 1}, operator={"F4": (1, 1), "F6": (1, 1), "F7": (1, 1)},
                 auto_block=4, sysid_trials=4),
}


def true_domain(B, rng, device="cpu", base=None, jitter=JITTER):
    base = dict(TRUE_DOMAIN, **(base or {}))
    # Fixed physical constants of one build (wheel radius, mass, inertia, motor constants) do not vary trial to trial;
    # only operating conditions (battery voltage, friction, latency) jitter.
    fixed = {"radius_scale", "mass_scale", "izz_scale", "stall_scale", "noload_scale", "kp_scale", "ki_scale"}
    d = {k: torch.tensor(v * (1 + (0.0 if k in fixed else jitter) * rng.standard_normal(B)), dtype=torch.float32,
                         device=device) for k, v in base.items()}
    d["cg_dx"] = torch.tensor(rng.normal(0, CG_STD, B), dtype=torch.float32, device=device)
    d["cg_dy"] = torch.tensor(rng.normal(0, CG_STD, B), dtype=torch.float32, device=device)
    d["wheel_yaw_err_std"] = 0.01
    return d


@contextlib.contextmanager
def perturbed_domain(base=None):
    """M6: synth_session.make_session draws its domain from sim.sample_domain (the TRAINING DR) and has no `dom=`
    argument, so a perturbed true plant cannot be passed in. Patch sim.sample_domain for the duration of the call."""
    orig = S.sample_domain
    S.sample_domain = lambda B, rng, dr=None, device="cpu": true_domain(B, rng, device, base)
    try:
        yield
    finally:
        S.sample_domain = orig


def sim_faults_d3(trials, asm, floor_mu=None):
    """D3 replacement for synth_session._sim_faults (WS-B; patched during the dry run only, like perturbed_domain):
    payload at the scheduled lifter-deck CG offset (module centre + dx_m/dy_m, assembly frame) and the F7 sheet as a
    PER-WHEEL friction factor (sim (B,n) mu_factor, WS-A2): wheels on the scheduled side of the nominal CG
    (datagen.patch_wheels) get factor mu_sheet / floor mu, the others 1. Schedule entries without the D3 fields fall
    back to the old behaviour (payload at the module centre, global factor 0.3)."""
    from capxfer import datagen as DG
    floor_mu = float(TRUE_DOMAIN["mu"] if floor_mu is None else floor_mu)
    B, n = len(trials), asm.n
    mode = torch.zeros(B, n, dtype=torch.long)
    g = torch.ones(B, n)
    pm, px, py = torch.zeros(B), torch.zeros(B), torch.zeros(B)
    mu = torch.ones(B, n)
    code = {"weak": S.WEAK, "coast": S.COAST, "brake": S.BRAKE, "roller_locked": S.ROLLER}
    mods = [m for m, *_ in asm.modules]
    for b, tr in enumerate(trials):
        for f in tr["faults"]:
            if f["mode"] in code:
                w = asm.wheel_corner.index(f["wheel"])
                mode[b, w] = code[f["mode"]]
                if f["mode"] == "weak":
                    g[b, w] = float(f["g"])
            elif f["mode"] == "payload":
                k = mods.index(f["module"])
                pm[b] = float(f["mass_kg"])
                px[b] = float(asm.module_pos[k, 0]) + float(f.get("dx_m", 0.0))
                py[b] = float(asm.module_pos[k, 1]) + float(f.get("dy_m", 0.0))
            elif f["mode"] == "low_mu":
                if "mu" in f:
                    on = torch.as_tensor(DG.patch_wheels(asm, f.get("region", "left")))
                    mu[b] = torch.where(on, torch.tensor(min(1.0, float(f["mu"]) / floor_mu)), torch.tensor(1.0))
                else:
                    mu[b] = 0.3
    payload = dict(mass=pm, x=px, y=py) if (pm > 0).any() else None
    return mode, g, payload, (mu if (mu < 1).any() else None)


@contextlib.contextmanager
def d3_ood_faults():
    orig = SS._sim_faults
    SS._sim_faults = sim_faults_d3
    try:
        yield
    finally:
        SS._sim_faults = orig


def make_collection(out_dir, scale="small", seed=1, device=None, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sc = SCALES[scale]
    device = device or AD.gpu_budget()
    t0 = time.time()
    seed_hex = f"{seed:032x}"
    w = SCH.write(out / "schedule", seed_hex=seed_hex, seed_dir=out / "sealed", counts=sc["counts"],
                  operator=sc["operator"], auto_block=sc["auto_block"], type_overrides=sc.get("type_overrides", {}),
                  n_sessions=sc.get("n_sessions", 1))
    sched = SCH.load(w["schedule"])
    log(f"schedule {w['sha256'][:12]} trials {sched['n_trials']}")
    sess_dirs = []
    with perturbed_domain(), d3_ood_faults():
        # D3c: every field session holds blocks of all types. WS-B's synthetic writer writes one assembly per FieldLog
        # folder, so each (field session, assembly) part goes to its own folder; the pipeline clusters on the scheduled
        # field session (adapters.trial_meta), so the folder split does not change the analysis.
        k = 0
        for fs in SCH.field_sessions(sched):
            for asm_name in fs["assemblies"]:
                sub = SCH.sub_schedule(sched, fs["id"], asm_name)
                n = sub["n_trials"]
                t1 = time.time()
                d, _ = SS.make_session(out / "sessions", asm_name, n_trials=n, seed=seed * 100 + k, schedule=sub,
                                       current_wired=True, device=device)
                k += 1
                sess_dirs.append(str(d))
                log(f"session {fs['id']}/{asm_name}: {n} trials -> {Path(d).name} ({time.time() - t1:.0f} s)")
        # separate healthy single-module sys-ID session (own one-assembly schedule with only F0 trials)
        ss = SCH.generate(f"{seed + 7:032x}", assemblies=("single",),
                          counts={"F0": sc["sysid_trials"], "F1": 0, "F2": 0, "F3": 0, "F5": 0},
                          operator={"F4": (0, 1), "F6": (0, 1), "F7": (0, 1)}, n_sessions=1)
        (out / "sysid_schedule").mkdir(exist_ok=True)
        raw = SCH.canonical(ss)
        (out / "sysid_schedule" / "schedule.json").write_bytes(raw)
        (out / "sysid_schedule" / "schedule.sha256").write_text(f"{SCH.sha256_bytes(raw)}  schedule.json\n")
        d, _ = SS.make_session(out / "sysid_sessions", "single", n_trials=sc["sysid_trials"], seed=seed * 100 + 99,
                               schedule=ss, current_wired=True, device=device)
    info = dict(schedule=str(w["schedule"]), schedule_sha256=w["sha256"], sessions=sess_dirs, sysid_sessions=[str(d)],
                sysid_schedule=str(out / "sysid_schedule" / "schedule.json"), true_domain=TRUE_DOMAIN, jitter=JITTER,
                scale=scale, seed=seed, wall_s=time.time() - t0, label="SIMULATION")
    (out / "dryrun_collection.json").write_text(json.dumps(info, indent=1))
    log(f"collection written in {info['wall_s']:.0f} s")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--scale", default="small", choices=sorted(SCALES))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    make_collection(a.out_dir, a.scale, a.seed)


if __name__ == "__main__":
    main()

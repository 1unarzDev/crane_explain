"""Local stand-in for WS-A's sensors/datagen and WS-B's featurize (CONTRACTS rule 2).

Generates Feat dicts (CONTRACTS C2) straight from `sim.simulate` ground-truth streams with a simple sensor model and a
simple 20 Hz featurizer. Features are produced for ALL seven primitives and ALL rungs; `data.apply_view` then zeroes
the primitives / rungs a given evaluation is not allowed to see (that is exactly what featurize(raw, rung, pids) does
when its layout is "fixed-length block per primitive, LIBRARY order").

Stub layout (data.LAYOUT documents it; swap when featurize.py lands):
  wheel_x  (B,n,L,4): [enc_speed rad/s, duty, current A, ref_speed rad/s]
  body_x   (B,L,9):   [cmd vx, cmd vy, cmd wz, gyro_z, acc_x, acc_y, lidar vx, lidar vy, lidar wz]
  L = sum over primitives of 2 legs x round(20 Hz x leg_duration) = 576

CLI (one worker of a sharded dataset):
  python -m capxfer.learn._stub --name train --n-asm 120 --B 256 --sizes 1,2,3 --worker 0 --nworkers 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from capxfer import assembly as A
from capxfer import sim
from capxfer.probes import LIBRARY, PIDS, Schedule, leg_duration

ROOT = Path(os.path.expanduser("~/.cache/capxfer/C"))
PARAMS = Path(__file__).resolve().parents[2] / "params" / "module_measured.yaml"
FEAT_HZ = 20
SEG_LEN = {p.pid: int(round(leg_duration(p) * FEAT_HZ)) for p in LIBRARY}   # per leg
WHEEL_CH = ("speed", "duty", "current", "ref")
BODY_CH = ("cmd_vx", "cmd_vy", "cmd_wz", "gyro_z", "acc_x", "acc_y", "lidar_vx", "lidar_vy", "lidar_wz")
RUNG_BODY = {"E0": 3, "E1": 6, "E2": 9, "E3": 9}          # number of body channels visible (cumulative)
NOMINAL_DR = dict(mu=(0.8, 0.8), s0=(0.03, 0.03), c_roller=(2.0, 2.0), crr=(0.015, 0.015), tau_bd=(0.9, 0.9),
                  latency=(0.04, 0.04), stall_scale=(1.0, 1.0), noload_scale=(1.0, 1.0), mass_scale=(1.0, 1.0),
                  izz_scale=(1.0, 1.0), radius_scale=(1.0, 1.0), cg_off=0.0, kp_scale=(1.0, 1.0),
                  ki_scale=(1.0, 1.0), v_batt=(11.1, 11.1), wheel_yaw_err=0.0)


def load_params():
    return A.load_params(PARAMS)


def params_hash():
    import hashlib
    return hashlib.sha256(PARAMS.read_bytes()).hexdigest()


def seg_slices(pids=PIDS):
    """Time slices of each primitive's block in the stub layout (all 7 primitives present)."""
    out, t = {}, 0
    for p in LIBRARY:
        L = 2 * SEG_LEN[p.pid]
        out[p.pid] = slice(t, t + L)
        t += L
    return out


def wheel_geom(asm):
    cg = asm.cg()
    p = asm.wheel_pos - cg
    Lc = asm.char_length()
    return np.concatenate([p, asm.w_hat, asm.r_hat, np.linalg.norm(p, axis=1, keepdims=True) / Lc,
                           asm.wheel_module[:, None] / 4.0], 1).astype(np.float32)


def capability_targets(twist, t, sched, Lc):
    """y, drift (B,14) from body-frame twist (B,T,3) over each hold window (stub: true twist, not AprilTag diff)."""
    w = torch.tensor([1.0, 1.0, Lc * Lc], device=twist.device)
    ys, ds = [], []
    for (pid, leg, ta, tb, tw) in sched.windows():
        m = (t >= ta) & (t <= tb)
        xi = twist[:, m].mean(1)                                    # (B,3)
        xc = torch.tensor(tw, dtype=torch.float32, device=twist.device)
        nn2 = (xc * xc * w).sum()
        rho = (xi * xc * w).sum(-1) / nn2
        res = xi - rho[:, None] * xc
        ys.append(rho)
        ds.append(torch.sqrt((res * res * w).sum(-1) / nn2))
    return torch.stack(ys, 1), torch.stack(ds, 1)


def featurize_sim(R, asm, params, rng, sched, noise=True):
    """Simple sensor model + 20 Hz featurizer on ground-truth sim streams -> Feat (all pids, all rungs)."""
    nom = sim.nominal(params)
    sp = params["sensors"]
    dev = R["twist"].device
    B, T, n = R["omega"].shape
    t = R["t"] + 0.01                                            # record k is the state after step k+1
    cpr = nom["cpr"]
    counts = torch.floor(R["wheel_angle"] * cpr / (2 * math.pi)) if noise else R["wheel_angle"] * cpr / (2 * math.pi)
    pad = torch.cat([counts[:, :1].expand(B, 5, n), counts], 1)
    speed100 = (pad[:, 5:] - pad[:, :-5]) / cpr * 2 * math.pi / 0.05
    cmd_pub = torch.tensor(sched.twist(t.cpu().numpy()), dtype=torch.float32, device=dev)  # (T,3) as published
    ref100 = torch.tensor(asm.ik(cmd_pub.cpu().numpy(), nom["radius"]), dtype=torch.float32, device=dev)  # (T,n)
    kt = params["motor"]["stall_torque_nm_12v"] / params["motor"]["stall_current_a_12v"]
    cur = R["torque"] / kt
    g = lambda *s: torch.tensor(rng.normal(size=s), dtype=torch.float32, device=dev)
    gz = R["twist"][..., 2].clone()
    acc = R["accel"].clone()
    if noise:
        cur = cur + sp["current_noise_a"] * g(B, T, n)
        gz = gz + sp["imu_gyro_noise_radps"] * g(B, T) + 0.003 * g(B, 1)
        acc = acc + sp["imu_acc_noise_mps2"] * g(B, T, 2)
    if asm.has_lidar():
        lat = int(round(sp["lidar_odom_latency_s"] * 100))
        hold = 100 // sp["lidar_odom_rate_hz"]
        idx = torch.arange(T, device=dev)
        src = torch.clamp((idx - lat) // hold * hold, min=0)
        lid = R["twist"][:, src].clone()
        if noise:
            lid = lid + torch.tensor([sp["lidar_odom_noise_mps"]] * 2 + [0.02], device=dev) * g(B, T, 3)
        lid[:, idx < lat] = 0.0
    else:
        lid = torch.zeros(B, T, 3, device=dev)
    # resample each segment onto the 20 Hz grid
    sel = []
    for s in sched.segments:
        L = SEG_LEN[s.pid]
        tt = s.t0 + (np.arange(L) + 1) / FEAT_HZ
        sel.append(np.clip(np.round(tt * 100).astype(int) - 1, 0, T - 1))
    sel = torch.tensor(np.concatenate(sel), device=dev)
    wheel_x = torch.stack([speed100[:, sel], R["duty"][:, sel], cur[:, sel], ref100[sel][None].expand(B, -1, -1)],
                          -1).permute(0, 2, 1, 3)                    # (B,n,L,4)
    body_x = torch.cat([cmd_pub[sel][None].expand(B, -1, -1), gz[:, sel, None], acc[:, sel], lid[:, sel]], -1)
    Lc = asm.char_length()
    y, drift = capability_targets(R["twist"], t, sched, Lc)
    return dict(wheel_x=wheel_x, wheel_geom=torch.tensor(wheel_geom(asm), device=dev).expand(B, n, 8).clone(),
                body_x=body_x, y=y, drift=drift, n_wheels=torch.full((B,), n, dtype=torch.long, device=dev))


def simulate_feat(asm, params, B, rng, dr=None, ood="none", device="cuda", noise=True, mode=None, g=None, dt=0.005,
                  seed=0):
    """Run B trials of one assembly and return a CPU Feat dict (all pids, all rungs, labels attached)."""
    dom = sim.sample_domain(B, rng, dr, device=device)
    if mode is None:
        mode, g = sim.sample_faults(B, asm.n, rng)
    payload = mu_factor = None
    if ood == "payload":
        cells = asm.module_pos
        k = rng.integers(len(cells), size=B)
        off = rng.uniform(-0.1, 0.1, size=(B, 2))
        payload = dict(mass=torch.tensor(rng.uniform(1.5, 3.0, B), dtype=torch.float32),
                       x=torch.tensor(cells[k, 0] + off[:, 0], dtype=torch.float32),
                       y=torch.tensor(cells[k, 1] + off[:, 1], dtype=torch.float32))
    elif ood == "low_mu":
        mu_factor = torch.tensor(rng.uniform(0.2, 0.4, B), dtype=torch.float32)
    sched = Schedule(asm.char_length())
    R = sim.simulate(asm, params, dom, mode, g, schedule=sched, dt=dt, device=device, payload=payload,
                     mu_factor=mu_factor, seed=seed)
    F = featurize_sim(R, asm, params, rng, sched, noise=noise)
    F = {k: v.cpu() for k, v in F.items()}
    F["wheel_x"] = F["wheel_x"].half()
    F["body_x"] = F["body_x"].half()
    F["labels"] = dict(mode=mode.cpu(), g=g.cpu(), ood=ood, params_sha256=params_hash(), family=asm.meta.get("family", asm.name), asm=asm.name,
                       modules=list(asm.modules),
                       domain={k: (v.cpu() if torch.is_tensor(v) else v) for k, v in dom.items()})
    return F


# ----------------------------------------------------------------------------- sharded dataset generation
PHYSICAL = ("single", "pair", "L3", "square4")


def make_assemblies(spec, seed):
    params = load_params()
    rng = np.random.default_rng(seed)
    out = []
    if spec.get("physical"):
        for name in spec["physical"]:
            a = A.physical(name, params)
            a.meta["family"] = f"phys_{name}"
            out.append(a)
    fams = spec.get("families")
    for _ in range(spec.get("n_asm", 0)):
        out.append(A.random_assembly(rng, params, sizes=tuple(spec["sizes"]), families=fams))
    return out


def generate(name, spec, worker=0, nworkers=1, device="cuda", mem_frac=0.06):
    """Writes ROOT/data/<name>/shard_<i>.pt (one per assembly) and, from worker 0 at the end, manifest.json."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(mem_frac)
    params = load_params()
    out = ROOT / "data" / name
    out.mkdir(parents=True, exist_ok=True)
    ph = params_hash()
    (out / "spec.json").write_text(json.dumps(dict(spec, params_sha256=ph), indent=1))
    asms = make_assemblies(spec, spec.get("seed", 0))
    oods = spec.get("ood", ["none"])
    jobs = [(i, a, o) for i, a in enumerate(asms) for o in oods]
    for j, (i, a, o) in enumerate(jobs):
        if j % nworkers != worker:
            continue
        f = out / f"shard_{i:04d}_{o}_{ph[:8]}.pt"
        if f.exists():
            continue
        rng = np.random.default_rng([spec.get("seed", 0), i, zlib.crc32(o.encode())])
        t0 = time.time()
        for attempt in range(3):
            try:
                F = simulate_feat(a, params, spec["B"], rng, dr=spec.get("dr"), ood=o, device=device,
                                  seed=int(rng.integers(1 << 30)))
                break
            except Exception as e:  # sim.py may be mid-edit by WS-A; retry
                print("retry", i, o, repr(e), flush=True)
                time.sleep(20)
        else:
            continue
        torch.save(F, f.with_suffix(".tmp"))
        os.replace(f.with_suffix(".tmp"), f)
        print(f"{name} shard {i} {o} n={a.n} fam={a.meta.get('family')} {time.time() - t0:.1f}s", flush=True)
    return out


def write_manifest(name):
    out = ROOT / "data" / name
    ph = params_hash()
    shards = sorted(str(p) for p in out.glob(f"shard_*_{ph[:8]}.pt"))
    (out / "manifest.json").write_text(json.dumps(dict(source="C_stub", params_sha256=ph, shards=shards), indent=1))
    return out / "manifest.json"


DATASETS = {
    # training families: sizes 1-3 (includes the single, pair and L3 physical types' families)
    "train": dict(n_asm=240, B=192, sizes=[1, 2, 3], seed=1),
    "val": dict(n_asm=16, B=128, sizes=[1, 2, 3], seed=2),
    # held-out families: sizes 4-6 (square4's family k4_bb2x2 is in here)
    "test_heldout": dict(n_asm=36, B=128, sizes=[4, 5, 6], seed=3),
    "test_physical": dict(n_asm=0, physical=list(PHYSICAL), B=256, seed=4, ood=["none", "payload", "low_mu"]),
    "train_nominal": dict(n_asm=240, B=192, sizes=[1, 2, 3], seed=1, dr=NOMINAL_DR),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--manifest", action="store_true")
    a = ap.parse_args()
    if a.manifest:
        print(write_manifest(a.name))
    else:
        generate(a.name, DATASETS[a.name], a.worker, a.nworkers)

"""Sharded simulated RawTrial datasets (CONTRACTS C1) for training / held-out evaluation.

    python -m capxfer.datagen --name dev --n-trials 20000 --mix physical=0.3,random=0.7 --seed 0
    python -m capxfer.datagen --name ood_payload --n-trials 4000 --ood payload --mix physical=1
    python -m capxfer.datagen --name heldout --n-trials 4000 --only-families k4_bb2x2,k5_bb2x3

Output: ~/.cache/capxfer/A/<name>/shard_XXXXX.pt (a list of compact RawTrials, one per assembly group) and
manifest.json (counts, seeds, params sha256, code hash, args). Load with `load_shard(path)` -> list[RawTrial] with
full dtypes and live `Assembly` / `Schedule` objects restored.

Compact storage: counts int32, duty/gyro/acc/lidar/current float16, truth_pose float32; cmd and t are rebuilt from
the schedule on load (cmd is deterministic: the published probe twist). Float16 resolution (≈1e-3 relative) is well
below every modelled noise level.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import time

import numpy as np
import torch

from . import assembly as A
from . import sensors, sim
from .probes import PIDS, Schedule

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARAMS = ROOT / "params" / "module_measured.yaml"
OUT = pathlib.Path(os.path.expanduser("~/.cache/capxfer/A"))
CODE_FILES = ("capxfer/sim.py", "capxfer/sensors.py", "capxfer/datagen.py", "capxfer/assembly.py", "capxfer/probes.py")
HALF = ("duty", "gyro_z", "acc", "lidar_twist", "current")
CONSEQUENTIAL = 0.1               # |delta rho| on any target that changes a capability decision


def sha256_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def code_hash():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(f.encode())
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- (de)serialisation
def asm_to_dict(a):
    return dict(name=a.name, modules=[list(m) for m in a.modules], wheel_pos=a.wheel_pos, w_hat=a.w_hat, r_hat=a.r_hat,
                wheel_module=a.wheel_module, wheel_corner=list(a.wheel_corner), module_mass=a.module_mass,
                module_pos=a.module_pos, lidar=a.lidar, side=a.side, meta=dict(a.meta))


def asm_from_dict(d):
    return A.Assembly(name=d["name"], modules=[tuple(m) for m in d["modules"]], wheel_pos=np.asarray(d["wheel_pos"]),
                      w_hat=np.asarray(d["w_hat"]), r_hat=np.asarray(d["r_hat"]),
                      wheel_module=np.asarray(d["wheel_module"]), wheel_corner=list(d["wheel_corner"]),
                      module_mass=np.asarray(d["module_mass"]), module_pos=np.asarray(d["module_pos"]),
                      lidar=np.asarray(d["lidar"]), side=float(d["side"]), meta=dict(d["meta"]))


def compact(raw):
    """RawTrial -> storage dict (CPU, reduced precision, plain python/numpy metadata only)."""
    m = raw["meta"]
    c = {k: raw[k].detach().half().cpu() for k in HALF}
    c["counts"] = raw["counts"].detach().int().cpu()
    c["truth_pose"] = raw["truth_pose"].detach().cpu().float()
    lab = m["labels"]
    c["meta"] = dict(assembly=asm_to_dict(m["assembly"]), cpr=m["cpr"], char_length=m["char_length"],
                     pids=[s.pid for s in m["schedule"].segments[::2]], source=m["source"],
                     trial_ids=list(m["trial_ids"]), n=m["n"], dt=m.get("dt"), T=int(raw["t"].shape[0]),
                     t_step=float(raw["t"][1] - raw["t"][0]), extra=m.get("extra", {}))
    c["labels"] = dict(mode=lab["mode"].cpu(), g=lab["g"].cpu(), ood=lab["ood"],
                       domain={k: v.cpu().float() for k, v in lab["domain"].items()},
                       **{k: v for k, v in lab.items() if k not in ("mode", "g", "ood", "domain")})
    return c


def expand(c):
    """Storage dict -> RawTrial (float32/int64 tensors, Assembly and Schedule objects)."""
    m = c["meta"]
    asm = asm_from_dict(m["assembly"])
    sched = Schedule(m["char_length"], pids=tuple(m["pids"]))
    t = torch.arange(m["T"], dtype=torch.float32) * m["t_step"]
    B = c["counts"].shape[0]
    raw = {k: c[k].float() for k in HALF}
    raw["t"] = t
    raw["cmd"] = torch.tensor(sched.twist(t.numpy()), dtype=torch.float32)[None].expand(B, -1, -1).contiguous()
    raw["counts"] = c["counts"].long()
    raw["truth_pose"] = c["truth_pose"].float()
    labels = c["labels"]
    raw["meta"] = dict(assembly=asm, cpr=m["cpr"], schedule=sched, source=m["source"], trial_ids=m["trial_ids"],
                       labels=labels, n=m["n"], char_length=m["char_length"], dt=m["dt"], extra=m.get("extra", {}))
    raw["labels"] = labels
    return raw


def load_shard(path):
    return [expand(c) for c in torch.load(path, weights_only=False)]


def iter_shard(path, keep=None):
    """Lazily expand the groups of one shard; keep(compact_group) -> bool filters BEFORE the float32 expansion
    (integration note M9: expanding a whole ~800 MB shard at once can exceed free RAM)."""
    for c in torch.load(path, weights_only=False):
        if keep is None or keep(c):
            yield expand(c)


def iter_dataset(name_or_dir):
    d = pathlib.Path(name_or_dir)
    if not d.is_absolute() and not d.exists():
        d = OUT / name_or_dir
    for p in sorted(d.glob("shard_*.pt")):
        yield from load_shard(p)


# ----------------------------------------------------------------------------- generation
def physical_family(name):
    cells = [(i, j) for _, i, j, _ in A.PHYSICAL_TYPES[name]]
    return A.shape_family(cells)


def make_assembly(rng, params, kind, sizes, only_families=None, exclude_families=None):
    if kind in A.PHYSICAL_TYPES:
        a = A.physical(kind, params)
        a.meta["family"] = physical_family(kind)
        a.meta["type"] = kind
        return a
    fams = None
    if only_families or exclude_families:
        fams = set(only_families or A.all_families(sizes)) - set(exclude_families or ())
        fams = {f for f in fams if int(f.split("_")[0][1:]) in sizes}
        if not fams:
            raise ValueError("no families left")
    a = A.random_assembly(rng, params, sizes=sizes, families=fams)
    a.meta["type"] = "random"
    return a


OOD_KINDS = ("payload", "low_mu", "payload_lifter", "low_mu_patch", "low_mu_sheet")
OOD_LABEL = dict(payload="payload", low_mu="low_mu", payload_lifter="payload", low_mu_patch="low_mu",
                 low_mu_sheet="low_mu")
# A2 (v3) field-realisable OOD specs (see docs/coordination/A.md, section A2)
LIFTER_MASS = (2.0, 6.0)          # kg, rigid payload on a lifter deck
LIFTER_OFFSET = 0.12              # m, max |dx|, |dy| of the payload CG from the lifter module centre
PATCH_FACTOR = (0.2, 0.4)         # per-wheel mu multiplier under the low-friction sheet
PATCH_SIDES = ("left", "right", "front", "rear")


def lifter_modules(asm, params):
    """Module indices of the assembly that carry a lifter (params module.lifter_modules); all modules if none."""
    lift = set(params["module"].get("lifter_modules", []))
    idx = [k for k, m in enumerate(asm.modules) if m[0] in lift]
    return idx or list(range(len(asm.modules)))


def patch_wheels(asm, side):
    """bool (n,) wheels under a low-friction sheet laid along one side of the assembly (relative to the nominal CG)."""
    p = asm.wheel_pos - asm.cg()
    ax, sg = _SIDE_AX[side]
    return sg * p[:, ax] > 1e-6


_SIDE_AX = dict(left=(1, 1), right=(1, -1), front=(0, 1), rear=(0, -1))


def ood_args(rng, asm, dom, b, ood, params):
    """Per-trial OOD parameters.
    v2 kinds: payload (F6) 0.5-2.5 kg point mass on a random module deck; low_mu (F7) global mu in the low-mu sheet
    range (mu_factor = mu_target / mu, shape (b,)).
    v3 kinds (A2): payload_lifter = 2-6 kg on a lifter module, CG offset U(-0.12, 0.12) m in x and y from the module
    centre; low_mu_patch = sheet under one side (left/right/front/rear of the CG), per-wheel mu factor U(0.2, 0.4) on
    those wheels (mu_factor shape (b, n), 1 elsewhere, fixed for the whole run); low_mu_sheet = the same sheet but
    world-fixed (a floor half-plane whose edge passes through the nominal CG, so the wheels on that side start on it;
    each wheel's mu follows its contact position as the probes move the assembly). mu_factor then records the
    initial per-wheel factor only."""
    f = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
    if ood == "payload":
        k = rng.integers(len(asm.module_pos), size=b)
        xy = asm.module_pos[k] + rng.uniform(-0.3, 0.3, size=(b, 2)) * asm.side
        return dict(payload=dict(mass=f(rng.uniform(0.5, 2.5, b)), x=f(xy[:, 0]), y=f(xy[:, 1])))
    if ood == "low_mu":
        lo, hi = params["floor"]["low_mu_sheet"]
        target = torch.tensor(rng.uniform(lo, hi, b), dtype=torch.float32)
        return dict(mu_factor=target / dom["mu"].cpu())
    if ood == "payload_lifter":
        mods = lifter_modules(asm, params)
        k = np.asarray(mods)[rng.integers(len(mods), size=b)]
        off = rng.uniform(-LIFTER_OFFSET, LIFTER_OFFSET, size=(b, 2))
        xy = asm.module_pos[k] + off
        return dict(payload=dict(mass=f(rng.uniform(*LIFTER_MASS, b)), x=f(xy[:, 0]), y=f(xy[:, 1]),
                                 module=f(k), dx=f(off[:, 0]), dy=f(off[:, 1])))
    if ood == "low_mu_patch":
        side = rng.integers(len(PATCH_SIDES), size=b)
        masks = np.stack([patch_wheels(asm, s) for s in PATCH_SIDES])           # (4,n)
        fac = rng.uniform(*PATCH_FACTOR, size=b)
        mf = np.where(masks[side], fac[:, None], 1.0)
        return dict(mu_factor=f(mf), patch_side=f(side), patch_factor=f(fac))
    if ood == "low_mu_sheet":
        side = rng.integers(len(PATCH_SIDES), size=b)
        masks = np.stack([patch_wheels(asm, s_) for s_ in PATCH_SIDES])
        fac = rng.uniform(*PATCH_FACTOR, size=b)
        cg = asm.cg()
        ax = np.array([_SIDE_AX[PATCH_SIDES[k]] for k in side])        # (b,2): axis, sign
        nrm = np.zeros((b, 2))
        nrm[np.arange(b), ax[:, 0]] = ax[:, 1]
        c = (nrm * cg[None]).sum(1)
        return dict(sheet=dict(nx=f(nrm[:, 0]), ny=f(nrm[:, 1]), c=f(c), factor=f(fac)),
                    mu_factor=f(np.where(masks[side], fac[:, None], 1.0)), patch_side=f(side), patch_factor=f(fac))
    return {}


def _sim_extra(e):
    """ood_args output -> (payload dict for sim or None, mu_factor or None, sheet or None)."""
    pay = e.get("payload")
    if pay is not None:
        pay = {k: pay[k] for k in ("mass", "x", "y")}
    if "sheet" in e:
        return pay, None, e["sheet"]
    return pay, e.get("mu_factor"), None


def _domain_extra(e):
    """ood_args output -> flat evaluator-only domain entries (payload_* / mu_factor / patch_*)."""
    out = {}
    for k, v in e.items():
        if k in ("payload", "sheet"):
            out.update({k + "_" + kk: vv.float() for kk, vv in v.items()})
        else:
            out[k] = v.float()
    return out


REL_FLOOR = 0.05                 # D17: floor on the healthy baseline in relative ratios (healthy rho is ~0.6-1.0)
TRUTH_LATENCY_HALF = 0.03        # MR: sensor DR half-width of the AprilTag latency around the params value (as before)


def measured_sensor_dr(params):
    """MR: sensor DR overrides that follow the params file. sensors.SENSOR_DR hard-codes the AprilTag latency range
    (0.12, 0.18) around the old 0.15 s estimate; featurize aligns its target windows with params truth.apriltag_latency_s
    (MEASURED 0.078 s), so the simulated latency must be centred there too, else every label is shifted by ~70 ms.
    With the ESTIMATE_V1 params (0.15 s) this reproduces the old range exactly."""
    lat = float(params["truth"]["apriltag_latency_s"])
    return dict(truth_latency=(max(0.0, lat - TRUTH_LATENCY_HALF), lat + TRUTH_LATENCY_HALF))


def generate(name, n_trials, mix, seed=0, sizes=(1, 2, 3, 4, 5, 6), trials_per_asm=64, phys_trials_per_asm=256,
             wheel_budget=16384, ood="none", ood_faults=False, only_families=None, exclude_families=None,
             shard_trials=2048, engine="compile", device="cuda", out_root=OUT, log=print, fault_kw=None,
             dr=None, sensor_dr=None, ood_twin=None, healthy_baseline=False):
    """ood: "none" or one of OOD_KINDS. ood_twin (default: on for OOD sets) also simulates every OOD trial without its
    OOD condition (same domain, faults and sim seed) and stores evaluator-only labels rho_twist_noood (B,14),
    ood_dy (B,14) = rho_twist - rho_twist_noood and ood_consequential (B,) = max_k |ood_dy| > 0.1."""
    out = pathlib.Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    params = A.load_params(PARAMS)
    sensor_dr = {**measured_sensor_dr(params), **(sensor_dr or {})}
    rng = np.random.default_rng(seed)
    kinds, probs = zip(*mix.items())
    probs = np.asarray(probs, float) / sum(probs)
    torch._dynamo.config.cache_size_limit = 64
    if ood not in ("none",) + OOD_KINDS:
        raise ValueError(f"unknown ood kind {ood}")
    ood_twin = (ood != "none") if ood_twin is None else bool(ood_twin and ood != "none")
    # D17 healthy baseline: every trial is also simulated with all wheels healthy (same domain theta, same sim seed, no
    # OOD condition) -> evaluator-only labels rho_healthy (B,14) (noise-free healthy-twin capability ratios of the same
    # assembly in the same domain) and rho_rel_twist = rho_twist / rho_healthy. Relative capability y / rho_healthy is
    # the sim analogue of the field's ratio / healthy-build median (docs/SPINE.md D17).
    manifest = dict(name=name, created=time.strftime("%Y-%m-%dT%H:%M:%S"), seed=seed, n_trials_requested=n_trials,
                    params_path=str(PARAMS), params_sha256=sha256_file(PARAMS), code_sha256=code_hash(),
                    args=dict(mix=dict(mix), sizes=list(sizes), trials_per_asm=trials_per_asm,
                              phys_trials_per_asm=phys_trials_per_asm, wheel_budget=wheel_budget, ood=ood,
                              ood_faults=ood_faults, ood_twin=ood_twin, only_families=only_families, exclude_families=exclude_families,
                              engine=engine, dt=0.005, fault_kw=fault_kw or {}, dr=dr or {},
                              sensor_dr={k: (list(v) if isinstance(v, tuple) else v) for k, v in sensor_dr.items()},
                              healthy_baseline=bool(healthy_baseline)),
                    torch=torch.__version__, shards=[], counts=dict(trials=0, groups=0, by_type={}, by_family={},
                                                                     by_n={}, by_mode={}))
    if ood != "none":
        manifest["counts"]["ood_consequential"] = 0
        manifest["counts"]["ood_consequential_by_type"] = {}
    pending, pending_trials, shard_idx, done = [], 0, 0, 0
    t0 = time.time()
    gseed = 0

    def flush():
        nonlocal pending, pending_trials, shard_idx
        if not pending:
            return
        p = out / f"shard_{shard_idx:05d}.pt"
        torch.save(pending, p)
        manifest["shards"].append(dict(file=p.name, groups=len(pending), trials=pending_trials,
                                       sha256=sha256_file(p)))
        shard_idx += 1
        pending, pending_trials = [], 0
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    tpa = lambda k: phys_trials_per_asm if k in A.PHYSICAL_TYPES else trials_per_asm
    draw_p = np.array([p / tpa(k) for k, p in zip(kinds, probs)])   # mix weights are in TRIALS, not assemblies
    draw_p /= draw_p.sum()
    buckets = {}                                                       # n -> list of (asm, b)
    planned = 0

    def next_batch():
        """Sample assemblies into per-n buckets; return the first bucket that fills its batch (B*n <= budget), or,
        once the requested count is planned, the leftovers."""
        nonlocal planned
        while planned < n_trials:
            k = kinds[rng.choice(len(kinds), p=draw_p)]
            a = make_assembly(rng, params, k, sizes, only_families, exclude_families)
            b = min(tpa(k), n_trials - planned)
            buckets.setdefault(a.n, []).append((a, b))
            planned += b
            B_max = max(64, (wheel_budget // a.n) // 64 * 64)
            over = sum(x[1] for x in buckets[a.n]) - B_max
            if over >= 0:                                              # exact B_max keeps compiled shapes stable
                grp = buckets.pop(a.n)
                if over > 0:
                    grp[-1] = (a, b - over)
                    buckets[a.n] = [(a, over)]
                return grp
        for n_ in sorted(buckets):
            return buckets.pop(n_)
        return None

    while True:
        grp = next_batch()
        if grp is None:
            break
        asms, bs = [x[0] for x in grp], [x[1] for x in grp]
        n = asms[0].n
        doms, modes, gs, extra = [], [], [], []
        for a, b in zip(asms, bs):
            dom = sim.sample_domain(b, rng, dr)
            if ood != "none" and not ood_faults:
                mode, g = torch.zeros(b, n, dtype=torch.long), torch.ones(b, n)
            else:
                mode, g = sim.sample_faults(b, n, rng, **(fault_kw or {}))
            doms.append(dom)
            modes.append(mode)
            gs.append(g)
            extra.append(ood_args(rng, a, dom, b, ood, params))
        sx = [_sim_extra(e) for e in extra]
        sseed = seed * 100003 + gseed
        twins = None
        if ood_twin:                                                   # paired no-OOD run (same dom/faults/seed)
            twins = sim.simulate_multi(asms, params, [dict(d) for d in doms], modes, gs, device=device, engine=engine,
                                       seed=sseed)
            twins = [sim.targets_from_twist(Rt)[0].cpu() for Rt in twins]
        healthy = None
        if healthy_baseline:                                           # D17: healthy twin (same dom / seed, no OOD)
            healthy = sim.simulate_multi(asms, params, [dict(d) for d in doms],
                                         [torch.zeros_like(m) for m in modes], [torch.ones_like(x) for x in gs],
                                         device=device, engine=engine, seed=sseed)
            healthy = [sim.targets_from_twist(Rh)[0].cpu() for Rh in healthy]
        Rs = sim.simulate_multi(asms, params, doms, modes, gs, device=device, engine=engine, seed=sseed,
                                payloads=[x[0] for x in sx] if any(x[0] is not None for x in sx) else None,
                                mu_factors=[x[1] for x in sx] if any(x[1] is not None for x in sx) else None,
                                sheets=[x[2] for x in sx] if any(x[2] is not None for x in sx) else None)
        for i_g, (R, a, dom, e) in enumerate(zip(Rs, asms, doms, extra)):
            b = R["omega"].shape[0]
            ids = [f"{name}_g{gseed}_{i}" for i in range(b)]
            raw = sensors.to_raw(R, params, dom, rng, sensor_dr=sensor_dr, trial_ids=ids,
                                 ood=OOD_LABEL.get(ood, ood))
            rho, drift = sim.targets_from_twist(R)
            lab = raw["meta"]["labels"]
            lab["rho_twist"] = rho.cpu()                               # noise-free reference target (evaluator-only)
            lab["drift_twist"] = drift.cpu()
            if healthy is not None:
                lab["rho_healthy"] = healthy[i_g]
                lab["rho_rel_twist"] = rho.cpu() / healthy[i_g].clamp(min=REL_FLOOR)
            lab["family"] = a.meta.get("family")
            lab["asm_type"] = a.meta.get("type")
            if ood != "none":
                lab["ood_kind"] = ood
            if e:
                lab["domain"].update(_domain_extra(e))
            if twins is not None:
                dy = rho.cpu() - twins[i_g]
                lab["rho_twist_noood"] = twins[i_g]
                lab["ood_dy"] = dy
                lab["ood_consequential"] = dy.abs().amax(1) > CONSEQUENTIAL
                nc = int(lab["ood_consequential"].sum())
                manifest["counts"]["ood_consequential"] += nc
                bt = manifest["counts"]["ood_consequential_by_type"]
                bt[a.meta.get("type")] = bt.get(a.meta.get("type"), 0) + nc
            pending.append(compact(raw))
            pending_trials += b
            done += b
            gseed += 1
            cnt = manifest["counts"]
            cnt["trials"] += b
            cnt["groups"] += 1
            for key, val in (("by_type", a.meta.get("type")), ("by_family", a.meta.get("family")), ("by_n", str(n))):
                cnt[key][val] = cnt[key].get(val, 0) + b
            mm = R["mode"].cpu()
            nf = (mm > 0).sum(1)
            for b_i in range(b):
                key = "healthy" if nf[b_i] == 0 else (
                    sim.MODE_NAMES[int(mm[b_i].max())] if nf[b_i] == 1 else "double")
                cnt["by_mode"][key] = cnt["by_mode"].get(key, 0) + 1
            del raw
        del Rs
        if pending_trials >= shard_trials:
            flush()
        el = time.time() - t0
        log(f"[{name}] {done}/{n_trials} trials, n={n}, groups={len(asms)}, {el:.0f}s, {done / el * 60:.0f}/min")
    flush()
    manifest["elapsed_s"] = time.time() - t0
    manifest["complete"] = True
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ----------------------------------------------------------------------------- OOD consequence study (A2)
SIG_GROUPS = dict(P1=("P1",), P5=("P5",), P2=("P2",), P34=("P3", "P4"), P6=("P6x", "P6y"))


def _seg_masks(R):
    t = R["t"].cpu().numpy()
    sched = R["meta"]["schedule"]
    out = {}
    for gname, pids in SIG_GROUPS.items():
        m = np.zeros(len(t), bool)
        for s_ in sched.segments:
            if s_.pid in pids:
                m |= (t >= s_.t0) & (t < s_.t1)
        out[gname] = torch.tensor(m)
    return out


def _signals(R0, R1):
    """Paired per-trial RMS differences (OOD minus twin) of robot-visible signals per primitive group, and the
    in-model spread of the same signal across the twin batch (domain randomisation + faults) for scale."""
    masks = _seg_masks(R0)
    keys = dict(omega=("omega", None), duty=("duty", None), gyro=("twist", 2), acc=("accel", None))
    out = {}
    for gname, m in masks.items():
        for kname, (k, ch) in keys.items():
            a = R0[k][:, m].float()
            b_ = R1[k][:, m].float()
            if ch is not None:
                a, b_ = a[..., ch], b_[..., ch]
            red = tuple(range(1, a.dim()))
            d = ((b_ - a) ** 2).mean(red).sqrt()
            spread = ((a - a.mean(0, keepdim=True)) ** 2).mean(red).sqrt()
            out[f"{gname}_{kname}"] = d.cpu().numpy()
            out[f"{gname}_{kname}_spread"] = spread.cpu().numpy()
    return out


def ood_study(B=512, seed=0, device="cuda", engine="graph", conds=OOD_KINDS, types=tuple(A.PHYSICAL_TYPES),
              out_path=OUT / "ood_study_v3", log=print):
    """Paired-seed consequence study: every trial is simulated with and without its OOD condition (same domain,
    faults and sim seed). Fault arms: healthy wheels, and the training fault prior. Writes <out_path>.json
    (summary) and <out_path>_trials.npz (per-trial dy, OOD params and signal deltas)."""
    params = A.load_params(PARAMS)
    rng = np.random.default_rng(seed)
    from .probes import TARGET_KEYS
    rows, trials = [], {}
    for cond in conds:
        for arm in ("healthy", "faults"):
            for typ in types:
                asm = A.physical(typ, params)
                n = asm.n
                dom = sim.sample_domain(B, rng)
                if arm == "healthy":
                    mode, g = torch.zeros(B, n, dtype=torch.long), torch.ones(B, n)
                else:
                    mode, g = sim.sample_faults(B, n, rng)
                e = ood_args(rng, asm, dom, B, cond, params)
                pay, mf, sh = _sim_extra(e)
                s_ = int(rng.integers(2 ** 31))
                R0 = sim.simulate(asm, params, dict(dom), mode, g, device=device, engine=engine, seed=s_)
                R1 = sim.simulate(asm, params, dict(dom), mode, g, device=device, engine=engine, seed=s_,
                                  payload=pay, mu_factor=mf, sheet=sh)
                y0 = sim.targets_from_twist(R0)[0].cpu().numpy()
                y1 = sim.targets_from_twist(R1)[0].cpu().numpy()
                dy = y1 - y0
                sig = _signals(R0, R1)
                key = f"{cond}|{arm}|{typ}"
                trials[key + "|dy"] = dy.astype(np.float32)
                trials[key + "|y0"] = y0.astype(np.float32)
                trials[key + "|mode"] = mode.numpy()
                for k, v in _domain_extra(e).items():
                    trials[key + "|" + k] = v.numpy()
                for k, v in sig.items():
                    trials[key + "|sig_" + k] = v
                adq = np.abs(dy)
                row = dict(cond=cond, arm=arm, type=typ, B=B, max_abs_dy=float(adq.max()),
                           p90_abs_dy=float(np.quantile(adq.max(1), 0.9)), mean_abs_dy=float(adq.mean()),
                           frac_consequential=float((adq.max(1) > CONSEQUENTIAL).mean()),
                           worst_key=TARGET_KEYS[int(adq.max(0).argmax())],
                           per_key_p90={k: float(np.quantile(adq[:, i], 0.9)) for i, k in enumerate(TARGET_KEYS)},
                           per_key_max={k: float(adq[:, i].max()) for i, k in enumerate(TARGET_KEYS)})
                for gname in SIG_GROUPS:
                    for kname in ("omega", "duty", "gyro", "acc"):
                        d = sig[f"{gname}_{kname}"]
                        sp = sig[f"{gname}_{kname}_spread"]
                        row[f"sig_{gname}_{kname}"] = float(np.median(d))
                        row[f"sigz_{gname}_{kname}"] = float(np.median(d) / max(np.median(sp), 1e-9))
                rows.append(row)
                log(f"{key}: max|dy|={row['max_abs_dy']:.3f} p90={row['p90_abs_dy']:.3f} "
                    f"cons={row['frac_consequential']:.2f} worst={row['worst_key']}")
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(params_sha256=sha256_file(PARAMS), code_sha256=code_hash(), seed=seed, B=B,
                   consequential_threshold=CONSEQUENTIAL, rows=rows,
                   note="dy = rho_twist(with OOD) - rho_twist(paired twin without OOD); sig_* = median paired RMS "
                        "difference over the primitive group's legs (omega rad/s, duty, gyro rad/s, acc m/s^2); "
                        "sigz_* = that / median in-model spread of the same signal across the twin batch")
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(str(out_path) + "_trials.npz", **trials)
    return summary


def _parse_mix(s):
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        if k == "physical":
            for t in A.PHYSICAL_TYPES:
                out[t] = out.get(t, 0) + float(v) / len(A.PHYSICAL_TYPES)
        else:
            out[k] = out.get(k, 0) + float(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None)
    ap.add_argument("--ood-study", action="store_true", help="run the paired OOD consequence study (A2) instead")
    ap.add_argument("--study-B", type=int, default=512)
    ap.add_argument("--n-trials", type=int, default=20000)
    ap.add_argument("--mix", default="physical=0.3,random=0.7",
                    help="comma list kind=weight; kinds: physical (all four), single, pair, L3, square4, random")
    ap.add_argument("--sizes", default="1,2,3,4,5,6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials-per-asm", type=int, default=64)
    ap.add_argument("--phys-trials-per-asm", type=int, default=256)
    ap.add_argument("--wheel-budget", type=int, default=16384, help="B*n per batched run (GPU memory)")
    ap.add_argument("--ood", default="none", choices=("none",) + OOD_KINDS,
                    help="v2: payload, low_mu; v3 (A2): payload_lifter, low_mu_patch, low_mu_sheet")
    ap.add_argument("--no-ood-twin", action="store_true", help="skip the paired no-OOD run (no ood_dy labels)")
    ap.add_argument("--ood-faults", action="store_true", help="also inject prior wheel faults in OOD sets")
    ap.add_argument("--only-families", default=None)
    ap.add_argument("--exclude-families", default=None)
    ap.add_argument("--shard-trials", type=int, default=2048)
    ap.add_argument("--engine", default="compile")
    ap.add_argument("--dr", default=None, help="domain randomisation: 'wide' (sim.DR_WIDE), 'measured' "
                    "(sim.DR_MEASURED, centred on params MEASURED_V1) or a JSON file of "
                    "overrides, e.g. the output of sysid.dr_from_sysid")
    ap.add_argument("--healthy-baseline", action="store_true",
                    help="D17: also simulate a healthy twin per trial -> labels rho_healthy, rho_rel_twist")
    ap.add_argument("--out-root", default=str(OUT), help="dataset root (default ~/.cache/capxfer/A)")
    ap.add_argument("--fault-probs", default=None, help="mode_probs for weak,coast,brake,roller, e.g. 0.3,0.15,0.15,0.4")
    ap.add_argument("--gpu-frac", type=float, default=0.5, help="per-process GPU memory fraction (WS-A <= 8 GB)")
    a = ap.parse_args(argv)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    split = lambda s: s.split(",") if s else None
    if a.ood_study:
        return ood_study(B=a.study_B, seed=a.seed, log=lambda s: print(s, flush=True))
    if not a.name:
        ap.error("--name is required")
    dr = None
    if a.dr:
        dr = dict(sim.DR_PRESETS[a.dr]) if a.dr in sim.DR_PRESETS else json.loads(pathlib.Path(a.dr).read_text())
        dr = {k: (tuple(v) if isinstance(v, list) else v) for k, v in dr.items()}
    fault_kw = dict(mode_probs=tuple(float(x) for x in a.fault_probs.split(","))) if a.fault_probs else None
    generate(a.name, a.n_trials, _parse_mix(a.mix), dr=dr, fault_kw=fault_kw, seed=a.seed, sizes=tuple(int(x) for x in a.sizes.split(",")),
             trials_per_asm=a.trials_per_asm, phys_trials_per_asm=a.phys_trials_per_asm, wheel_budget=a.wheel_budget,
             ood=a.ood, ood_faults=a.ood_faults, ood_twin=not a.no_ood_twin, only_families=split(a.only_families),
             exclude_families=split(a.exclude_families), shard_trials=a.shard_trials, engine=a.engine,
             out_root=pathlib.Path(a.out_root).expanduser(), healthy_baseline=a.healthy_baseline,
             log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()

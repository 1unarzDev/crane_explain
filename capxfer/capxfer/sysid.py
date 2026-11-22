"""System identification hook: fit simulator domain parameters to healthy single-module RawTrials (CMA-ES).

    fit = fit_params(real_raw_list, params, init_dom=None, keys=FIT_KEYS, iters=40, popsize=24)
    dr  = dr_around(fit, rel_width=0.15)         # narrowed DR dict for sim.sample_domain (retraining)

Only HEALTHY SINGLE-MODULE trials may be used (plan: fault and assembly transfer stay zero-shot); fit_params
refuses trials whose labels say otherwise when labels exist.

Loss (trajectory, per real trial, averaged): normalised squared error between the deterministic simulation of the
same probe schedule and the observed streams:
* wheel speed from encoder count differences (50 ms window), scale 0.5 rad/s;
* duty, scale 0.05;
* gyro_z minus its at-rest mean (bias), scale 0.05 rad/s;
* truth_pose (AprilTag; calibration use only) vs the simulated pose delayed by the nominal tag latency, scale 0.01 m
  / 0.02 rad.
The commanded schedule is identical for every real trial of one module, so each CMA-ES candidate needs ONE simulated
trial; the population is one batched simulator call.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import assembly as A
from . import sim
from .probes import PIDS, Schedule

# key -> (lo, hi) search bounds (absolute for mu/s0/c_roller/crr/latency, scales otherwise)
FIT_BOUNDS = dict(mu=(0.3, 1.2), s0=(0.005, 0.08), c_roller=(0.1, 8.0), crr=(0.001, 0.05), latency=(0.0, 0.15),
                  stall_scale=(0.6, 1.4), noload_scale=(0.75, 1.25), izz_scale=(0.6, 1.6), kp_scale=(0.4, 2.0),
                  ki_scale=(0.4, 2.0), radius_scale=(0.92, 1.08), v_batt=(10.5, 12.8))
# A2: radius_scale and v_batt were FIXED (1.0 / 11.1 V) before; both move rho directly (radius 1:1), so a fit that
# cannot set them compensates with wrong friction/roller values (dry-run diagnosis, docs/coordination/A.md A2).
FIT_KEYS = tuple(FIT_BOUNDS)
LEGACY_KEYS = ("mu", "s0", "c_roller", "crr", "latency", "stall_scale", "noload_scale", "izz_scale", "kp_scale",
               "ki_scale")
# Comparative gate (A2, for the prereg): see gate().
GATE_MARGIN = 0.002      # fitted must beat defaults by >= this (median |d rho| over held-out trials x keys)
GATE_CAP = 0.03          # and its own held-out error must be <= this
RHO_FLOOR = 0.004        # rho term of the fit loss: per-key scale sqrt(RHO_FLOOR^2 + SE_k^2), SE_k = standard error of
RHO_WEIGHT = 0.25        # the mean real rho over the fit trials (tag noise ~0.013 per trial); weight vs trajectory
FIXED = dict(tau_bd=0.6, mass_scale=1.0, radius_scale=1.0, cg_dx=0.0, cg_dy=0.0)
SCALES = dict(omega=0.5, duty=0.05, gyro=0.05, pos=0.01, yaw=0.02)


def nominal_values(params=None):
    """Default ('nominal') domain values: the middle of the training DR (sim.DR_DEFAULT) for every fit key, the
    nominal battery voltage. This is the comparison point of the gate."""
    v = {k: 0.5 * (sim.DR_DEFAULT[k][0] + sim.DR_DEFAULT[k][1]) for k in FIT_KEYS if k in sim.DR_DEFAULT}
    if params is not None:
        v["v_batt"] = float(params["battery"]["nominal_v"])
    v["radius_scale"] = 1.0
    return v


def nominal_domain(B, params, vals=None, device="cpu", fixed=None):
    """Public deterministic domain dict for sim.simulate (B identical trials): nominal values overridden by `vals`
    (e.g. fit["best"]). Replaces the pipeline's use of the private _dom_from (integration note M7)."""
    return _dom_from({**nominal_values(params), **(vals or {})}, B, params, device, fixed)


def _dom_from(vals, B, params, device, fixed=None):
    fx = {"v_batt": float(params["battery"]["nominal_v"]), **FIXED, **(fixed or {})}
    d = {k: torch.full((B,), float(v), dtype=torch.float32, device=device) for k, v in fx.items()}
    for k, v in vals.items():
        d[k] = torch.as_tensor(v, dtype=torch.float32, device=device).reshape(-1).expand(B).clone()
    for k in sim.DR_DEFAULT:
        if k not in d and k not in ("cg_off", "wheel_yaw_err"):
            lo, hi = sim.DR_DEFAULT[k]
            d[k] = torch.full((B,), 0.5 * (lo + hi), device=device)
    d["wheel_yaw_err_std"] = 0.0
    return d


def _wheel_speed(angle, dt, win=5):
    """Speed from angle differences over `win` samples (centred), (B,T,n)."""
    d = torch.zeros_like(angle)
    d[:, win:] = (angle[:, win:] - angle[:, :-win]) / (win * dt)
    return d


def observables_real(raws, rest_s=0.4, device="cpu"):
    """Stack real RawTrials (same assembly/schedule) into the loss observables."""
    cat = lambda k: torch.cat([r[k].float() for r in raws], 0).to(device)
    r0 = raws[0]
    t = r0["t"].to(device)
    dt = float(t[1] - t[0])
    cpr = float(r0["meta"]["cpr"])
    ang = torch.cat([r["counts"].double() for r in raws], 0).to(device) * (2 * math.pi / cpr)
    om = _wheel_speed(ang.float(), dt)
    gz = cat("gyro_z")
    rest = t < rest_s
    gz = gz - gz[:, rest].mean(1, keepdim=True)
    tp = cat("truth_pose")
    # tag world frame -> frame of the start pose (first 0.1 s average)
    p0 = tp[:, :10].mean(1)
    c, s_ = torch.cos(p0[:, 2])[:, None], torch.sin(p0[:, 2])[:, None]
    dx, dy = tp[..., 0] - p0[:, None, 0], tp[..., 1] - p0[:, None, 1]
    tp = torch.stack([c * dx + s_ * dy, -s_ * dx + c * dy, tp[..., 2] - p0[:, None, 2]], -1)
    rho = torch.cat([_rho_real(r) for r in raws], 0).to(device)
    return dict(t=t, dt=dt, omega=om, duty=cat("duty"), gyro=gz, pose=tp, rho=rho)


def _rho_real(raw, truth_latency=None):
    """Evaluator capability ratios of real trials (WS-B featurize.targets on truth_pose), (B,14)."""
    from . import featurize as FZ
    tl = truth_latency if truth_latency is not None else float(raw["meta"].get("truth_latency_s", 0.15))
    return FZ.targets(raw, raw["meta"]["assembly"], tl)[0].float()


def rho_sim(R):
    """Same estimator (featurize.targets) applied to the simulated pose (no tag latency), (B,14). Using one estimator
    for both sides cancels its small bias against the twist-mean ratio (~+0.005-0.01)."""
    from . import featurize as FZ
    raw = {"t": R["t"], "truth_pose": R["pose"].float(), "meta": {"schedule": R["meta"]["schedule"]}}
    return FZ.targets(raw, R["meta"]["assembly"], 0.0)[0].float()


def observables_sim(R, truth_latency=0.15, with_rho=True):
    dt = float(R["t"][1] - R["t"][0])
    om = _wheel_speed(R["wheel_angle"].float(), dt)
    lag = int(round(truth_latency / dt))
    pose = R["pose"].float()
    pd = torch.cat([pose[:, :1].expand(-1, lag, -1), pose[:, :-lag]], 1) if lag > 0 else pose
    out = dict(omega=om, duty=R["duty"].float(), gyro=R["twist"][..., 2].float(), pose=pd)
    if with_rho:
        out["rho"] = rho_sim(R).to(om.device)
    return out


def trajectory_loss(sim_obs, real_obs, weights=None):
    """(P,) loss of P simulated candidates against N real trials (mean over trials, time, channels)."""
    w = {**dict(omega=1.0, duty=1.0, gyro=1.0, pose=1.0, rho=RHO_WEIGHT), **(weights or {})}
    T = min(sim_obs["omega"].shape[1], real_obs["omega"].shape[1])
    tot = 0.0
    for k in ("omega", "duty", "gyro"):
        s = sim_obs[k][:, None, :T]
        r = real_obs[k][None, :, :T]
        e = ((s - r) / SCALES[k]) ** 2
        tot = tot + w[k] * e.reshape(e.shape[0], e.shape[1], -1).mean(-1).mean(1)
    sp, rp = sim_obs["pose"][:, None, :T], real_obs["pose"][None, :, :T]
    dyaw = torch.atan2(torch.sin(sp[..., 2] - rp[..., 2]), torch.cos(sp[..., 2] - rp[..., 2]))
    ep = ((sp[..., :2] - rp[..., :2]) / SCALES["pos"]) ** 2
    ey = (dyaw / SCALES["yaw"]) ** 2
    tot = tot + w["pose"] * (ep.mean((-1, -2)) + ey.mean(-1)).mean(1) * 0.5
    if w["rho"] > 0 and "rho" in sim_obs and "rho" in real_obs:      # A2: capability ratios directly (mean over trials)
        yr = real_obs["rho"]
        rr = torch.nanmean(yr, 0)[None]                                # (1,14)
        nk = torch.isfinite(yr).sum(0).clamp(min=1)
        var = torch.nanmean((yr - rr) ** 2, 0) * nk / (nk - 1).clamp(min=1)
        se2 = torch.nan_to_num(var / nk, nan=0.0)
        er = (sim_obs["rho"] - rr) ** 2 / (RHO_FLOOR ** 2 + se2)[None]
        ok = torch.isfinite(er)
        tot = tot + w["rho"] * torch.where(ok, er, torch.zeros_like(er)).sum(-1) / ok.sum(-1).clamp(min=1)
    return tot


def rho_error(vals, raws, params, device=None, fixed=None, B=1):
    """Held-out capability error of a parameter point: median over trials x keys of |rho_sim(vals) - rho_real|.
    Also returns per-key medians and the paired per-trial medians."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    asm = raws[0]["meta"]["assembly"]
    dom = nominal_domain(B, params, vals, device, fixed)
    R = sim.simulate(asm, params, dom, torch.zeros(B, asm.n, dtype=torch.long), torch.ones(B, asm.n), device=device)
    rs = rho_sim(R).mean(0).cpu().numpy()
    yr = torch.cat([_rho_real(r) for r in raws], 0).cpu().numpy()
    err = np.abs(yr - rs[None])
    return dict(median=float(np.nanmedian(err)), per_key=np.nanmedian(err, 0).tolist(),
                per_trial=np.nanmedian(err, 1).tolist())


def gate(fit, heldout_raws, params, margin=GATE_MARGIN, cap=GATE_CAP, device=None, fixed=None):
    """Comparative sys-ID gate (A2; rule text in docs/coordination/A.md for the prereg).
    e_fit / e_nom = median |rho_sim - rho_real| over held-out healthy single-module trials x 14 keys, for the fitted
    point and for the nominal (default) point. PASS iff e_fit <= e_nom - margin AND e_fit <= cap. The parameter point
    used downstream is the fitted one on PASS, otherwise the defaults (`use`)."""
    ef = rho_error(fit["best"], heldout_raws, params, device, fixed)
    en = rho_error({}, heldout_raws, params, device, fixed)
    ok = bool(ef["median"] <= en["median"] - margin and ef["median"] <= cap)
    return dict(e_fit=ef["median"], e_nom=en["median"], e_fit_per_key=ef["per_key"], e_nom_per_key=en["per_key"],
                margin=margin, cap=cap, gate_pass=ok, use="fitted" if ok else "nominal",
                n_heldout=len(ef["per_trial"]))


def _check_healthy_single(raws):
    for r in raws:
        asm = r["meta"]["assembly"]
        assert asm.n == 4 and len(asm.modules) == 1, "sys-ID uses single-module trials only"
        lab = r["meta"].get("labels")
        if lab is not None and "mode" in lab:
            assert (torch.as_tensor(lab["mode"]) == 0).all(), "sys-ID uses healthy trials only"
        if lab is not None and lab.get("ood", "none") != "none":
            raise AssertionError("sys-ID uses in-model trials only")


def simulate_candidates(X, keys, asm, params, device, fixed=None, engine="auto", pids=None):
    """X: (P, len(keys)) parameter values -> sim result for P candidates (healthy)."""
    P = X.shape[0]
    vals = {k: torch.tensor(X[:, i], dtype=torch.float32) for i, k in enumerate(keys)}
    dom = _dom_from(vals, P, params, device, fixed)
    mode = torch.zeros(P, asm.n, dtype=torch.long)
    sched = Schedule(asm.char_length(), pids=tuple(pids)) if pids else None
    return sim.simulate(asm, params, dom, mode, torch.ones(P, asm.n), device=device, engine=engine, schedule=sched)


def subset_trials(raw, idx):
    """RawTrial restricted to trial indices idx (batch-first tensors, trial_ids, labels)."""
    idx = list(idx)
    B = raw["counts"].shape[0]
    it = torch.as_tensor(idx, dtype=torch.long)
    out = {k: (v[it.to(v.device)] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B and k != "t" else v)
           for k, v in raw.items() if k not in ("meta", "labels")}
    meta = dict(raw["meta"])
    meta["trial_ids"] = [meta["trial_ids"][i] for i in idx] if meta.get("trial_ids") is not None else None
    lab = meta.get("labels")
    if lab is not None:
        lab = {k: (v[it] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B else v) for k, v in lab.items()}
        meta["labels"] = lab
        out["labels"] = lab
    out["meta"] = meta
    return out


def _cma_run(real, keys, asm, params, device, fixed, x0, iters, popsize, sigma0, seed, tl, weights, pids, log, tag):
    import cma
    lo = np.array([FIT_BOUNDS[k][0] for k in keys])
    hi = np.array([FIT_BOUNDS[k][1] for k in keys])
    z0 = np.clip((np.asarray(x0) - lo) / (hi - lo), 0.02, 0.98)
    es = cma.CMAEvolutionStrategy(z0.tolist(), sigma0, dict(bounds=[0, 1], popsize=popsize, seed=seed + 1,
                                                            verbose=-9))
    hist, best = [], (np.inf, lo + z0 * (hi - lo))
    for it in range(iters):
        Z = np.asarray(es.ask())
        X = lo + Z * (hi - lo)
        R = simulate_candidates(X, keys, asm, params, device, fixed, pids=pids)
        L = trajectory_loss(observables_sim(R, tl), real, weights).cpu().numpy()
        L = np.where(np.isfinite(L), L, 1e9)
        es.tell(Z.tolist(), L.tolist())
        i = int(np.argmin(L))
        if L[i] < best[0]:
            best = (float(L[i]), X[i].copy())
        hist.append(float(L[i]))
        if log:
            log(f"sysid {tag} it {it}: best {best[0]:.4f} " + " ".join(f"{k}={v:.3g}" for k, v in zip(keys, best[1])))
    std = np.asarray(es.result.stds) * (hi - lo)                       # CMA search std in parameter units
    return dict(best={k: float(v) for k, v in zip(keys, best[1])}, loss=best[0], history=hist,
                std={k: float(v) for k, v in zip(keys, std)})


def fit_params(real_raw_list, params, init_dom=None, keys=FIT_KEYS, iters=40, popsize=24, sigma0=0.25, seed=0,
               device=None, fixed=None, truth_latency=None, log=None, weights=None, pids=None, restarts=3,
               select="val"):
    """CMA-ES fit of domain parameters to healthy single-module RawTrials.

    Loss: trajectory loss (encoder speed, duty, gyro, tag pose) + RHO_WEIGHT x capability-ratio term (mean over the
    fit trials of featurize.targets on the tag pose vs the same estimator on the simulated pose, per-key scale
    sqrt(RHO_FLOOR^2 + standard error^2)). A2 changes (dry-run diagnosis in A.md):
    * radius_scale and v_batt are fit keys (were fixed at 1.0 / 11.1 V);
    * `restarts` CMA-ES runs (restart 0 from the nominal point, the others from random points in FIT_BOUNDS);
    * select="val" (needs >= 4 trials): the restarts fit on every other trial and the winner is chosen among the
      restart optima AND the nominal point by held-out capability error (rho_error) on the remaining trials, so the
      returned point is never worse than defaults on that internal split. fit["use"] says which won.
      select="loss": lowest training loss over restarts on all trials (no defaults fallback; legacy behaviour).
    init_dom: optional dict key -> start value of restart 0 (default: nominal_values()).
    pids: optional PREFIX of probes.PIDS to simulate only the start of the schedule (faster).
    Returns dict(best, loss, history, keys, evals, std (search std per key), restarts=[...], use, val)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _check_healthy_single(real_raw_list)
    if pids:
        assert tuple(pids) == PIDS[:len(pids)], "pids must be a prefix of the probe library order"
    keys = tuple(keys)
    asm = real_raw_list[0]["meta"]["assembly"]
    tl = float(params["truth"]["apriltag_latency_s"]) if truth_latency is None else truth_latency
    n_tr = sum(int(r["counts"].shape[0]) for r in real_raw_list)
    use_val = select == "val" and n_tr >= 4
    if use_val:                                                        # alternate trials: fit / internal validation
        fit_l, val_l = [], []
        for r in real_raw_list:
            b = int(r["counts"].shape[0])
            ia, ib = list(range(0, b, 2)), list(range(1, b, 2))
            if ia:
                fit_l.append(subset_trials(r, ia))
            if ib:
                val_l.append(subset_trials(r, ib))
    else:
        fit_l, val_l = real_raw_list, []
    real = observables_real(fit_l, device=device)
    rng = np.random.default_rng(seed)
    nom = {**nominal_values(params), **(fixed or {}), **(init_dom or {})}
    runs = []
    for k in range(max(1, restarts)):
        if k == 0:
            x0 = [nom.get(kk, 0.5 * sum(FIT_BOUNDS[kk])) for kk in keys]
        else:
            x0 = [rng.uniform(*FIT_BOUNDS[kk]) for kk in keys]
        runs.append(_cma_run(real, keys, asm, params, device, fixed, x0, iters, popsize, sigma0, seed + 17 * k, tl,
                             weights, pids, log, f"r{k}"))
    val = None
    if use_val:
        cands = [("fitted_r%d" % k, r_["best"]) for k, r_ in enumerate(runs)] + [("nominal", {})]
        val = {nm: rho_error(v, val_l, params, device, fixed)["median"] for nm, v in cands}
        win = min(val, key=val.get)
        if win == "nominal":
            chosen = dict(best={kk: float(nom.get(kk, 0.5 * sum(FIT_BOUNDS[kk]))) for kk in keys}, loss=float("nan"),
                          history=runs[0]["history"], std=runs[0]["std"])
        else:
            chosen = runs[int(win.split("_r")[1])]
        use = "nominal" if win == "nominal" else "fitted"
    else:
        chosen = min(runs, key=lambda r_: r_["loss"])
        use = "fitted"
    # uncertainty: max of the chosen run's CMA search std and the spread of restart optima that are within 2x its loss
    good = [r_ for r_ in runs if np.isfinite(chosen["loss"]) and r_["loss"] <= 2 * chosen["loss"]] or runs
    spread = {kk: float(np.std([r_["best"][kk] for r_ in good])) for kk in keys}
    std = {kk: max(chosen["std"][kk], spread[kk]) for kk in keys}
    return dict(best=chosen["best"], loss=chosen["loss"], history=chosen["history"], keys=list(keys),
                evals=iters * popsize * len(runs), std=std, use=use, val=val,
                restarts=[dict(best=r_["best"], loss=r_["loss"]) for r_ in runs],
                n_fit=sum(int(r["counts"].shape[0]) for r in fit_l), n_val=sum(int(r["counts"].shape[0]) for r in val_l))


def dr_around(fit, rel_width=0.15, base=None):
    """Domain-randomisation dict for sim.sample_domain centred on fitted values (+-rel_width relative)."""
    dr = dict(base or {})
    for k, v in fit["best"].items():
        lo, hi = FIT_BOUNDS[k]
        dr[k] = (max(lo, v * (1 - rel_width)), min(hi, v * (1 + rel_width)))
    return dr


KEEP_WIDE = ("mu", "s0", "crr", "izz_scale", "tau_bd")   # not identifiable from healthy single-module trials


def dr_from_sysid(fit, inflate=3.0, min_rel=0.03, max_rel_identified=0.25, base=None, gate_result=None,
                  keep_wide=KEEP_WIDE):
    """Retraining DR spec centred on a sys-ID result, widths from its uncertainty (A2).
    For each fitted key with value v and search/restart std s: half-width h = max(inflate * s, min_rel * |v|),
    range = [v - h, v + h] clipped to FIT_BOUNDS. Keys whose relative std exceeds max_rel_identified are treated as
    NOT identified and keep the `base` range (default sim.DR_WIDE), as do the `keep_wide` keys: healthy wheels need
    almost no traction, so mu/s0/crr (and izz, tau_bd) fitted on healthy trials say little about the faulted-wheel
    regime where they matter (A2 validation: a mu = 0.4 plant fitted to mu 0.71 with correct rho). If gate_result
    says the fit was not adopted
    (use == "nominal"), the base DR is returned unchanged. Pass the result to datagen.generate(dr=...)."""
    base = dict(sim.DR_WIDE if base is None else base)
    if (gate_result is not None and gate_result.get("use") == "nominal") or fit.get("use") == "nominal":
        return base
    dr = dict(base)
    for k, v in fit["best"].items():
        if k not in sim.DR_DEFAULT or k in keep_wide:
            continue
        s_ = float(fit.get("std", {}).get(k, 0.0))
        if abs(v) > 0 and s_ / abs(v) > max_rel_identified:
            continue
        h = max(inflate * s_, min_rel * abs(v))
        lo, hi = FIT_BOUNDS[k]
        dr[k] = (float(max(lo, v - h)), float(min(hi, v + h)))
    return dr


def make_pseudo_real(true_vals, params, n_trials=8, seed=0, device=None, fixed=None, jitter=None):
    """Sim-generated 'pseudo-real' healthy single-module RawTrials with known parameters (for validation).
    jitter: optional dict key -> relative per-trial std (e.g. dict(v_batt=0.01, mu=0.01)); default none."""
    from . import sensors
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    asm = A.physical("single", params)
    dom = _dom_from({k: v for k, v in true_vals.items()}, n_trials, params, device, fixed)
    if jitter:
        g = np.random.default_rng(seed + 991)
        for k, r in jitter.items():
            dom[k] = dom[k] * torch.tensor(1 + r * g.standard_normal(n_trials), dtype=torch.float32, device=device)
    mode = torch.zeros(n_trials, 4, dtype=torch.long)
    R = sim.simulate(asm, params, dom, mode, torch.ones(n_trials, 4), device=device, seed=seed)
    raw = sensors.to_raw(R, params, dom, np.random.default_rng(seed))
    return [raw]

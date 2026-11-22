"""Pipeline stages (each takes and returns plain data; run_all.py wires them together)."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from capxfer import assembly as A
from capxfer import featurize as FZ
from capxfer import sim
from capxfer.analysis import table as TB
from capxfer.analysis.frozen import eval_targets
from capxfer.conformal import capability as CAP, ood as OOD, wheel_sets as WS
from capxfer.field import loader as LD
from capxfer.probes import TARGET_KEYS

from . import adapters as AD

REPO = Path(__file__).resolve().parents[2]
PARAMS = REPO / "params" / "module_measured.yaml"
SEEN_PIDS = ("P1", "P5")


def params():
    return A.load_params(PARAMS)


def study_threshold():
    import yaml
    return float(yaml.safe_load((REPO / "configs" / "study_constants.yaml").read_text())["capability_threshold"])


# ----------------------------------------------------------------------------------------------- 1. load
def load_sessions(session_dirs, sched, allow_simulation=False, log=print):
    """FieldLog sessions -> {assembly_key: RawTrial} (same assembly across sessions concatenated) and per-trial meta.
    allow_simulation=False is the real-data setting: anything but FIELD_OK is refused by the loader."""
    sidx = AD.schedule_index(sched)
    by_asm = {}
    for d in session_dirs:
        raws = LD.load_session(d, allow_unverified=allow_simulation)
        for r in raws:
            if not allow_simulation and r["meta"]["source"] != "FIELD":
                raise LD.SessionRefused(f"{d}: source {r['meta']['source']}")
            by_asm.setdefault(AD.asm_key(r["meta"]["assembly"]), []).append(r)
            log(f"loaded {Path(d).name}: {r['meta']['assembly'].name} {len(r['meta']['trial_ids'])} trials "
                f"verdict {r['meta']['verdict']} source {r['meta']['source']}")
    out = {}
    for k, rs in by_asm.items():
        raw = AD.concat_raws(rs)
        out[k] = (raw, AD.trial_meta(raw, sidx))
    srcs = {m["source"][0] for _, m in out.values()}
    if len(srcs) != 1:
        raise ValueError(f"mixed sources across sessions: {srcs}")
    return out


# ----------------------------------------------------------------------------------------------- 2. sys-ID gate
def _sim_rho(asm, prm, dom_vals, B, device):
    from capxfer import sysid
    dom = sysid._dom_from(dom_vals, B, prm, device)                    # M7: private helper, no public nominal-dom API
    R = sim.simulate(asm, prm, dom, torch.zeros(B, asm.n, dtype=torch.long), torch.ones(B, asm.n), device=device)
    rho, _ = sim.targets_from_twist(R)
    return rho.float().mean(0).cpu().numpy()


def sysid_stage(loaded, cfg, log=print):
    """Fit sim parameters on the first half of the healthy single-module trials, gate on the second half.
    Gate (WS-D note, prereg): median |rho_sim(fitted) - rho_real| over held-out healthy trials x 14 keys <= 0.05."""
    from capxfer import sysid
    prm = params()
    dev = AD.gpu_budget()
    picked = []
    for raw, meta in loaded.values():
        idx = AD.healthy_single_idx(raw, meta)
        if idx:
            picked.append(AD.subset_raw(raw, idx))
    if not picked:
        return dict(status="NOT_RUN (no healthy single-module trials)")
    raw = AD.concat_raws(picked) if len(picked) > 1 else picked[0]
    n = len(raw["meta"]["trial_ids"])
    # interleaved split per the prereg gate rule: odd trial order -> fit, even -> gate
    fit_idx, ho_idx = list(range(0, n, 2)), list(range(1, n, 2))
    nf = len(fit_idx)
    fit_raw, ho_raw = AD.subset_raw(raw, fit_idx), AD.subset_raw(raw, ho_idx)
    t0 = time.time()
    fit = sysid.fit_params([fit_raw], prm, iters=cfg.get("sysid_iters", 20), popsize=cfg.get("sysid_popsize", 16),
                           device=dev, seed=cfg.get("seed", 0))
    log(f"sysid fit {time.time() - t0:.0f} s loss {fit['loss']:.3f} " +
        " ".join(f"{k}={v:.3g}" for k, v in fit["best"].items()))
    out = dict(fit=fit["best"], loss=fit["loss"], evals=fit["evals"], n_fit=nf, n_heldout=n - nf,
               fit_trials=fit_raw["meta"]["trial_ids"], heldout_trials=ho_raw["meta"]["trial_ids"],
               wall_s=time.time() - t0)
    if n - nf > 0:
        # Comparative gate (WS-A2; prereg): fitted must beat defaults on held-out healthy trials by a margin and be under
        # an absolute cap; otherwise downstream uses the defaults.
        g = sysid.gate(fit, [ho_raw], prm, device=dev)
        out.update(heldout_error_fitted=g["e_fit"], heldout_error_nominal=g["e_nom"],
                   heldout_error_fitted_per_key=g["e_fit_per_key"], heldout_error_nominal_per_key=g["e_nom_per_key"],
                   gate_margin=g["margin"], gate_cap=g["cap"], gate_pass=g["gate_pass"], use=g["use"])
    out["status"] = "RUN"
    return out


# ----------------------------------------------------------------------------------------------- 3. predict
def predict_stage(loaded, methods, rungs, model_dir, suffix, n_samples=256, log=print, keep=(), pids=SEEN_PIDS):
    """-> tables {method: {rung: PredTable}}, kept {(method, rung): {asm_key: Pred numpy}} for `keep` pairs.
    pids: the probes the predictors see (default P1+P5; the H2 probe ladder passes larger sets)."""
    tables, kept, timing = {}, {}, {}
    for m in methods:
        t0 = time.time()
        model = AD.load_predictor(AD.model_path(model_dir, m, suffix))
        q = model.cfg.get("q_levels", (0.05, 0.5, 0.95))
        for r in rungs.get(m, rungs.get("*", ("E2",))):
            preds, ys, labs, metas = [], [], [], []
            for key, (raw, meta) in loaded.items():
                F = FZ.featurize(raw, r, tuple(pids))
                with torch.no_grad():
                    p = AD.pred_numpy(model.predict(F, n_samples=n_samples), q)
                preds.append(p)
                ys.append(F["y"].cpu().numpy())
                labs.append(dict(mode=raw["meta"]["labels"]["mode"].numpy()))
                metas.append({k: meta[k] for k in TB.STR_KEYS + TB.OPT_STR_KEYS if k in meta})
                if (m, r) in keep:
                    kept.setdefault((m, r), {})[key] = dict(pred=p, y=ys[-1])
            tables.setdefault(m, {})[r] = TB.from_preds(preds, ys, labs, metas)
        timing[m] = time.time() - t0
        log(f"predict {m} rungs {list(tables[m])}: {timing[m]:.0f} s")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return tables, kept, timing


def save_tables(tables, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for m, rr in tables.items():
        for r, t in rr.items():
            p = out_dir / f"pred_{m}_{r}.npz"
            TB.save(p, t)
            paths.setdefault(m, {})[r] = str(p)
    return paths


# H2 probe ladder (prereg, approved 2026-11-20): ordered probe sets for the ladder method at the primary rung
PROBE_LADDER = (("P1P5", ("P1", "P5")), ("P1P5P2", ("P1", "P5", "P2")), ("P1-P5", ("P1", "P2", "P3", "P4", "P5")),
                ("all", ("P1", "P2", "P3", "P4", "P5", "P6x", "P6y")))


def probe_ladder_stage(loaded, method, rung, model_dir, suffix, out_dir, n_samples=256, log=print,
                       ladder=PROBE_LADDER):
    """One PredTable per probe-ladder rung for `method` at `rung` -> [{label, pids, table}] for the analysis manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for label, pids in ladder:
        tabs, _, tm = predict_stage(loaded, [method], {method: (rung,)}, model_dir, suffix, n_samples, log=lambda *_: None,
                                    pids=pids)
        p = out_dir / f"ladder_{method}_{rung}_{label}.npz"
        TB.save(p, tabs[method][rung])
        entries.append(dict(label=label, pids=list(pids), table=str(p), method=method, rung=rung))
        log(f"probe ladder {method} {rung} {label}: {tm[method]:.0f} s")
    return entries


# ----------------------------------------------------------------------------------------------- 4. explanations
def capability_overclaim(explanations, y_true, decision, keys=TARGET_KEYS):
    """Per explanation: True iff it certifies (supported claim value "reliable") some decision leg whose true ratio is
    below the decision threshold. Matches analysis.frozen.overclaim on the certified decision legs."""
    tau = float(decision["threshold"])
    req = set(decision["required"])
    out = []
    for e, y in zip(explanations, np.asarray(y_true, float)):
        oc = False
        for c in e.get("claims", []):
            if (c.get("type") == "capability" and c.get("support") == "supported" and c.get("value") == "reliable"
                    and c.get("subject") in req and y[list(keys).index(c["subject"])] < tau):
                oc = True
        out.append(oc)
    return np.array(out, bool)


def explain_stage(table, loaded, cfg, rung="E2", log=print, wheel_table=None):
    """Per LOAO fold: conformal (WS-D) on the other types' in-model trials -> intervals, wheel sets, OOD flag for the
    held-out type -> WS-E explain() and point_estimate() -> WS-E score() -> the explain CSV run_frozen reads (H3).
    Uses the same calibration rule as analysis.frozen.fold_eval (split joint CQR on the unseen-leg targets).

    D2 wiring (prereg H1(a)/H3): `table` is the CAPABILITY method's PredTable (the deployed `ensemble`): its split_joint
    bound gives the capability claims and its OOD flag gates them. `wheel_table` (default: `table`) is the DIAGNOSIS
    method's PredTable (`structured_ensemble`): its APS sets give the wheel-mode / cause claims and its mode_prob the
    point-estimate baseline's wheel claims. Both tables must hold the same trials in the same order. The CSV also
    carries `capability_overclaim` (a certified decision leg whose true ratio is < tau)."""
    from capxfer import explain as EX
    a = cfg.get("alpha", 0.10)
    tau = study_threshold()
    decision = dict(task="sideways dock (strafe)", required=list(cfg.get("decision_required", ("P2+", "P2-"))),
                    threshold=tau)
    T = eval_targets(SEEN_PIDS)
    wt = table if wheel_table is None else wheel_table
    if not np.array_equal(np.asarray(wt["trial_id"]), np.asarray(table["trial_id"])):
        raise ValueError("explain_stage: capability and wheel tables hold different trials")
    asm_of = {raw["meta"]["assembly"].name: raw["meta"]["assembly"] for raw, _ in loaded.values()}
    rows, folds = [], {}
    for fold in sorted(set(table["assembly_type"].tolist())):
        cal = (table["assembly_type"] != fold) & (table["ood"] == "none")
        test = np.flatnonzero(table["assembly_type"] == fold)
        tc, tt = TB.subset(table, np.flatnonzero(cal)), TB.subset(table, test)
        wc, wtt = TB.subset(wt, np.flatnonzero(cal)), TB.subset(wt, test)
        pc, pt = TB.as_pred(tc), TB.as_pred(tt)
        pwc, pwt = TB.as_pred(wc), TB.as_pred(wtt)
        cc = CAP.calibrate(pc, tc["y"], a, method="split", joint=True, targets=T, seed=cfg.get("seed", 0))
        lo, hi = (np.asarray(v) for v in cc.interval(pt))
        asm = asm_of[fold]
        n = asm.n
        sets = None
        if "mode_prob" in pwc and np.isfinite(np.asarray(wc["mode_prob"])[:, :n]).all():
            sc = WS.calibrate(pwc, wc["mode_true"], a, pooling="joint", seed=cfg.get("seed", 0))
            sets = np.asarray(sc.sets(pwt))[:, :n]                      # M8: PredTable pads to 16 wheels; explain wants n
        oc = OOD.calibrate(tc["ppc_score"], a)
        flag = np.asarray(oc.flag(tt["ppc_score"]))
        mp = wtt["mode_prob"][:, :n] if "mode_prob" in wtt else None
        pred = dict(y_q=tt["y_q"], mode_prob=mp)
        cal_ex = EX.explain(pred, (lo, hi), sets, flag, asm, rung, decision=decision)
        pe_ex = EX.point_estimate(pred, asm, rung, decision=decision)
        labels = dict(mode=tt["mode_true"][:, :n], ood=list(tt["ood"]))
        folds[fold] = dict(n_cal=int(cal.sum()), n_test=int(test.size), ood_threshold=float(oc.threshold),
                           cap_threshold=np.asarray(cc.threshold).tolist(), n_ood_flagged=int(flag.sum()))
        for name, ex in (("calibrated", cal_ex), ("point_estimate", pe_ex)):
            sc_ = EX.score(ex, labels, tt["y"], decision=decision)
            ocl = capability_overclaim(ex, tt["y"], decision)
            folds[fold][name] = {k: v for k, v in sc_.items() if k != "per_explanation"}
            folds[fold][name]["capability_overclaim_rate"] = float(ocl.mean()) if ocl.size else math.nan
            for i, (e, s) in enumerate(zip(ex, sc_["per_explanation"])):
                rows.append(dict(trial_id=tt["trial_id"][i], explainer=name, fold=fold,
                                 n_unsupported_cause=int(bool(s["unsupported_cause"])),
                                 wheel_cause_asserted=int(bool(s["names_wheel_fault"])),
                                 correct_claim_rate=s["correct_rate"], informativeness=s["informativeness"],
                                 verified=bool(s["verified"]), outcome=s.get("outcome"),
                                 capability_overclaim=bool(ocl[i]), ood_flag=bool(flag[i]),
                                 text=e["text"].replace("\n", " ")))
        log(f"explain fold {fold}: {test.size} trials, calibrated unsupported "
            f"{folds[fold]['calibrated']['unsupported_cause_rate']:.3f}, point unsupported "
            f"{folds[fold]['point_estimate']['unsupported_cause_rate']:.3f}")
    return pd.DataFrame(rows), dict(decision=decision, folds=folds)


# ----------------------------------------------------------------------------------------------- 5. docking
def docking_stage(src, out_csv, assembly="single", fault="F4x2d"):
    """STUB for the dry run: the docking arms come from WS-F's SIM pilot rows (jsonl), not from field docking runs.
    Real data: pass a CSV with arm,run,success[,final_pose_error] written from the AprilTag docking log."""
    src = Path(src)
    if src.suffix == ".csv":
        df = pd.read_csv(src)
    else:
        rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r.get("assembly") == assembly and r.get("fault") == fault]
        df = pd.DataFrame([dict(arm=r["arm"], run=int(r["run"]), success=int(bool(r["success"])),
                                final_pose_error=float(r.get("pos_err", math.nan)), source="SIMULATION_PILOT_STUB")
                           for r in rows])
    if not {"arm", "run", "success"} <= set(df.columns):
        raise ValueError("docking CSV needs arm, run, success")
    df.to_csv(out_csv, index=False)
    return df


# ----------------------------------------------------------------------------------------------- 6. sim held-out (H5)
def _truncate_compact(c, k):
    B = c["counts"].shape[0]
    out = {}
    for key, v in c.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B:
            out[key] = v[:k].clone()
        else:
            out[key] = v
    m = dict(c["meta"])
    m["trial_ids"] = list(m["trial_ids"])[:k]
    out["meta"] = m
    lab = {}
    for key, v in c["labels"].items():
        lab[key] = v[:k].clone() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B else v
    if "domain" in lab and isinstance(lab["domain"], dict):
        lab["domain"] = {kk: (vv[:k] if torch.is_tensor(vv) and vv.dim() >= 1 and vv.shape[0] == B else vv)
                         for kk, vv in lab["domain"].items()}
    out["labels"] = lab
    return out


def sim_heldout_stage(shard, methods, model_dir, suffix, rung="E2", max_groups=12, per_group=48, min_n=16,
                      max_n=TB.MAX_WHEELS, n_samples=256, log=print):
    """Sim held-out PredTables for H5 from one WS-A shard: random families with min_n <= n <= max_n wheels (size 4,
    never in training), in-model trials only. M10: analysis.table.MAX_WHEELS = 16 rejects the size-5/6 (20/24-wheel)
    held-out families WS-C evaluates on, so they are skipped until WS-D raises the pad to 24."""
    from capxfer import datagen
    # M9: datagen.load_shard expands every group of an ~800 MB shard to float32 at once (> available RAM here);
    # filter and truncate the compact groups first, then expand only those.
    # D3b: `shard` may be a list of shards (H5 power needs a few thousand sim held-out trials; one devW shard has only
    # ~13 qualifying groups); groups are taken in shard order up to max_groups.
    shards = [shard] if isinstance(shard, (str, Path)) else list(shard)
    raws, stems = [], []
    for sh in shards:
        if len(raws) >= max_groups:
            break
        comp = torch.load(sh, weights_only=False)
        keep = [c for c in comp if c["labels"].get("asm_type") == "random" and min_n <= int(c["meta"]["n"]) <= max_n
                and c["labels"].get("ood", "none") == "none"][:max_groups - len(raws)]
        del comp
        raws += [datagen.expand(_truncate_compact(c, per_group)) for c in keep]
        stems += [Path(sh).stem] * len(keep)
        del keep
    code = {0: "F0", 1: "F1", 2: "F2", 3: "F3", 4: "F4"}
    sel = []
    for gi, r in enumerate(raws):
        mode = r["meta"]["labels"]["mode"].cpu().numpy()
        nf = (mode != 0).sum(1)
        fault = [("F0" if k == 0 else "F5" if k > 1 else code[int(mode[i][mode[i] != 0][0])]) for i, k in enumerate(nf)]
        fam = str(r["meta"]["labels"].get("family"))
        N = len(fault)
        meta = dict(trial_id=np.array([f"sim:{stems[gi]}:{gi}:{i}" for i in range(N)]),
                    assembly_type=np.full(N, fam), session=np.full(N, f"sim{gi}"), module_set=np.full(N, fam),
                    fault=np.array(fault), ood=np.full(N, "none"), source=np.full(N, "SIMULATION"))
        sel.append((r, meta))
    tables = {}
    for m in methods:
        model = AD.load_predictor(AD.model_path(model_dir, m, suffix))
        q = model.cfg.get("q_levels", (0.05, 0.5, 0.95))
        preds, ys, labs, metas = [], [], [], []
        for r, meta in sel:
            F = FZ.featurize(r, rung, SEEN_PIDS)
            with torch.no_grad():
                preds.append(AD.pred_numpy(model.predict(F, n_samples=n_samples), q))
            ys.append(F["y"].cpu().numpy())
            labs.append(dict(mode=r["meta"]["labels"]["mode"].cpu().numpy()))
            metas.append(meta)
        tables[m] = TB.from_preds(preds, ys, labs, metas)
        del model
    log(f"sim held-out: {len(sel)} families, {sum(len(m['trial_id']) for _, m in sel)} trials, methods {methods}")
    return tables


# ----------------------------------------------------------------------------------------------- 7. paper
def paper_stage(results_json, manifest, out_dir, log=print):
    """Copy paper/ (WS-G owns it; never written in place), bind values with its fill_results.py, build the PDF."""
    import os
    out_dir = Path(out_dir)
    pdir = out_dir / "paper"
    if pdir.exists():
        shutil.rmtree(pdir)
    shutil.copytree(REPO / "paper", pdir, ignore=shutil.ignore_patterns("generated"))
    py = str(Path(REPO / ".venv/bin/python"))
    fr = subprocess.run([py, str(pdir / "fill_results.py"), str(results_json), "--manifest", str(manifest),
                         "--out", str(pdir / "generated" / "results_values.tex")], capture_output=True, text=True)
    missing = [l.strip() for l in fr.stderr.splitlines() if "MISSING" in l]
    log(fr.stdout.strip())
    env = dict(os.environ)
    tb = Path.home() / ".local" / "bin" / "tectonic"
    if "TECTONIC_BIN" not in env and tb.is_file():
        env["TECTONIC_BIN"] = str(tb)
        env.setdefault("TECTONIC_ONLY_CACHED", "1")
    bp = subprocess.run(["bash", str(pdir / "build_paper.sh"), str(out_dir / "paper_build")], capture_output=True,
                        text=True, env=env)
    summary = [l for l in bp.stdout.splitlines() if l.startswith("capxfer paper:")]
    pdf = out_dir / "paper_build" / "main.pdf"
    log(summary[0] if summary else bp.stderr[-2000:])
    return dict(fill_rc=fr.returncode, fill_stdout=fr.stdout.strip(), missing=missing, build_rc=bp.returncode,
                build_summary=summary[0] if summary else None, pdf=str(pdf) if pdf.is_file() else None,
                build_stderr_tail=bp.stderr[-1500:] if bp.returncode else "")

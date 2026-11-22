"""The single post-collection command: sessions + frozen models + freeze manifest -> results.json, figure CSVs, paper.

  python -m capxfer.pipeline.run_all \
      --freeze FREEZE.json --schedule SCHED_DIR/schedule.json --sessions S1 S2 ... [--sysid-sessions D ...] \
      [--sysid-schedule SYSID_SCHED/schedule.json] --docking docking.csv [--sim-shard shard.pt] --out OUT

Stages (OUT/run_report.json records each one with its wall time):
  0 freeze   verify every frozen file's sha256 (freeze.py); models come from the freeze's model dir; the sealed schedule
             must be the frozen one. Any mismatch refuses the run (dry runs with --allow-simulation only warn).
  1 load     field.loader.load_session (FIELD_OK only, unless --allow-simulation) -> RawTrials; join the sealed schedule
             for fault ids (F0..F7) and cross-check the logged fault echoes against it.
  2 sysid    sysid.fit_params on healthy single-module trials (half), gate on the other half (provenance.sysid).
  3 predict  featurize (P1+P5) at each rung -> frozen predictors -> PredTables (analysis/table.py), OUT/tables/.
  3b ladder  H2 probe ladder: structured_ensemble at E2 with P1P5 / P1P5P2 / P1-P5 / all -> OUT/tables/ladder_*.npz.
  4 explain  per LOAO fold: `ensemble` split-joint intervals + OOD flag (capability claims), `structured_ensemble`
             APS wheel sets (wheel claims) -> WS-E explain + point_estimate ->
             WS-E score -> OUT/explain.csv.
  5 docking  docking CSV (arm,run,success,...) -> OUT/docking.csv (dry run: WS-F sim pilot rows, a stub).
  6 sim      optional sim held-out PredTables (H5) from a WS-A shard.
  7 analysis analysis.run_frozen(OUT/analysis_manifest.json) -> OUT/analysis/results.json + fig_*.csv.
  8 paper    WS-G paper/fill_results.py + build_paper.sh on a COPY of paper/ -> OUT/paper_build/main.pdf.
Nothing is written outside OUT.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from capxfer.field import schedule as SCH

from . import freeze as FR
from . import stages as ST

DEFAULT_METHODS = ("structured_npe", "structured_ensemble", "blackbox", "ensemble", "analytic_bayes", "nominal_sim")
# D2 (prereg approved 2026-11-20): capability claims / H1(a) / H3 certificate from `ensemble`; wheel claims / H1(b)
# from `structured_ensemble`; H2 probe ladder on `structured_ensemble` at E2.
CAP_METHOD, DIAG_METHOD, LADDER_METHOD, LADDER_RUNG = "ensemble", "structured_ensemble", "structured_ensemble", "E2"
DEFAULT_RUNGS = {"structured_npe": ("E0", "E1", "E2", "E3"), "*": ("E2",)}


def _log_to(path):
    fh = open(path, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    return log


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = _log_to(out / "run_all.log")
    rep = dict(started=time.strftime("%Y-%m-%dT%H:%M:%S"), args=vars(args), stages={})
    t_all = time.time()

    def stage(name, fn):
        t0 = time.time()
        log(f"== {name}")
        try:
            v = fn()
            rep["stages"][name] = dict(ok=True, wall_s=round(time.time() - t0, 1))
            return v
        except Exception as e:
            rep["stages"][name] = dict(ok=False, wall_s=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}",
                                       traceback=traceback.format_exc()[-3000:])
            (out / "run_report.json").write_text(json.dumps(rep, indent=1, default=str))
            raise

    # 0 freeze
    def _freeze():
        man = FR.load(args.freeze)
        probs = FR.verify(man)
        sched_sha = FR.sha256_file(args.schedule)
        if sched_sha != man["schedule_sha256"]:
            probs.append(f"schedule {args.schedule} sha {sched_sha[:12]} != frozen {man['schedule_sha256'][:12]}")
        for m in args.methods:
            if m not in man["models"]:
                probs.append(f"method {m} has no frozen model")
        for p in probs:
            log(f"FREEZE MISMATCH: {p}")
        if probs and not args.allow_simulation:
            raise RuntimeError(f"{len(probs)} freeze mismatch(es); refusing the real analysis")
        return dict(manifest=man, problems=probs)
    fz = stage("freeze", _freeze)
    man = fz["manifest"]
    rep["freeze"] = dict(aggregate_sha256=man["aggregate_sha256"], problems=fz["problems"])
    model_dir, suffix = man["roots"]["models"], man.get("model_suffix", "")

    # 1 load
    sched = SCH.load(args.schedule)
    loaded = stage("load", lambda: ST.load_sessions(args.sessions, sched, args.allow_simulation, log))
    source = next(iter(loaded.values()))[1]["source"][0]
    n_trials = sum(len(m["trial_id"]) for _, m in loaded.values())
    rep["source"], rep["n_trials"] = source, n_trials
    log(f"{n_trials} trials, {len(loaded)} assemblies, source {source}")

    # 2 sysid
    cfg = dict(seed=args.seed, sysid_iters=args.sysid_iters, sysid_popsize=args.sysid_popsize, alpha=0.10)
    def _sysid():
        if args.sysid_sessions:
            ss = SCH.load(args.sysid_schedule)
            src = ST.load_sessions(args.sysid_sessions, ss, args.allow_simulation, log)
        else:
            src = loaded
        return ST.sysid_stage(src, cfg, log)
    sy = stage("sysid", _sysid)
    rep["sysid"] = {k: v for k, v in sy.items() if not k.endswith("_trials")}

    # 3 predict
    rungs = dict(DEFAULT_RUNGS)
    tables, _, timing = stage("predict", lambda: ST.predict_stage(loaded, args.methods, rungs, model_dir, suffix,
                                                                     args.n_samples, log))
    paths = ST.save_tables(tables, out / "tables")
    rep["predict_timing_s"] = timing

    # 3b H2 probe ladder
    ladder = []
    if args.ladder_method and args.ladder_method in args.methods:
        ladder = stage("probe_ladder", lambda: ST.probe_ladder_stage(loaded, args.ladder_method, LADDER_RUNG, model_dir,
                                                                     suffix, out / "tables", args.n_samples, log))

    # 4 explain
    def _pick(m):
        return m if m in tables and "E2" in tables[m] else "structured_npe"
    cap_m, diag_m = _pick(args.cap_method), _pick(args.diag_method)
    rep["explain_methods"] = dict(capability=cap_m, wheel=diag_m)
    ex_df, ex_info = stage("explain", lambda: ST.explain_stage(tables[cap_m]["E2"], loaded, cfg, "E2", log,
                                                               wheel_table=tables[diag_m]["E2"]))
    ex_info["methods"] = rep["explain_methods"]
    ex_df.to_csv(out / "explain.csv", index=False)
    (out / "explain_folds.json").write_text(json.dumps(ex_info, indent=1, default=str))

    # 5 docking
    dk = stage("docking", lambda: ST.docking_stage(args.docking, out / "docking.csv")) if args.docking else None

    # 6 sim held-out
    sim_paths = {}
    if args.sim_shard:
        sims = stage("sim_heldout", lambda: ST.sim_heldout_stage(args.sim_shard, ["structured_npe", "nominal_sim"],
                                                                 model_dir, suffix, max_groups=args.sim_max_groups,
                                                                 per_group=args.sim_per_group, log=log))
        for m, t in sims.items():
            p = out / "tables" / f"sim_{m}.npz"
            ST.TB.save(p, t)
            sim_paths[m] = str(p)

    # provenance (sysid + counts that PredTables cannot carry)
    prov = dict(counts=dict(n_sysid_trials=int(sy.get("n_fit", 0) + sy.get("n_heldout", 0))), sysid=rep["sysid"])
    if args.train_manifest and Path(args.train_manifest).is_file():
        tm = json.loads(Path(args.train_manifest).read_text())
        prov["counts"]["n_sim_train"] = int(tm["counts"]["trials"])
        prov["counts"]["n_sim_families"] = len(tm["counts"].get("by_family", {}))
        prov["train_manifest"] = dict(path=args.train_manifest, sha256=FR.sha256_file(args.train_manifest),
                                      params_sha256=tm.get("params_sha256"))
    prov["freeze_aggregate_sha256"] = man["aggregate_sha256"]
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))

    # 7 analysis
    am = dict(output_dir="analysis", pred_tables=paths, explain_scores="explain.csv", provenance="provenance.json",
              frozen=bool(source == "FIELD" and not fz["problems"]), config=dict(n_boot=args.n_boot, seed=args.analysis_seed))
    if dk is not None:
        am["docking"] = "docking.csv"
    if sim_paths:
        am["sim_tables"] = sim_paths
    if ladder:
        am["probe_ladder"] = [dict(label=e["label"], pids=e["pids"], table=e["table"]) for e in ladder]
        am["probe_ladder_method"] = dict(method=args.ladder_method, rung=LADDER_RUNG)
    (out / "analysis_manifest.json").write_text(json.dumps(am, indent=1))
    from capxfer.analysis.frozen import run_frozen
    res = stage("analysis", lambda: run_frozen(out / "analysis_manifest.json"))
    rep["results"] = res["results_path"]
    rep["status"] = res.get("status")
    rep["hypotheses"] = res.get("hypotheses")

    # 8 paper
    if not args.no_paper:
        rep["paper"] = stage("paper", lambda: ST.paper_stage(res["results_path"], out / "analysis_manifest.json", out, log))
    rep["wall_s"] = round(time.time() - t_all, 1)
    (out / "run_report.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"done in {rep['wall_s']:.0f} s -> {out}")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", required=True, help="freeze manifest from capxfer.pipeline.freeze")
    ap.add_argument("--schedule", required=True, help="sealed schedule.json (schedule.sha256 next to it)")
    ap.add_argument("--sessions", nargs="+", required=True, help="FieldLog v2 session dirs of the main collection")
    ap.add_argument("--sysid-sessions", nargs="*", default=[], help="healthy single-module sys-ID sessions")
    ap.add_argument("--sysid-schedule", help="schedule.json of the sys-ID sessions")
    ap.add_argument("--docking", help="docking CSV (arm,run,success[,final_pose_error]) or WS-F pilot jsonl (stub)")
    ap.add_argument("--sim-shard", nargs="+", help="WS-A datagen shard(s) for the sim held-out tables (H5)")
    ap.add_argument("--sim-max-groups", type=int, default=12, help="H5 sim held-out: max assembly groups (D3b: use >= 60)")
    ap.add_argument("--sim-per-group", type=int, default=48, help="H5 sim held-out: trials per group (D3b: 64)")
    ap.add_argument("--train-manifest", default=str(Path.home() / ".cache/capxfer/A/trainW/manifest.json"))
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    ap.add_argument("--cap-method", default=CAP_METHOD, help="capability claims / overclaim (falls back to structured_npe)")
    ap.add_argument("--diag-method", default=DIAG_METHOD, help="wheel-mode / cause claims (APS sets)")
    ap.add_argument("--ladder-method", default=LADDER_METHOD, help="H2 probe-ladder method ('' = skip)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-simulation", action="store_true", help="dry run: load SIMULATION sessions, warn on freeze")
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analysis-seed", type=int, default=20261120)
    ap.add_argument("--sysid-iters", type=int, default=20)
    ap.add_argument("--sysid-popsize", type=int, default=16)
    ap.add_argument("--no-paper", action="store_true")
    a = ap.parse_args(argv)
    if a.sysid_sessions and not a.sysid_schedule:
        ap.error("--sysid-sessions needs --sysid-schedule")
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())

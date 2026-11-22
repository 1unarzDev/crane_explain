"""Frozen choice bundle for the field day (WS-I3F) and the ONE calibration-constant format (`lambda_v2.json`).

Robot laptop (CPU only, offline, no cluster):
  python -m capxfer.field.bundle verify BUNDLE_DIR [--lambda lambda_v2.json]     # hashes, code version, CPU model load
AI side, at the freeze (reads ~/.cache, writes one folder that is copied unchanged to the robot laptop):
  python -m capxfer.field.bundle build OUT_DIR [--models ~/.cache/capxfer/C/models/W] [--library LIB.npz]
         [--wheelsets WS.npz] [--bb-offset T] [--tau 0.8] [--note ...]
Analysis side, between MASK and DEMO (PROTOCOL_V2 §9): `capxfer.field.builders lambda ...` writes lambda_v2.json with
`make_lambda_record`; `python -m capxfer.field.bundle check-lambda FILE [--bundle DIR]` validates it.

Bundle = a `capxfer.choice_bundle/v1` folder (choose_module.write_bundle) plus, from this module:
  * `provenance` in bundle.yaml: model sha256 (+ the W manifest entries, training params_sha256), the current
    params/module_measured.yaml sha256, the library / wheel-set / carry-over-offset sources, git commit, and the
    certificate code version = sha256 over the sorted (path, sha256) list of the certificate code files;
  * `code/`: a verbatim copy of those code files (listed in bundle.yaml `files`, so load_bundle hash-checks them);
  * `lambda_v2.PLACEHOLDER.json`: the calibration file with every constant null (status PLACEHOLDER). The real file is
    written on the field day from the MASK CAL units only (PROTOCOL_V2 §3, §9) and hashed into the freeze record;
  * `MANIFEST.sha256`: sha256 of every file in the folder (bundle.yaml included), `sha256sum -c` compatible.

lambda_v2.json (schema `capxfer.lambda_v2/v1`; the only format choose_module, builders and realv2 exchange):
  schema, status (FINAL | PLACEHOLDER | SYNTHETIC), lambda_hat (alpha 0.10) and lambda_hat_alpha05: float or null (null =
  +inf = nothing certified), alpha, alpha_sens, tau, n_null_cal, n_null_cal_needed {alpha: n}, carry_over_offset (float
  or null = +inf: the carry-over never says go), carry_over_alpha, n_carry_cal, rule, score, bundle_sha256 (bundle.yaml
  sha256 the scores were computed with; null only for PLACEHOLDER), inputs {name: sha256}, created_utc, note.
  Invariants: lambda_hat_alpha05 >= lambda_hat (a smaller alpha takes a higher order statistic); a finite lambda needs
  n_null_cal >= ceil(1/alpha) - 1; FINAL needs a bundle_sha256. Legacy files ({"lambda_hat", "lambda_hat_alpha05"} only,
  as the P2 dry run writes) read as status LEGACY and are accepted on SIMULATION sessions only (choose_module).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
MODELS_W = Path.home() / ".cache" / "capxfer" / "C" / "models" / "W"
SPLIT_LIBRARY = Path.home() / ".cache" / "capxfer" / "SPLIT" / "library.npz"
PARAMS = REPO / "params" / "module_measured.yaml"

LAMBDA_SCHEMA = "capxfer.lambda_v2/v1"
LAMBDA_STATUS = ("FINAL", "PLACEHOLDER", "SYNTHETIC")
ALPHA, ALPHA_SENS, TAU = 0.10, 0.05, 0.8
LAMBDA_RULE = ("lambda_hat = ceil((1-alpha)(n0+1))-th smallest certificate score among the n0 UNSAFE (z_alone < tau) "
               "CAL module units of the MASK assembly trials (realv2.classcond_threshold); +inf (null) if n0 < "
               "ceil(1/alpha)-1; certify iff s > lambda_hat. carry_over_offset = same order statistic of "
               "(black-box q05 - measured current-body z) over the CAL MASK assembly trials; carry-over lb = q05 - offset")
LAMBDA_SCORE = ("SPLIT (a) COMP lower bound on the module's post-split strafe z from the assembly's P1+P5+P2 evidence "
                "(capxfer.field.choose_module.certificate_bounds with the frozen bundle)")

# certificate code: everything between a logged trial file and the module score (hashed + copied into the bundle)
CODE_FILES = ("capxfer/field/choose_module.py", "capxfer/field/bundle.py", "capxfer/field/loader.py",
              "capxfer/field/schedule.py", "capxfer/field/fieldlog_config.py", "capxfer/featurize.py",
              "capxfer/probes.py", "capxfer/assembly.py", "capxfer/learn/base.py", "capxfer/learn/ensemble.py",
              "capxfer/learn/data.py", "capxfer/learn/nets.py", "capxfer/learn/_stub.py",
              "capxfer/learn/blackbox.py", "capxfer/learn/structured_npe.py", "capxfer/ideas/split/evaluate.py", "capxfer/ideas/split/common.py",
              "capxfer/ideas/i1/stage1.py")


class LambdaFormatError(ValueError):
    pass


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# =============================================================================================== lambda_v2.json
def n_needed(alpha):
    """Smallest n0 with a finite class-conditional threshold: ceil((1-a)(n0+1)) <= n0  <=>  n0 >= ceil(1/a) - 1."""
    n = 0
    while math.ceil((1 - alpha) * (n + 1) - 1e-12) > n:
        n += 1
    return n


def _fin(v):
    return None if v is None or not math.isfinite(float(v)) else float(v)


def make_lambda_record(lambda_hat, lambda_hat_alpha05, n_null_cal, carry_over_offset=None, n_carry_cal=0,
                       status="FINAL", bundle_sha256=None, inputs=None, alpha=ALPHA, alpha_sens=ALPHA_SENS, tau=TAU,
                       note=""):
    """-> a validated lambda_v2 dict. +inf / None thresholds are written as null."""
    rec = {"schema": LAMBDA_SCHEMA, "status": status, "lambda_hat": _fin(lambda_hat),
           "lambda_hat_alpha05": _fin(lambda_hat_alpha05), "alpha": float(alpha), "alpha_sens": float(alpha_sens),
           "tau": float(tau), "n_null_cal": int(n_null_cal),
           "n_null_cal_needed": {str(alpha): n_needed(alpha), str(alpha_sens): n_needed(alpha_sens)},
           "carry_over_offset": _fin(carry_over_offset), "carry_over_alpha": float(alpha),
           "n_carry_cal": int(n_carry_cal), "rule": LAMBDA_RULE, "score": LAMBDA_SCORE,
           "bundle_sha256": bundle_sha256, "inputs": dict(inputs or {}), "created_utc": _now(), "note": note}
    errs = validate_lambda(rec)
    if errs:
        raise LambdaFormatError("; ".join(errs))
    return rec


def placeholder_lambda(note="PLACEHOLDER: fill from the MASK CAL units on the field day (PROTOCOL_V2 §3, §9)"):
    return make_lambda_record(None, None, 0, None, 0, status="PLACEHOLDER", note=note)


def validate_lambda(d):
    """-> list of format errors ([] = valid). Legacy dicts (no `schema`) need only lambda_hat (+ optional alpha05)."""
    errs = []
    if not isinstance(d, dict):
        return ["not a JSON object"]

    def num_or_null(k):
        v = d.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)):
            errs.append(f"{k} must be a finite number or null (null = +inf), got {v!r}")
            return False
        return True

    if "schema" not in d:
        if "lambda_hat" not in d:
            errs.append("legacy lambda file without lambda_hat")
        num_or_null("lambda_hat")
        num_or_null("lambda_hat_alpha05")
        return errs
    if d["schema"] != LAMBDA_SCHEMA:
        return [f"schema {d['schema']!r} != {LAMBDA_SCHEMA}"]
    for k in ("status", "lambda_hat", "lambda_hat_alpha05", "alpha", "alpha_sens", "tau", "n_null_cal",
              "carry_over_offset", "carry_over_alpha", "n_carry_cal", "bundle_sha256"):
        if k not in d:
            errs.append(f"missing key {k}")
    if errs:
        return errs
    if d["status"] not in LAMBDA_STATUS:
        errs.append(f"status {d['status']!r} not in {LAMBDA_STATUS}")
    ok = all([num_or_null(k) for k in ("lambda_hat", "lambda_hat_alpha05", "carry_over_offset")])
    for k in ("alpha", "alpha_sens", "carry_over_alpha"):
        if not (isinstance(d[k], (int, float)) and 0 < d[k] < 1):
            errs.append(f"{k} must be in (0, 1)")
    if not (isinstance(d["n_null_cal"], int) and d["n_null_cal"] >= 0):
        errs.append("n_null_cal must be a non-negative int")
    if errs or not ok:
        return errs
    la, lb = d["lambda_hat"], d["lambda_hat_alpha05"]
    if d["alpha_sens"] < d["alpha"] and la is None and lb is not None:
        errs.append("lambda_hat is +inf but lambda_hat_alpha05 is finite (alpha 0.05 needs more nulls)")
    if la is not None and lb is not None and lb < la - 1e-12:
        errs.append(f"lambda_hat_alpha05 {lb} < lambda_hat {la} (a smaller alpha must give a higher threshold)")
    for k, a in (("lambda_hat", d["alpha"]), ("lambda_hat_alpha05", d["alpha_sens"])):
        if d[k] is not None and d["n_null_cal"] < n_needed(a):
            errs.append(f"{k} finite with n_null_cal {d['n_null_cal']} < {n_needed(a)} needed at alpha {a}")
    if d["status"] == "FINAL" and not d["bundle_sha256"]:
        errs.append("a FINAL lambda file must name the bundle_sha256 its scores were computed with")
    if d["status"] == "PLACEHOLDER" and any(d[k] is not None for k in ("lambda_hat", "lambda_hat_alpha05",
                                                                      "carry_over_offset")):
        errs.append("a PLACEHOLDER lambda file must hold only nulls")
    return errs


def read_lambda(path):
    """-> normalised dict: lambda_hat / lambda_hat_alpha05 / carry_over_offset as floats (+inf for null; offset None if
    absent), status (LEGACY for old files), sha256 of the file, raw dict. Raises LambdaFormatError."""
    p = Path(path)
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise LambdaFormatError(f"{p}: not JSON ({e})") from None
    errs = validate_lambda(d)
    if errs:
        raise LambdaFormatError(f"{p}: " + "; ".join(errs))
    inf = lambda v: math.inf if v is None else float(v)                                   # noqa: E731
    legacy = "schema" not in d
    return {"status": "LEGACY" if legacy else d["status"], "lambda_hat": inf(d.get("lambda_hat")),
            "lambda_hat_alpha05": inf(d.get("lambda_hat_alpha05", d.get("lambda_hat"))),
            "carry_over_offset": (None if legacy else inf(d.get("carry_over_offset"))),
            "bundle_sha256": d.get("bundle_sha256"), "sha256": sha256_file(p), "raw": d}


def write_lambda(path, rec):
    errs = validate_lambda(rec)
    if errs:
        raise LambdaFormatError("; ".join(errs))
    Path(path).write_text(json.dumps(rec, indent=1) + "\n")
    return Path(path)


# =============================================================================================== bundle
def code_hashes(repo=REPO, files=CODE_FILES):
    return {f: sha256_file(Path(repo) / f) for f in files if (Path(repo) / f).is_file()}


def code_version(hashes):
    return hashlib.sha256("".join(f"{k} {v}\n" for k, v in sorted(hashes.items())).encode()).hexdigest()


def _git_commit(repo=REPO):
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def wheelsets_from_split(p="P1P5P2", alpha1=0.05, K=2):
    """I1 sparse wheel-set calibration scores exactly as SPLIT method (a): structured_ensemble W predictions on devW,
    calA episodes, one OK / one faulted wheel per trial (seed 0). -> (s_ok, s_f). Simulation; AI side only."""
    from ..ideas.i1 import stage1 as S1
    from ..ideas.split import common as C
    E = C.episodes(p)
    calA = C.split_groups(E)["calA"]
    D = S1.load("structured_ensemble", p)
    T = S1.calibrate(D, np.array([E[k]["b"] for k in calA]), alpha1, K=K, seed=0)
    return np.asarray(T["s_ok"], float), np.asarray(T["s_f"], float)


def bb_offset_from_split(p="P1P5P2", alpha=ALPHA):
    """Simulation carry-over offset (SPLIT (b)): conformal quantile of (ensemble q05 of z - true current-body z) over the
    calA episodes. Used only until the real CAL calibration (lambda_v2.json carry_over_offset) exists."""
    from ..ideas.split import common as C
    from . import choose_module as CM
    E = C.episodes(p)
    calA = C.split_groups(E)["calA"]
    Pe = C.load_pred("ensemble", p)
    zs = np.minimum(Pe["ys"][..., 2], Pe["ys"][..., 3])
    qz = np.quantile(zs, 0.05, 1)
    b = np.array([E[k]["b"] for k in calA])
    rho = np.stack([E[k]["rho"] for k in calA])
    return float(CM.cq(qz[b] - np.minimum(rho[:, 2], rho[:, 3]), alpha))


def build_field_bundle(out, models=MODELS_W, structured="structured_ensemble.pt", blackbox="ensemble.pt",
                       library=SPLIT_LIBRARY, wheelsets=None, bb_offset=None, tau=TAU, alpha1=0.05, K=2, rung="E1",
                       pids=("P1", "P5", "P2"), note="", library_source=None, synthetic=False, overwrite=False):
    """Self-contained choice bundle for choose_module (see the module docstring). `wheelsets`: npz {s_ok, s_f} or None
    (computed with wheelsets_from_split); `bb_offset`: float or None (bb_offset_from_split). Returns the folder."""
    from . import choose_module as CM
    out = Path(out)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists (a bundle is written once; pass overwrite=True for a new one)")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.parent / f".{out.name}.stage"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    src = {}
    if wheelsets is None:
        s_ok, s_f = wheelsets_from_split("".join(pids), alpha1, K)
        wheelsets = stage / "_wheelsets_src.npz"
        np.savez(wheelsets, s_ok=s_ok, s_f=s_f)
        src["wheelsets"] = f"wheelsets_from_split({''.join(pids)}, alpha1={alpha1}, K={K}) [SIMULATION calA]"
    else:
        src["wheelsets"] = f"{Path(wheelsets).name} sha256 {sha256_file(wheelsets)}"
    if bb_offset is None:
        bb_offset = bb_offset_from_split("".join(pids))
        src["bb_offset"] = f"bb_offset_from_split({''.join(pids)}) [SIMULATION calA; replaced by lambda_v2 carry_over_offset]"
    else:
        src["bb_offset"] = "given"
    models = Path(models)
    sp = structured if str(structured).startswith("stub:") else models / structured
    bp = blackbox if str(blackbox).startswith("stub:") else models / blackbox
    b_dir = CM.write_bundle(out, sp, bp, wheelsets, library, bb_offset, tau=tau, alpha1=alpha1, K=K, rung=rung,
                            pids=pids, synthetic=synthetic, note=note)
    shutil.rmtree(stage, ignore_errors=True)
    b = yaml.safe_load((b_dir / "bundle.yaml").read_text())
    # code snapshot
    hashes = code_hashes()
    for rel in hashes:
        dst = out / "code" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, dst)
        b["files"][f"code/{rel}"] = sha256_file(dst)
    man = {}
    mf = models / "manifest.json"
    if mf.is_file():
        mj = json.loads(mf.read_text())
        for key, fn in (("structured", structured), ("blackbox", blackbox)):
            stem = Path(str(fn)).stem
            ent = (mj.get("methods") or {}).get(stem)
            if ent:
                man[key] = {k: ent.get(k) for k in ("file", "sha256", "params_sha256", "code_sha256", "created",
                                                    "dataset", "steps")}
        man["manifest_sha256"] = sha256_file(mf)
    lib_sha = sha256_file(library)
    b["provenance"] = {
        "built_utc": _now(), "git_commit": _git_commit(),
        "certificate_code_version": code_version(hashes), "code_sha256": hashes,
        "models_dir": str(models), "models": man,
        "model_sha256": {k: b["files"][v] for k, v in (("structured", b["structured"]), ("blackbox", b["blackbox"]))
                         if v in b["files"]},
        "params_sha256_current": sha256_file(PARAMS) if PARAMS.is_file() else None,
        "library": str(library), "library_sha256": lib_sha,
        "library_source": library_source or ("SPLIT DR_WIDE single-module library (PLACEHOLDER: PROTOCOL_V2 §3 wants "
                                             "library_v2.npz rebuilt at the sys-ID +-20% box)"),
        "wheelsets_source": src["wheelsets"], "bb_offset_source": src["bb_offset"],
        "runtime": "CPU only, offline: python -m capxfer.field.choose_module ... --device cpu",
    }
    b["calibration"] = {"lambda_file": "lambda_v2.PLACEHOLDER.json", "status": "PLACEHOLDER",
                        "fill": "capxfer.field.builders lambda (MASK CAL units only), then hash it into the freeze record"}
    write_lambda(out / "lambda_v2.PLACEHOLDER.json", placeholder_lambda())
    b["files"]["lambda_v2.PLACEHOLDER.json"] = sha256_file(out / "lambda_v2.PLACEHOLDER.json")
    (out / "bundle.yaml").write_text(yaml.safe_dump(b, sort_keys=False))
    write_manifest(out)
    return out


def write_manifest(d):
    d = Path(d)
    lines = [f"{sha256_file(p)}  {p.relative_to(d).as_posix()}" for p in sorted(d.rglob("*"))
             if p.is_file() and p.name != "MANIFEST.sha256"]
    (d / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return d / "MANIFEST.sha256"


def verify_bundle(d, lambda_path=None, load_models=True, repo=REPO):
    """Offline check on the robot laptop. -> dict(ok, errors, warnings, bundle_sha256, code_version). Checks the
    MANIFEST, every hashed file (choose_module.load_bundle), that the INSTALLED certificate code equals the bundle's
    snapshot, that the models load on CPU, and (optional) that lambda_v2.json is valid and made for this bundle."""
    from . import choose_module as CM
    d = Path(d)
    errs, warn = [], []
    mf = d / "MANIFEST.sha256"
    if mf.is_file():
        listed = set()
        for ln in mf.read_text().splitlines():
            if not ln.strip():
                continue
            h, name = ln.split("  ", 1)
            listed.add(name)
            p = d / name
            if not p.is_file():
                errs.append(f"missing {name}")
            elif sha256_file(p) != h:
                errs.append(f"sha256 differs: {name}")
        extra = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()} - listed - {"MANIFEST.sha256"}
        if extra:
            errs.append(f"files not in the manifest: {sorted(extra)[:5]}")
    else:
        warn.append("no MANIFEST.sha256 (not built by capxfer.field.bundle)")
    b = None
    try:
        b = CM.load_bundle(d)
    except SystemExit as e:
        errs.append(str(e))
    prov = (b or {}).get("provenance") or {}
    if b is not None and prov:
        inst = code_hashes(repo, tuple(prov.get("code_sha256", {})))
        diff = sorted(k for k, v in prov.get("code_sha256", {}).items() if inst.get(k) != v)
        if diff:
            errs.append(f"installed certificate code differs from the bundle snapshot: {diff} (use the frozen checkout)")
    if b is not None and load_models and not b.get("synthetic"):
        try:
            import torch
            CM.Models(b, device="cpu")
            if torch.cuda.is_available():
                warn.append("CUDA visible; choose_module runs with --device cpu anyway")
        except Exception as e:                                                          # noqa: BLE001
            errs.append(f"model load on CPU failed: {type(e).__name__}: {e}")
    if lambda_path is not None and b is not None:
        try:
            lam = read_lambda(lambda_path)
            if lam["status"] != "FINAL":
                warn.append(f"lambda file status {lam['status']} (field use needs FINAL)")
            if lam["bundle_sha256"] and lam["bundle_sha256"] != b["_sha"]:
                errs.append("lambda_v2.json was computed with another bundle (bundle_sha256 differs)")
        except LambdaFormatError as e:
            errs.append(str(e))
    return {"ok": not errs, "errors": errs, "warnings": warn, "bundle_sha256": b["_sha"] if b else None,
            "code_version": prov.get("certificate_code_version"), "synthetic": bool(b and b.get("synthetic"))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("out")
    p.add_argument("--models", default=str(MODELS_W))
    p.add_argument("--structured", default="structured_ensemble.pt")
    p.add_argument("--blackbox", default="ensemble.pt")
    p.add_argument("--library", default=str(SPLIT_LIBRARY))
    p.add_argument("--library-source")
    p.add_argument("--wheelsets")
    p.add_argument("--bb-offset", type=float)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--note", default="")
    p = sub.add_parser("verify")
    p.add_argument("bundle")
    p.add_argument("--lambda", dest="lam")
    p.add_argument("--no-models", action="store_true")
    p = sub.add_parser("check-lambda")
    p.add_argument("file")
    p.add_argument("--bundle")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        import torch
        torch.set_num_threads(4)
        d = build_field_bundle(a.out, a.models, a.structured, a.blackbox, a.library, a.wheelsets, a.bb_offset, a.tau,
                               note=a.note, library_source=a.library_source)
        print(f"bundle {d}\n  bundle.yaml sha256 {sha256_file(d / 'bundle.yaml')} (hash it into PROTOCOL_V2 §12)")
        return 0
    if a.cmd == "verify":
        r = verify_bundle(a.bundle, a.lam, load_models=not a.no_models)
        print(json.dumps(r, indent=1))
        return 0 if r["ok"] else 2
    lam = read_lambda(a.file)
    if a.bundle:
        from . import choose_module as CM
        bs = CM.sha256_file(Path(a.bundle) / "bundle.yaml")
        if lam["bundle_sha256"] and lam["bundle_sha256"] != bs:
            print(f"{a.file}: computed with bundle {lam['bundle_sha256']}, not {bs}")
            return 2
    print(f"{a.file}: valid {lam['status']} lambda_hat={lam['lambda_hat']} alpha05={lam['lambda_hat_alpha05']} "
          f"carry_over_offset={lam['carry_over_offset']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

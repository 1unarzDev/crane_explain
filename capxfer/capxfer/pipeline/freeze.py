"""Freeze manifest: every file the frozen analysis depends on, with sha256 (schema capxfer.freeze/v1).

  python -m capxfer.pipeline.freeze write FREEZE.json --models DIR [--suffix _A] [--methods m1,m2,...]
         --schedule SCHEDULE_DIR [--extra NAME=PATH ...] [--note TEXT]
  python -m capxfer.pipeline.freeze verify FREEZE.json            # exit 1 on any mismatch

Format (JSON, canonical key order):
  {"schema": "capxfer.freeze/v1", "created_utc": ..., "note": ...,
   "roots": {"repo": "/abs/capxfer", "models": "/abs/models", "schedule": "/abs/schedule_dir", "extra": "/"},
   "models": {"structured_npe": "structured_npe_A.pt", ...},          # method -> file under roots.models
   "schedule_sha256": "<sha of schedule.json>",                       # must equal schedule.sha256 / commit.json
   "params_sha256": "<sha of params/module_measured.yaml>",
   "files": [{"root": "repo", "path": "capxfer/sim.py", "sha256": ..., "bytes": ...}, ...],   # sorted by (root, path)
   "aggregate_sha256": sha256 over the lines "<root>/<path> <sha256>\n" in file order}
The aggregate hash is what goes into the pre-specification. The seed of the sealed schedule is deliberately NOT hashed
here (only its commitment in commit.json): it stays sealed until unblinding.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = "capxfer.freeze/v1"
REPO = Path(__file__).resolve().parents[2]
METHODS = ("structured_npe", "structured_ensemble", "blackbox", "ensemble", "analytic_bayes", "nominal_sim")
REPO_GLOBS = ("capxfer/**/*.py", "params/*.yaml", "configs/*", "paper/results_map.json", "paper/fill_results.py",
              "paper/build_paper.sh", "docs/PROTOCOL_DRAFT.md", "docs/BENCHMARK_DESIGN.md", "docs/CONTRACTS.md",
              "docs/DOCKING_PROTOCOL.md")
SCHEDULE_FILES = ("schedule.json", "schedule.sha256", "commit.json")
MODEL_SIDECARS = ("study_constants.yaml", "wheel_claim_contracts_v1.json")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry(root_name, root, rel):
    p = Path(root) / rel
    return dict(root=root_name, path=str(rel), sha256=sha256_file(p), bytes=p.stat().st_size)


def aggregate(files):
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f['root']}/{f['path']} {f['sha256']}\n".encode())
    return h.hexdigest()


def repo_files(repo=REPO):
    out = set()
    for g in REPO_GLOBS:
        for p in repo.glob(g):
            if p.is_file() and "__pycache__" not in p.parts:
                out.add(str(p.relative_to(repo)))
    return sorted(out)


def build(models_dir, schedule_dir, methods=METHODS, suffix="", extra=None, note="", repo=REPO):
    models_dir, schedule_dir = Path(models_dir).resolve(), Path(schedule_dir).resolve()
    files = [_entry("repo", repo, r) for r in repo_files(repo)]
    models = {}
    for m in methods:
        f = f"{m}{suffix}.pt"
        if not (models_dir / f).is_file():
            raise FileNotFoundError(models_dir / f)
        models[m] = f
        files.append(_entry("models", models_dir, f))
    for f in MODEL_SIDECARS:
        if (models_dir / f).is_file():
            files.append(_entry("models", models_dir, f))
    for f in SCHEDULE_FILES:
        if not (schedule_dir / f).is_file():
            raise FileNotFoundError(schedule_dir / f)
        files.append(_entry("schedule", schedule_dir, f))
    for name, p in (extra or {}).items():
        p = Path(p).resolve()
        files.append(dict(root="extra", path=str(p), name=name, sha256=sha256_file(p), bytes=p.stat().st_size))
    files.sort(key=lambda f: (f["root"], f["path"]))
    sched_sha = sha256_file(schedule_dir / "schedule.json")
    sealed = (schedule_dir / "schedule.sha256").read_text().split()[0]
    commit = json.loads((schedule_dir / "commit.json").read_text())["schedule_sha256"]
    if not (sched_sha == sealed == commit):
        raise ValueError(f"schedule hash mismatch: file {sched_sha} sealed {sealed} commit {commit}")
    return dict(schema=SCHEMA, created_utc=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                note=note, roots=dict(repo=str(repo), models=str(models_dir), schedule=str(schedule_dir), extra="/"),
                models=models, model_suffix=suffix, schedule_sha256=sched_sha,
                params_sha256=sha256_file(repo / "params" / "module_measured.yaml"),
                files=files, aggregate_sha256=aggregate(files))


def write(path, **kw):
    man = build(**kw)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    return man


def load(path):
    man = json.loads(Path(path).read_text())
    if man.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} manifest")
    return man


def verify(path_or_man, check_new_code=True):
    """-> list of problems (empty = the frozen inputs are byte-identical). Also reports new repo code files that the
    freeze does not cover (a silently added module would otherwise escape the hash)."""
    man = load(path_or_man) if not isinstance(path_or_man, dict) else path_or_man
    probs = []
    if aggregate(man["files"]) != man["aggregate_sha256"]:
        probs.append("aggregate_sha256 does not match the file list (manifest edited)")
    for f in man["files"]:
        p = Path(man["roots"][f["root"]]) / f["path"]
        if not p.is_file():
            probs.append(f"missing {f['root']}/{f['path']}")
        elif sha256_file(p) != f["sha256"]:
            probs.append(f"changed {f['root']}/{f['path']}")
    if check_new_code:
        have = {f["path"] for f in man["files"] if f["root"] == "repo"}
        for r in repo_files(Path(man["roots"]["repo"])):
            if r not in have:
                probs.append(f"not in freeze: repo/{r}")
    return probs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("out")
    w.add_argument("--models", required=True)
    w.add_argument("--suffix", default="")
    w.add_argument("--methods", default=",".join(METHODS))
    w.add_argument("--schedule", required=True, help="sealed schedule dir (schedule.json, schedule.sha256, commit.json)")
    w.add_argument("--extra", action="append", default=[], help="NAME=PATH, any further frozen input")
    w.add_argument("--note", default="")
    v = sub.add_parser("verify")
    v.add_argument("freeze")
    a = ap.parse_args(argv)
    if a.cmd == "write":
        extra = dict(e.split("=", 1) for e in a.extra)
        man = write(a.out, models_dir=a.models, schedule_dir=a.schedule, methods=a.methods.split(","), suffix=a.suffix,
                    extra=extra, note=a.note)
        print(f"wrote {a.out}: {len(man['files'])} files, aggregate {man['aggregate_sha256']}")
        return 0
    probs = verify(a.freeze)
    for p in probs:
        print("FREEZE MISMATCH:", p)
    print("freeze OK" if not probs else f"{len(probs)} problem(s)")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())

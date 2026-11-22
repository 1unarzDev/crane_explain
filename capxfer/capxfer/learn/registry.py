"""Versioned model generations (integrator M2): ~/.cache/capxfer/C/models/<tag>/<method>.pt + manifest.json.

One directory per training generation, exactly one checkpoint per method, and a manifest recording for each method
the training-data manifest sha256, the params sha256, the WS-C code sha256 and the checkpoint sha256. Legacy flat
files (models/<method><suffix>.pt) are still readable via `resolve`.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(os.path.expanduser("~/.cache/capxfer/C"))
CODE_FILES = ("capxfer/learn/*.py", "capxfer/featurize.py", "capxfer/datagen.py", "capxfer/sensors.py",
              "capxfer/sim.py")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256(repo=Path(__file__).resolve().parents[2]):
    h = hashlib.sha256()
    for pat in CODE_FILES:
        for p in sorted(repo.glob(pat)):
            h.update(p.relative_to(repo).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def tag_name(tag):
    t = str(tag or "").lstrip("_")
    return t or "default"


def model_dir(tag, root=ROOT):
    d = Path(root) / "models" / tag_name(tag)
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path(tag, method, root=ROOT):
    return model_dir(tag, root) / f"{method}.pt"


def resolve(tag, method, root=ROOT):
    """Versioned path if present, else the legacy flat file models/<method><tag>.pt (None if neither exists)."""
    p = Path(root) / "models" / tag_name(tag) / f"{method}.pt"
    if p.exists():
        return p
    legacy = Path(root) / "models" / f"{method}{tag or ''}.pt"
    return legacy if legacy.exists() else None


def record(tag, method, data_manifests=(), extra=None, root=ROOT):
    """Add/replace `method`'s entry in models/<tag>/manifest.json (call after saving the checkpoint)."""
    from . import _stub
    d = model_dir(tag, root)
    mp = d / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else dict(tag=tag_name(tag), methods={})
    data = []
    for p in data_manifests:
        p = Path(p)
        dm = json.loads(p.read_text())
        data.append(dict(path=str(p), sha256=sha256_file(p), name=dm.get("name"), params_sha256=dm.get("params_sha256"),
                         complete=dm.get("complete")))
    ck = d / f"{method}.pt"
    man["methods"][method] = dict(file=ck.name, sha256=sha256_file(ck) if ck.exists() else None,
                                  params_sha256=_stub.params_hash(), code_sha256=code_sha256(), data=data,
                                  created=time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {}))
    mp.write_text(json.dumps(man, indent=1, default=str))
    return man

"""I4 shared helpers: paths, family-filtered feature loading, LOFO family map, free-polyomino canon.

Everything I4 writes lives in ~/.cache/capxfer/I4/ (models under I4/models/<tag>/ via capxfer.learn.registry with
root=I4). capxfer/learn is imported, never edited.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))   # shared-machine policy (integrator)

from capxfer import assembly as A
from capxfer import datagen, featurize
from capxfer.learn import data as D
from capxfer.probes import PIDS, TARGET_KEYS

I4 = Path(os.path.expanduser("~/.cache/capxfer/I4"))
LOGS = Path(os.path.expanduser("~/.cache/capxfer/logs"))
C_MODELS = Path(os.path.expanduser("~/.cache/capxfer/C/models"))
A_ROOT = Path(os.path.expanduser("~/.cache/capxfer/A"))

# physical type -> the shape family label (assembly.shape_family) that contains it and all its rotations/reflections
TYPE_FAMILY = {t: datagen.physical_family(t) for t in A.PHYSICAL_TYPES}
TYPES = tuple(A.PHYSICAL_TYPES)
UNTESTED = np.array([i for i, k in enumerate(TARGET_KEYS) if not (k.startswith("P1") or k.startswith("P5"))])
EVAL_RUNG, EVAL_PIDS = "E2", ("P1", "P5")
ALPHA = 0.1


def feat_from_raw(raw):
    """One RawTrial -> WS-C layout Feat (all pids, E3; views applied later), same as learn.data._load_shard_A."""
    F = D.from_featurize(featurize.featurize(raw, "E3", PIDS))
    lab = raw["labels"]
    F = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in F.items()}
    F["wheel_x"] = F["wheel_x"].half()
    F["body_x"] = F["body_x"].half()
    F["labels"] = dict(mode=lab["mode"].cpu().long(), g=lab["g"].cpu().float(), ood=lab.get("ood", "none"),
                       family=(f"phys_{lab['asm_type']}" if lab.get("asm_type") not in (None, "random")
                               else lab.get("family")), asm=raw["meta"]["assembly"].name)
    for k in ("free_family", "bbox_family", "stress", "rho_twist", "rho_twist_nominal", "size"):
        if k in lab:
            F["labels"][k] = lab[k]
    return F


def load_feats_dir(d, keep=None, max_shards=None, log=print):
    """Featurize every shard of a WS-A-format dataset directory; keep(compact_group) filters BEFORE expansion."""
    d = Path(d)
    if not d.is_absolute() and not d.exists():
        d = A_ROOT / d
    shards = sorted(d.glob("shard_*.pt"))[:max_shards]
    out = []
    for i, p in enumerate(shards):
        for raw in datagen.iter_shard(p, keep=keep):
            out.append(feat_from_raw(raw))
        log(f"  loaded {p.name} ({i + 1}/{len(shards)}), groups so far {len(out)}")
    return out


def family_of_compact(c):
    lab = c["labels"]
    return f"phys_{lab['asm_type']}" if lab.get("asm_type") not in (None, "random") else lab.get("family")


# ----------------------------------------------------------------------------- free polyominoes
def _norm(cells):
    mx, my = min(x for x, _ in cells), min(y for _, y in cells)
    return tuple(sorted((x - mx, y - my) for x, y in cells))


def d4_images(cells):
    """All 8 rotations/reflections of a cell set (translation-normalised)."""
    out = []
    c = list(cells)
    for refl in (False, True):
        cc = [(x, -y) for x, y in c] if refl else c
        for r in range(4):
            out.append(_norm(cc))
            cc = [(-y, x) for x, y in cc]
    return out


def free_canon(cells):
    return min(d4_images(cells))


def free_polyominoes(k):
    """{canonical free shape: [fixed orientations]} for size k."""
    out = {}
    for s in A._polyominoes(k):
        out.setdefault(free_canon(s), set()).add(s)
    return {c: sorted(v) for c, v in sorted(out.items())}


_NAMES = {(4, ((0, 0), (0, 1), (0, 2), (0, 3))): "I4", (4, ((0, 0), (0, 1), (1, 0), (1, 1))): "O4"}


def free_name(cells):
    """Stable family id: k<size>_f<index in sorted canonical order>, plus a readable tetromino tag."""
    k = len(cells)
    canon = free_canon(cells)
    keys = list(free_polyominoes(k))
    return f"k{k}_f{keys.index(canon):02d}"


def json_dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))

"""I3 (probe-until-certified): offline replay table. For every trial of a WS-A dataset and every probe view
(subset of the 7 full primitives, plus subsets using the P2 OUTBOUND leg only = a 2 s partial probe), store the
predictor's decision-relevant quantiles. Policies are then replayed offline (`seq.py`) with no further GPU work.

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.i3.extract --data devW --method ensemble --tag W

Output ~/.cache/capxfer/I3/views_<data>_<method>_<tag>.pt (owned by I3). GPU cap 0.12 (~1.9 GB).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from capxfer import datagen, featurize
from capxfer.learn import data as D
from capxfer.learn import registry as R
from capxfer.learn.base import Predictor, cap_memory
from capxfer.probes import PIDS, TARGET_KEYS

OUT = Path(os.path.expanduser("~/.cache/capxfer/I3"))
DEC = [TARGET_KEYS.index("P2+"), TARGET_KEYS.index("P2-")]
ACTIONS = tuple(PIDS) + ("P2o",)          # P2o = P2 outbound leg only (partial probe)


def all_views():
    """Views as sorted tuples of actions. P2 and P2o are mutually exclusive (P2o is a prefix of P2)."""
    others = [p for p in PIDS if p != "P2"]
    vs = []
    for r in range(len(others) + 1):
        for c in itertools.combinations(others, r):
            vs += [tuple(c), tuple(sorted(c + ("P2",))), tuple(sorted(c + ("P2o",)))]
    return sorted(set(vs), key=lambda v: (len(v), v))


CHAINS = {"C1_strafe_first": ("P2o", "P2", "P6y", "P3", "P4", "P1", "P5", "P6x"),
          "C2_default_first": ("P1", "P5", "P2o", "P2", "P6y", "P3", "P4", "P6x"),
          "C3_diag_first": ("P3", "P4", "P2o", "P2", "P6y", "P1", "P5", "P6x"),
          "C4_P2_first": ("P2", "P6y", "P3", "P4", "P1", "P5", "P6x"),
          "C5_prior_then_P2": ("P2",),
          "C6_prior_P2half_P2": ("P2o", "P2")}
FIXED = {"P1P5": ("P1", "P5"), "P1P5P2": ("P1", "P2", "P5"), "P2": ("P2",), "all": tuple(sorted(PIDS))}


def chain_views(chain):
    """Seen-set after each action of a chain (P2 after P2o replaces it)."""
    seen, out = [], [()]
    for a in chain:
        if a == "P2" and "P2o" in seen:
            seen.remove("P2o")
        seen.append(a)
        out.append(tuple(sorted(seen)))
    return out


def core_views(chains=CHAINS):
    acts = list(PIDS) + ["P2o"]
    vs = {()} | {(a,) for a in acts}
    vs |= {tuple(sorted(c)) for c in itertools.combinations(acts, 2) if set(c) != {"P2", "P2o"}}
    for c in chains.values():
        vs |= set(chain_views(c))
    vs |= {tuple(sorted(v)) for v in FIXED.values()}
    return sorted(vs, key=lambda v: (len(v), v))


def view_feat(F, rung, view):
    pids = tuple(p if p != "P2o" else "P2" for p in view)
    v = D.apply_view(F, rung, pids)
    if "P2o" in view:                       # zero the return leg of the P2 block
        s = D.SEG["P2"]
        half = s.start + D.SEG_LEN["P2"]
        v["wheel_x"][:, :, half:s.stop] = 0
        v["body_x"][:, half:s.stop] = 0
    return v


QL = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99)


@torch.no_grad()
def fast_quantiles(model, feat, S=40, chunk=None):
    """Mixture quantiles (B,14,len(QL)) of a blackbox ensemble: each member's piecewise-linear quantile function
    is sampled at S stratified levels (deterministic), samples pooled, empirical quantiles taken. Same estimand as
    Ensemble.predict's y_q (pooled member samples) without Monte-Carlo noise; also works for a single Blackbox."""
    from capxfer.learn.base import to_dev
    from capxfer.learn.blackbox import LEVELS, interp_quantiles
    members = getattr(model, "members", [model])
    dev = members[0].dev
    B = feat["wheel_x"].shape[0]
    chunk = chunk or max(32, 2048 // feat["wheel_x"].shape[1])
    out = []
    for s0 in range(0, B, chunk):
        sub = {k: (v[s0:s0 + chunk] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B else v)
               for k, v in feat.items() if k != "labels"}
        f = to_dev(D.ensure_batch_fields(sub), dev)
        u = ((torch.arange(S, device=dev) + 0.5) / S).expand(f["wheel_x"].shape[0], 14, S)
        ys = []
        for mbr in members:
            q, _ = mbr._forward(f)
            i, w = interp_quantiles(q, LEVELS, u)
            ys.append(torch.gather(q, 2, i - 1) * (1 - w) + torch.gather(q, 2, i) * w)
        ys = torch.cat(ys, -1)
        out.append(torch.quantile(ys, torch.tensor(QL, device=dev), dim=-1).permute(1, 2, 0).cpu())
    return torch.cat(out)


def group_feat(raw):
    F = D.from_featurize(featurize.featurize(raw, "E3", PIDS))
    lab = raw["labels"]
    F = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in F.items()}
    F["labels"] = dict(mode=lab["mode"].cpu().long(), g=lab["g"].cpu().float(), ood=lab.get("ood", "none"),
                       family=(f"phys_{lab['asm_type']}" if lab.get("asm_type") not in (None, "random")
                               else lab.get("family")), asm=raw["meta"]["assembly"].name)
    oc = lab.get("ood_consequential")
    F["_oodc"] = (torch.as_tensor(oc).cpu().bool() if oc is not None
                  else torch.zeros(F["wheel_x"].shape[0], dtype=torch.bool))
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="devW")
    ap.add_argument("--method", default="ensemble")
    ap.add_argument("--tag", default="W")
    ap.add_argument("--rung", default="E2")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--max-groups", type=int, default=0)
    ap.add_argument("--views", default="core", help="all | core | min | comma list of '+'-joined views")
    a = ap.parse_args()
    torch.set_num_threads(4)
    cap_memory(0.12)
    model = Predictor.load(R.resolve(a.tag, a.method), mem_frac=0.12)
    views = all_views() if a.views == "all" else core_views() if a.views == "core" else [
        tuple(sorted(v)) for v in {tuple(FIXED[k]) for k in FIXED} | set(chain_views(CHAINS["C1_strafe_first"]))
    ] if a.views == "min" else [tuple(sorted(v.split("+"))) if v else () for v in a.views.split(",")]
    mp = Path(os.path.expanduser(f"~/.cache/capxfer/A/{a.data}/manifest.json"))
    m = json.loads(mp.read_text())
    shards = [mp.parent / s["file"] for s in m["shards"]]
    rec = dict(yq=[], y=[], family=[], asm=[], gid=[], n=[], oodc=[], ood=[], mode=[])
    t0, gi = time.time(), 0
    for sp in shards:
        for raw in datagen.iter_shard(sp):
            F = group_feat(raw)
            B = F["wheel_x"].shape[0]
            q = torch.empty(B, len(views), 14, len(QL), dtype=torch.float16)
            CH = max(1, 2048 // B)
            for j0 in range(0, len(views), CH):
                vs = [view_feat(F, a.rung, v) for v in views[j0:j0 + CH]]
                big = D.collate(vs)
                big.pop("labels", None)
                qq = fast_quantiles(model, big)
                q[:, j0:j0 + len(vs)] = qq.reshape(len(vs), B, 14, len(QL)).permute(1, 0, 2, 3).half()
            rec["yq"].append(q)
            rec["y"].append(F["y"].float().cpu())
            rec["family"] += [str(F["labels"]["family"])] * B
            rec["asm"] += [str(F["labels"]["asm"])] * B
            rec["gid"] += [gi] * B
            rec["n"] += [F["wheel_x"].shape[1]] * B
            rec["oodc"].append(F["_oodc"])
            rec["ood"] += [str(F["labels"]["ood"])] * B
            md = F["labels"]["mode"]
            rec["mode"].append(torch.stack([(md > 0).sum(1), (md == 4).sum(1), ((md == 2) | (md == 3)).sum(1)], 1))
            gi += 1
            print(f"group {gi} B={B} n={F['wheel_x'].shape[1]} {time.time() - t0:.0f}s", flush=True)
            if a.max_groups and gi >= a.max_groups:
                break
        if a.max_groups and gi >= a.max_groups:
            break
    out = dict(views=views, q_levels=QL, yq=torch.cat(rec["yq"]), y=torch.cat(rec["y"]), family=rec["family"], asm=rec["asm"],
               gid=np.asarray(rec["gid"]), n=np.asarray(rec["n"]), oodc=torch.cat(rec["oodc"]).numpy(),
               ood=rec["ood"], fault_counts=torch.cat(rec["mode"]).numpy(), data=a.data, method=a.method, tag=a.tag,
               rung=a.rung, source="SIMULATION")
    sfx = (f"_g{a.max_groups}" if a.max_groups else "") + ("" if a.views == "core" else f"_{a.views}")
    torch.save(out, OUT / f"views_{a.data}_{a.method}_{a.tag}{sfx}.pt")
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()

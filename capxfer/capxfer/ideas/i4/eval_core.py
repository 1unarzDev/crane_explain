"""I4 evaluation core: model loading (incl. member-0 single nets), prediction records, coverage / width / diagnosis
metrics, conformal variants (raw, split, split-joint-10, LOFO split, LOFO group ss1/pooled over families), APS.

Primary capability metric (PROTOCOL): joint coverage of the 10 untested legs (all except P1+-, P5+-) at E2 / P1+P5;
per-key marginal coverage over the same 10 legs is reported next to it.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from capxfer.conformal import capability as CC
from capxfer.conformal import wheel_sets as WS
from capxfer.learn import c2_eval as C2
from capxfer.learn.base import Predictor

from .common import ALPHA, C_MODELS, EVAL_PIDS, EVAL_RUNG, I4, UNTESTED, load_feats_dir

MEM_FRAC = 0.08
MERGE = np.array([0, 1, 2, 2, 4])


def load_model(spec):
    """spec 'W:structured_ensemble' | 'I4_lofo_single:ensemble' | 'W:structured_ensemble#0' (member 0)."""
    gen, meth = spec.split(":")
    member = None
    if "#" in meth:
        meth, member = meth.split("#")
    root = C_MODELS if gen in ("W", "A") else I4 / "models"
    m = Predictor.load(root / gen / f"{meth}.pt", mem_frac=MEM_FRAC)
    if member is not None:
        m = m.members[int(member)]
    return m


def predict(model, feats, set_name, nmax=None, rung=EVAL_RUNG, pids=EVAL_PIDS):
    nmax = nmax or max(F["wheel_x"].shape[1] for F in feats)
    rec, rows = C2.predict_set(model, feats, rung, pids, set_name, nmax)
    rec["fault_recall_cb"] = np.concatenate([r["fault_recall_cb"] for r in rows]) if "fault_recall_cb" in rows[0] \
        else np.full(len(rec["y"]), np.nan)
    for k in ("stress", "free_family"):
        vals = []
        for F in feats:
            v = F["labels"].get(k)
            vals += [v] * F["wheel_x"].shape[0]
        rec[k] = vals
    return rec


def subset(rec, sel):
    sel = np.asarray(sel, bool)
    out = {}
    for k, v in rec.items():
        if torch.is_tensor(v) and v.shape[:1] == (len(sel),):
            out[k] = v[torch.as_tensor(sel)]
        elif isinstance(v, np.ndarray) and v.shape[:1] == (len(sel),):
            out[k] = v[sel]
        elif isinstance(v, list) and len(v) == len(sel):
            out[k] = [x for x, s in zip(v, sel) if s]
        else:
            out[k] = v
    return out


def concat(recs):
    out = {}
    for k in recs[0]:
        v = [r[k] for r in recs]
        if torch.is_tensor(v[0]):
            nm = max(x.shape[1] for x in v) if v[0].dim() >= 2 and k in ("mode_prob", "g_mean", "g_std", "mode_true",
                                                                           "g_true", "wheel_mask") else None
            if nm:
                fill = {"mode_true": -1, "wheel_mask": False}.get(k, float("nan"))
                v = [C2._pad(x, x.shape[1], fill, nm) if x.shape[1] < nm else x for x in v]
            out[k] = torch.cat(v)
        elif isinstance(v[0], np.ndarray):
            out[k] = np.concatenate(v)
        elif isinstance(v[0], list):
            out[k] = sum(v, [])
        else:
            out[k] = v[0]
    return out


def bounds(rec):
    yq = rec["y_q"].numpy()
    return yq[..., 0], yq[..., -1]


def cov_stats(lo, hi, y, legs=UNTESTED):
    lo, hi, y = lo[:, legs], hi[:, legs], y[:, legs]
    c = (y >= lo) & (y <= hi)
    w = hi - lo
    return dict(marg=float(c.mean()), joint=float(c.all(1).mean()),
                width=float(np.median(np.where(np.isfinite(w), w, np.nan).mean(1))) if np.isfinite(w).any() else np.inf,
                frac_inf=float((~np.isfinite(w)).any(1).mean()), n=int(len(y)))


def diag_stats(rec):
    mp = rec["mode_prob"].numpy()
    if np.isnan(mp[rec["wheel_mask"].numpy()]).all():
        return {}
    lab = rec["mode_true"].numpy()
    m = rec["wheel_mask"].numpy() & (lab >= 0)
    p = np.nan_to_num(mp)
    pc = p.copy()
    pc[..., 2] += pc[..., 3]
    pc[..., 3] = 0
    am = pc.argmax(-1)
    lc = MERGE[np.clip(lab, 0, 4)]
    ok = (am == lc) | ~m
    fault = (lab > 0) & m
    ft = fault.any(1)
    return dict(diag_trial=float(ok[ft].all(1).mean()) if ft.any() else None,
                wheel_recall_cb=float(ok[fault].mean()) if fault.any() else None,
                healthy_false_alarm=float((~ok[~ft]).any(1).mean()) if (~ft).any() else None, n_faulted=int(ft.sum()))


def conformal(cal, tests, alpha=ALPHA, legs=UNTESTED, groups=None):
    """Calibrate on rec `cal` and evaluate each test rec; returns {test: {variant: stats}}.
    variants: raw, split (per-key), split_joint (joint over legs); + group_ss1/_pooled(_joint) if groups given."""
    pc = dict(y_q=cal["y_q"].numpy())
    yc = cal["y"].numpy()
    cals = {"split": CC.calibrate(pc, yc, alpha, method="split", targets=legs),
            "split_joint": CC.calibrate(pc, yc, alpha, method="split", joint=True, targets=legs)}
    if groups is not None:
        g = np.asarray(cal[groups])
        for v in ("ss1", "pooled"):
            cals[f"group_{v}"] = CC.calibrate(pc, yc, alpha, groups=g, method="group", group_variant=v, targets=legs)
            cals[f"group_{v}_joint"] = CC.calibrate(pc, yc, alpha, groups=g, method="group", group_variant=v,
                                                    joint=True, targets=legs)
    out = {}
    for tn, t in tests.items():
        y = t["y"].numpy()
        r = {"raw": cov_stats(*bounds(t), y, legs)}
        for cn, c in cals.items():
            lo, hi = (np.asarray(a) for a in c.interval(dict(y_q=t["y_q"].numpy())))
            r[cn] = cov_stats(lo, hi, y, legs)
        out[tn] = r
    return out


def aps(cal, tests, alpha=ALPHA):
    if np.isnan(cal["mode_prob"].numpy()[cal["wheel_mask"].numpy()]).all():
        return {}

    def pred_of(r):
        return dict(mode_prob=torch.nan_to_num(r["mode_prob"]).numpy(), wheel_mask=r["wheel_mask"].numpy())
    lab_c = np.where(cal["mode_true"].numpy() < 0, 0, cal["mode_true"].numpy())
    sc = WS.calibrate(pred_of(cal), lab_c, alpha, pooling="joint")
    out = {}
    for tn, t in tests.items():
        sets = np.asarray(sc.sets(pred_of(t)))
        lab = t["mode_true"].numpy()
        m = t["wheel_mask"].numpy()
        hit = np.take_along_axis(sets, np.clip(lab, 0, 4)[..., None], -1)[..., 0] | ~m
        out[tn] = dict(trial_cover=float(hit.all(1).mean()), set_size=float(sets[m].sum(-1).mean()))
    return out


# ----------------------------------------------------------------------------- devW feature cache
def cached_feats(name, src, log=print):
    """Per-shard feature cache I4/fcache_<name>/part_*.pt (low peak RAM while building); loaded memory-mapped."""
    from capxfer import datagen
    d = I4 / f"fcache_{name}"
    src = Path(src)
    shards = sorted(src.glob("shard_*.pt"))
    if not (d / "complete").exists():
        d.mkdir(parents=True, exist_ok=True)
        from .common import feat_from_raw
        for p in shards:
            q = d / f"part_{p.stem}.pt"
            if not q.exists():
                torch.save([feat_from_raw(r) for r in datagen.iter_shard(p)], q)
                log(f"  cached {name}/{p.name}")
        (d / "complete").write_text("ok")
    return [F for q in sorted(d.glob("part_*.pt")) for F in torch.load(q, weights_only=False, mmap=True)]


def devw_feats(log=print):
    """devW split (same as c2_eval.c2_splits): random groups (with size) + physical types; cached in I4."""
    from .common import A_ROOT
    allf = cached_feats("devW", A_ROOT / "devW", log)
    return dict(random=[F for F in allf if not str(F["labels"]["family"]).startswith("phys_")],
                physical=[F for F in allf if str(F["labels"]["family"]).startswith("phys_")
                          and F["labels"].get("ood", "none") == "none"])


def size_of(fam):
    fam = str(fam)
    return int(fam.split("_")[0][1:]) if fam.startswith("k") else None


def bench_feats(name="bench_v1", log=print):
    """Benchmark features (family = free polyomino id), cached in I4 and memory-mapped on load."""
    return cached_feats(name, I4 / name, log)


if __name__ == "__main__":
    devw_feats()
    bench_feats()


# ----------------------------------------------------------------------------- one pass: ensemble + member 0
class _Replay:
    def __init__(self, preds, cfg):
        self.it, self.cfg = iter(preds), cfg

    def predict(self, v, n_samples=256):
        return next(self.it)


def _compose(ps, q_levels, n_samples=256):
    from capxfer.learn.base import quantiles_from_samples
    M = len(ps)
    per = max(1, -(-n_samples // M))
    ys = torch.cat([p["y_samples"][:, :per] for p in ps], 1)[:, :max(n_samples, M)]
    ds = torch.cat([p["drift_samples"][:, :per] for p in ps], 1)[:, :max(n_samples, M)]
    mp = torch.stack([p["mode_prob"] for p in ps]).mean(0) if ps[0]["mode_prob"] is not None else None
    gm = torch.stack([p["g_mean"] for p in ps])
    return dict(mode_prob=mp, g_mean=gm.mean(0),
                g_std=torch.sqrt((torch.stack([p["g_std"] for p in ps]) ** 2).mean(0) + gm.var(0)),
                y_samples=ys, y_q=quantiles_from_samples(ys, q_levels), drift_samples=ds,
                drift_q=quantiles_from_samples(ds, q_levels), ppc_score=torch.stack([p["ppc_score"] for p in ps]).mean(0))


@torch.no_grad()
def predict_ens_and_m0(ens, feats, set_name, nmax=None, rung=EVAL_RUNG, pids=EVAL_PIDS):
    """Ensemble record and member-0 (single network) record from ONE pass over the members (identical in
    distribution to Ensemble.predict: each member contributes ceil(256/M) of the pooled samples)."""
    from capxfer.learn import data as D
    nmax = nmax or max(F["wheel_x"].shape[1] for F in feats)
    P_ens, P_m0 = [], []
    for F in feats:
        v = D.apply_view(F, rung, pids)
        ps = [m.predict(v, 256) for m in ens.members]
        P_m0.append(ps[0])
        P_ens.append(_compose(ps, ens.cfg["q_levels"]))
    out = []
    for P, cfg in ((P_ens, ens.cfg), (P_m0, ens.members[0].cfg)):
        rec, rows = C2.predict_set(_Replay(P, cfg), feats, rung, pids, set_name, nmax)
        rec["fault_recall_cb"] = np.concatenate([r["fault_recall_cb"] for r in rows]) if "fault_recall_cb" in rows[0] \
            else np.full(len(rec["y"]), np.nan)
        for k in ("stress", "free_family"):
            rec[k] = [x for F in feats for x in [F["labels"].get(k)] * F["wheel_x"].shape[0]]
        out.append(rec)
    return out

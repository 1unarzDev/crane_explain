"""WS-C2: probe ladder, sensing ladder and calibrated (conformal) comparison on the WS-A dev_v2 split.

  PYTHONPATH=. .venv/bin/python -m capxfer.learn.c2_eval --methods structured_npe,blackbox      # predict + score
  PYTHONPATH=. .venv/bin/python -m capxfer.learn.c2_eval --merge                                  # tables -> json/md

Splits (dev_v2, same order as train.a_splits): random assemblies of size <= 3 are split into `val` (small[::3], the
early-stopping set used by train.py) and `cal` (the other 2/3, never seen in training or model selection);
`heldout` = random sizes 4-6; `physical` = the four physical types (ood none). Models are the trainA checkpoints
(`models/<method>_A.pt`), trained with random probe-subset masks.

Per (method, rung, pids) all per-trial predictions are saved to ~/.cache/capxfer/C/preds_<method>_<pidtag>[_<rung>].pt
(rung suffix omitted for E1) so WS-D can recalibrate. Metrics: raw (as train.evaluate) plus split-conformal CQR
calibrated on `cal` (sizes 1-3) and evaluated on `heldout` (4-6), group conformal across shape families (Dunn et al.
ss1 / pooled), joint-14 split CQR, and APS wheel-mode sets (joint pooling) for methods with mode_prob.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from capxfer.conformal import capability as CC
from capxfer.conformal import wheel_sets as WS
from . import data as D
from . import metrics as Mx

ROOT = Path(os.path.expanduser("~/.cache/capxfer/C"))
METHODS = ("structured_npe", "structured_ensemble", "blackbox", "ensemble", "nominal_sim", "analytic_bayes")
LADDER = {"P1P5": ("P1", "P5"), "P1P5P2": ("P1", "P5", "P2"), "P1-P5": ("P1", "P2", "P3", "P4", "P5"),
          "all": tuple(D.PIDS)}
CONFIGS = [("E1", t) for t in LADDER] + [("E0", "P1P5P2"), ("E2", "P1P5P2")]
SETS = ("cal", "heldout", "physical")
ALPHA = 0.1
S_SAVE = 64


def pid_tag_file(rung, tag):
    return f"{tag}" if rung == "E1" else f"{tag}_{rung}"


def _size(F):
    fam = str(F["labels"]["family"] or "")
    return int(fam.split("_")[0][1:]) if fam.startswith("k") else None


def c2_splits(name="dev_v2", cache=True):
    cp = ROOT / f"c2_feats_{name}.pt"
    if cache and cp.exists():
        return torch.load(cp, weights_only=False)
    D.SOURCE = "A_featurize"
    allf = D.load_feats(name)
    rnd = [F for F in allf if not str(F["labels"]["family"]).startswith("phys_")]
    small = [F for F in rnd if (_size(F) or 99) <= 3]
    sp = dict(cal=[F for i, F in enumerate(small) if i % 3],
              heldout=[F for F in rnd if (_size(F) or 0) >= 4],
              physical=[F for F in allf if str(F["labels"]["family"]).startswith("phys_")
                        and F["labels"].get("ood", "none") == "none"])
    if cache:
        torch.save(sp, cp)
    return sp


MEM_FRAC = 0.1
MODEL_TAG = "A"


def load_model(meth, device=None, tag=None):
    from . import registry as R
    from .base import Predictor
    p = R.resolve(tag or MODEL_TAG, meth) or R.resolve("_" + (tag or MODEL_TAG), meth)
    return Predictor.load(p, device=device, mem_frac=MEM_FRAC)


def _pad(x, n, fill, nmax):
    out = torch.full((x.shape[0], nmax) + tuple(x.shape[2:]), fill, dtype=x.dtype)
    out[:, :n] = x[:, :n]
    return out


def predict_set(model, feats, rung, pids, set_name, nmax, n_samples=256):
    """-> (records dict of concatenated per-trial tensors/lists, list of trial_metrics rows)."""
    rec = {k: [] for k in ("y", "drift", "y_q", "drift_q", "y_samples", "ppc_score", "mode_prob", "g_mean",
                           "g_std", "mode_true", "g_true", "wheel_mask", "n_wheels")}
    meta = {k: [] for k in ("family", "asm", "group_id", "set", "size", "trial_id")}
    rows = []
    for gi, F in enumerate(feats):
        v = D.apply_view(F, rung, pids)
        p = model.predict(v, n_samples)
        tm = Mx.trial_metrics(p, v, model.cfg["q_levels"])
        rows.append(tm)
        B, n = v["labels"]["mode"].shape
        rec["y"].append(v["y"].float().cpu())
        rec["drift"].append(torch.nan_to_num(v["drift"].float().cpu()))
        rec["y_q"].append(p["y_q"].float())
        rec["drift_q"].append(p["drift_q"].float())
        rec["y_samples"].append(p["y_samples"][:, :S_SAVE].half())
        rec["ppc_score"].append(p["ppc_score"].float())
        mp = p["mode_prob"] if p["mode_prob"] is not None else torch.full((B, n, 5), float("nan"))
        rec["mode_prob"].append(_pad(mp.float(), n, float("nan"), nmax))
        rec["g_mean"].append(_pad(p["g_mean"].float().reshape(B, -1), n, float("nan"), nmax))
        rec["g_std"].append(_pad(p["g_std"].float().reshape(B, -1), n, float("nan"), nmax))
        rec["mode_true"].append(_pad(v["labels"]["mode"].long(), n, -1, nmax))
        rec["g_true"].append(_pad(v["labels"]["g"].float(), n, float("nan"), nmax))
        wm = torch.zeros(B, nmax, dtype=torch.bool)
        wm[:, :n] = True
        rec["wheel_mask"].append(wm)
        rec["n_wheels"].append(torch.full((B,), n))
        fam = str(F["labels"]["family"])
        meta["family"] += [fam] * B
        meta["asm"] += [str(F["labels"].get("asm"))] * B
        meta["group_id"] += [f"{set_name}/{gi}"] * B
        meta["set"] += [set_name] * B
        meta["size"] += [n if _size(F) is None else _size(F)] * B
        meta["trial_id"] += [f"{set_name}/{gi}/{fam}/{b}" for b in range(B)]
        tm["consequential"] = np.asarray(tm["consequential"])
    out = {k: torch.cat(v) for k, v in rec.items()}
    out.update(meta)
    out["consequential"] = np.concatenate([r["consequential"] for r in rows])
    out["healthy_trial"] = np.concatenate([r["healthy_trial"] for r in rows])
    return out, rows


def raw_tables(rows_by_set, recs):
    res = {}
    for s, rows in rows_by_set.items():
        res[f"{s}/all"] = Mx.aggregate(rows)
        res[f"{s}/all/consequential"] = Mx.aggregate(rows, ("consequential", True))
        if s == "heldout":
            for k in (4, 5, 6):
                sel = [r for r, F in zip(rows, recs["_groups"][s]) if F == k]
                if sel:
                    res[f"{s}/k{k}"] = Mx.aggregate(sel)
        if s == "physical":
            for t in ("single", "pair", "L3", "square4"):
                sel = [r for r, F in zip(rows, recs["_groups_fam"][s]) if F == f"phys_{t}"]
                if sel:
                    res[f"{s}/{t}"] = Mx.aggregate(sel)
                    res[f"{s}/{t}/consequential"] = Mx.aggregate(sel, ("consequential", True))
    return res


def _cov_width(lo, hi, y, sel):
    lo, hi, y = np.asarray(lo)[sel], np.asarray(hi)[sel], np.asarray(y)[sel]
    c = (y >= lo) & (y <= hi)
    return dict(cover90=float(c.mean()) if c.size else None, width90=float((hi - lo).mean()) if c.size else None,
                joint_cover=float(c.all(1).mean()) if c.size else None, n_trials=int(sel.sum()))


def conformal_tables(cal, test_sets, alpha=ALPHA):
    """cal/test: record dicts from predict_set. CQR on y_q (5/95 % levels)."""
    pc = dict(y_q=cal["y_q"].numpy())
    yc = cal["y"].numpy()
    fam_c = np.asarray(cal["family"])
    cals = {"raw": None,
            "split": CC.calibrate(pc, yc, alpha, method="split"),
            "group_ss1": CC.calibrate(pc, yc, alpha, groups=fam_c, method="group", group_variant="ss1"),
            "group_pooled": CC.calibrate(pc, yc, alpha, groups=fam_c, method="group", group_variant="pooled"),
            "split_joint": CC.calibrate(pc, yc, alpha, method="split", joint=True)}
    asm_c = np.asarray(cal["group_id"])
    for v in ("ss1", "pooled"):      # groups = individual assemblies (exchangeable random draws; K = #cal assemblies)
        cals[f"group_asm_{v}"] = CC.calibrate(pc, yc, alpha, groups=asm_c, method="group", group_variant=v)
    res = {"n_cal": int(len(yc)), "n_cal_families": int(len(np.unique(fam_c))),
           "n_cal_assemblies": int(len(np.unique(asm_c))),
           "thresholds": {k: (None if c is None else np.asarray(c.threshold, float).tolist()) for k, c in cals.items()}}
    for sname, t in test_sets.items():
        pt = dict(y_q=t["y_q"].numpy())
        y = t["y"].numpy()
        size = np.asarray(t["size"])
        fam = np.asarray(t["family"])
        subsets = {"all": np.ones(len(y), bool), "consequential": np.asarray(t["consequential"], bool)}
        if sname == "heldout":
            subsets.update({f"k{k}": size == k for k in (4, 5, 6)})
        if sname == "physical":
            subsets.update({ty: fam == f"phys_{ty}" for ty in ("single", "pair", "L3", "square4")})
        for cname, c in cals.items():
            if c is None:
                lo, hi = pt["y_q"][..., 0], pt["y_q"][..., -1]
            else:
                lo, hi = (np.asarray(a) for a in c.interval(pt))
            for ss, sel in subsets.items():
                res[f"{sname}/{cname}/{ss}"] = _cov_width(lo, hi, y, sel)
    # APS wheel-mode sets (joint pooling: every wheel of the trial covered)
    if not torch.isnan(cal["mode_prob"][cal["wheel_mask"]]).any():
        def pred_of(r):
            return dict(mode_prob=torch.nan_to_num(r["mode_prob"]).numpy(), wheel_mask=r["wheel_mask"].numpy())
        lab_c = np.where(cal["mode_true"].numpy() < 0, 0, cal["mode_true"].numpy())
        sc = WS.calibrate(pred_of(cal), lab_c, alpha, pooling="joint")
        res["aps_threshold"] = float(np.asarray(sc.threshold))
        for sname, t in test_sets.items():
            sets = np.asarray(sc.sets(pred_of(t)))
            lab = t["mode_true"].numpy()
            m = t["wheel_mask"].numpy()
            hit = np.take_along_axis(sets, np.clip(lab, 0, 4)[..., None], -1)[..., 0] | ~m
            fault = (lab > 0) & m
            subsets = {"all": np.ones(len(lab), bool), "consequential": np.asarray(t["consequential"], bool)}
            for ss, sel in subsets.items():
                if not sel.any():
                    continue
                res[f"{sname}/aps/{ss}"] = dict(
                    trial_cover=float(hit[sel].all(1).mean()), wheel_cover=float(hit[sel][m[sel]].mean()),
                    faulted_wheel_cover=float(hit[sel][fault[sel]].mean()) if fault[sel].any() else None,
                    set_size=float(sets[sel][m[sel]].sum(-1).mean()),
                    singleton=float((sets[sel][m[sel]].sum(-1) == 1).mean()), n_trials=int(sel.sum()))
    return res


def lofo_tables(cal, held, alpha=ALPHA):
    """Leave-one-family-out group conformal over the random shape families: for each held-out family f (sizes 4-6),
    calibrate on all trials of the OTHER random families (cal sizes 1-3 + the other held-out families) and test on f.
    group_* treat shape families as the exchangeable groups (Dunn et al.); split pools them."""
    yq = np.concatenate([cal["y_q"].numpy(), held["y_q"].numpy()])
    y = np.concatenate([cal["y"].numpy(), held["y"].numpy()])
    fam = np.asarray(list(cal["family"]) + list(held["family"]))
    cons = np.concatenate([np.asarray(cal["consequential"], bool), np.asarray(held["consequential"], bool)])
    is_test = np.r_[np.zeros(len(cal["y"]), bool), np.ones(len(held["y"]), bool)]
    out = {}
    for meth, kw in (("split", dict(method="split")), ("group_ss1", dict(method="group", group_variant="ss1")),
                     ("group_pooled", dict(method="group", group_variant="pooled"))):
        cov, wid, cons_cov, fams = [], [], [], {}
        for f in np.unique(fam[is_test]):
            tr, te = fam != f, is_test & (fam == f)
            c = CC.calibrate(dict(y_q=yq[tr]), y[tr], alpha, groups=fam[tr] if meth != "split" else None, **kw)
            lo, hi = (np.asarray(a) for a in c.interval(dict(y_q=yq[te])))
            ci = ((y[te] >= lo) & (y[te] <= hi)).mean(1)
            cov.append(ci)
            wid.append((hi - lo).mean(1))
            cons_cov.append(ci[cons[te]])
            fams[str(f)] = float(ci.mean())
        cv, wd, cc = np.concatenate(cov), np.concatenate(wid), np.concatenate(cons_cov)
        out[meth] = dict(cover90=float(cv.mean()), width90=float(np.nanmean(np.where(np.isfinite(wd), wd, np.nan))),
                         frac_infinite=float((~np.isfinite(wd)).mean()), consequential_cover90=float(cc.mean()),
                         min_family_cover=min(fams.values()), per_family=fams, n_groups=int(len(np.unique(fam)) - 1))
    return out


def run_method(meth, sp, configs=CONFIGS, save_preds=True, model=None, out_root=ROOT, suffix="", dataset="dev_v2"):
    """suffix: "" for the trainA generation, e.g. "_W" for trainW (file names results_C2<suffix>_<m>.json,
    preds_<m>_<pids><suffix>.pt)."""
    model = model or load_model(meth)
    out = dict(method=meth, configs={})
    for rung, tag in configs:
        pids = LADDER[tag]
        t0 = time.time()
        recs, rows_by_set = {}, {}
        groups = {"_groups": {}, "_groups_fam": {}}
        nmax = max(F["wheel_x"].shape[1] for s in SETS for F in sp[s])
        for s in SETS:
            recs[s], rows_by_set[s] = predict_set(model, sp[s], rung, pids, s, nmax)
            groups["_groups"][s] = [_size(F) for F in sp[s]]
            groups["_groups_fam"][s] = [str(F["labels"]["family"]) for F in sp[s]]
        res = dict(raw=raw_tables(rows_by_set, groups),
                   conformal=conformal_tables(recs["cal"], {s: recs[s] for s in ("heldout", "physical")}),
                   lofo=lofo_tables(recs["cal"], recs["heldout"]),
                   wall_s=time.time() - t0)
        out["configs"][f"{rung}/{tag}"] = res
        if save_preds:
            merged = {}
            for k in recs["cal"]:
                vals = [recs[s][k] for s in SETS]
                merged[k] = torch.cat(vals) if torch.is_tensor(vals[0]) else (
                    np.concatenate(vals) if isinstance(vals[0], np.ndarray) else sum(vals, []))
            merged.update(method=meth, rung=rung, pids=pids, q_levels=list(model.cfg["q_levels"]), source="SIMULATION",
                          dataset=dataset, model_tag=MODEL_TAG, params_sha256=_params_sha(), note=(
                              "sets: cal = dev_v2 random sizes<=3 not used for early stopping; heldout = sizes 4-6; "
                              "physical = four physical types. y_samples truncated to 64 (fp16). Wheel arrays padded "
                              "to the max wheel count (wheel_mask). mode_prob NaN for black-box methods."))
            torch.save(merged, out_root / f"preds_{meth}_{pid_tag_file(rung, tag)}{suffix}.pt")
        print(meth, rung, tag, f"{res['wall_s']:.0f}s", json.dumps(
            {k: res["raw"][k].get(c) for k in ("heldout/all",) for c in ("cover90", "fault_recall")}), flush=True)
    (out_root / f"results_C2{suffix}_{meth}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def _params_sha():
    from . import _stub
    return _stub.params_hash()


# ------------------------------------------------------------------------------------------------ merge / tables
def fmt(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def merge(out_root=ROOT, suffix=""):
    allres = {m: json.loads((out_root / f"results_C2{suffix}_{m}.json").read_text())
              for m in METHODS if (out_root / f"results_C2{suffix}_{m}.json").exists()}
    lad, conf = {}, {}
    md = []
    md.append("### C2 probe ladder (E1; held-out sizes 4-6 / consequential; raw 90 % quantiles)\n")
    md.append("| method | pids | mode_acc | fault_recall | recall_cb | y_mae | cover90 | width90 | conseq recall_cb | "
              "conseq cover90 | conseq width90 | phys cover90 | phys recall_cb |")
    md.append("|" + "---|" * 13)
    for m, r in allres.items():
        for cfg, res in r["configs"].items():
            raw = res["raw"]
            h, hc, ph = raw["heldout/all"], raw["heldout/all/consequential"], raw["physical/all"]
            lad[f"{m}/{cfg}"] = dict(heldout=h, heldout_consequential=hc, physical=ph,
                                     physical_types={t: raw.get(f"physical/{t}") for t in
                                                     ("single", "pair", "L3", "square4")},
                                     physical_consequential=raw["physical/all/consequential"],
                                     k={k: raw.get(f"heldout/k{k}") for k in (4, 5, 6)})
            conf[f"{m}/{cfg}"] = {**res["conformal"], "lofo": res.get("lofo")}
            md.append(f"| {m} | {cfg} | {fmt(h.get('mode_acc'))} | {fmt(h.get('fault_recall'))} | "
                      f"{fmt(h.get('fault_recall_cb'))} | {fmt(h['y_mae'])} | {fmt(h['cover90'])} | {fmt(h['width90'])} | "
                      f"{fmt(hc.get('fault_recall_cb'))} | {fmt(hc['cover90'])} | {fmt(hc['width90'])} | "
                      f"{fmt(ph['cover90'])} | {fmt(ph.get('fault_recall_cb'))} |")
    md.append("\n### C2 calibrated comparison (cal = in-dist sizes 1-3, test = held-out sizes 4-6; alpha 0.1, CQR)\n")
    md.append("| method | cfg | raw cov/width | split cov/width | split k6 cov | split conseq cov/width | "
              "group_asm_ss1 cov/width | LOFO split cov/width | LOFO group_pooled cov/width (inf frac) | joint14 cov | "
              "APS trial cov / set size |")
    md.append("|" + "---|" * 11)
    for key, c in conf.items():
        m, cfg = key.split("/", 1)
        def cw(k):
            x = c.get(k)
            return "–" if x is None else f"{fmt(x['cover90'])}/{fmt(x['width90'])}"
        aps = c.get("heldout/aps/all")
        md.append(f"| {m} | {cfg} | {cw('heldout/raw/all')} | {cw('heldout/split/all')} | "
                  f"{fmt(c['heldout/split/k6']['cover90'])} | {cw('heldout/split/consequential')} | "
                  f"{cw('heldout/group_asm_ss1/all')} | "
                  f"{fmt(c['lofo']['split']['cover90'])}/{fmt(c['lofo']['split']['width90'])} | "
                  f"{fmt(c['lofo']['group_pooled']['cover90'])}/{fmt(c['lofo']['group_pooled']['width90'])} "
                  f"({fmt(c['lofo']['group_pooled']['frac_infinite'], 2)}) | "
                  f"{fmt(c['heldout/split_joint/all']['joint_cover'])} | "
                  + ("–" if aps is None else f"{fmt(aps['trial_cover'])} / {fmt(aps['set_size'], 2)}") + " |")
    (out_root / f"results_C2{suffix}_ladder.json").write_text(json.dumps(lad, indent=1, default=float))
    (out_root / f"results_C2{suffix}_conformal.json").write_text(json.dumps(conf, indent=1, default=float))
    (out_root / f"report_C2{suffix}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return lad, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="structured_npe,blackbox,ensemble")
    ap.add_argument("--configs", default="", help="comma list like E1/P1P5,E0/P1P5P2 (default: all)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--tag", default="A", help="model generation (models/<tag>/); outputs get suffix _<tag> unless A")
    ap.add_argument("--data", default="dev_v2", help="evaluation dataset (WS-A name)")
    ap.add_argument("--mem-frac", type=float, default=0.1)
    a = ap.parse_args()
    global MEM_FRAC, MODEL_TAG
    MEM_FRAC, MODEL_TAG = a.mem_frac, a.tag
    suffix = "" if a.tag == "A" and a.data == "dev_v2" else f"_{a.tag}" + ("" if a.data == "dev_v2" else f"_{a.data}")
    if a.merge:
        merge(suffix=suffix)
        return
    from .base import cap_memory
    cap_memory(a.mem_frac)
    cfgs = [tuple(c.split("/")) for c in a.configs.split(",") if c] or CONFIGS
    sp = c2_splits(a.data)
    print({k: (len(v), sum(F["wheel_x"].shape[0] for F in v)) for k, v in sp.items()}, flush=True)
    for m in a.methods.split(","):
        run_method(m, sp, cfgs, suffix=suffix, dataset=a.data)


if __name__ == "__main__":
    main()

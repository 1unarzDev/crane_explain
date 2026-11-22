"""WS-C2 tests: structured_ensemble API (mixture posterior), probe-ladder / conformal evaluation pipeline on stub data.
Each test < 60 s."""
import json

import numpy as np
import pytest
import torch

from capxfer import assembly as A
from capxfer.learn import _stub as S
from capxfer.learn import c2_eval as C
from capxfer.learn import data as D
from capxfer.learn.blackbox import Blackbox
from capxfer.learn.ensemble import StructuredEnsemble

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def splits():
    P = S.load_params()
    rng = np.random.default_rng(0)
    sp = {"cal": [], "heldout": [], "physical": []}
    for i in range(3):                       # 3 assemblies, relabelled into 6 cal "families" (cheap sim)
        a = A.random_assembly(rng, P, sizes=(1 + i % 2,))
        F = S.simulate_feat(a, P, 16, rng, device=DEV)
        for h in range(2):
            G = {k: (v[8 * h:8 * h + 8] if torch.is_tensor(v) and v.dim() and v.shape[0] == 16 else v)
                 for k, v in F.items() if k != "labels"}
            G["labels"] = {k: (v[8 * h:8 * h + 8] if torch.is_tensor(v) else v) for k, v in F["labels"].items()}
            G["labels"]["family"] = f"k{1 + i % 2}_f{2 * i + h}"
            sp["cal"].append(G)
    for i in range(2):
        n = 4 + i
        a = A.random_assembly(rng, P, sizes=(n,))
        a.meta["family"] = f"k{n}_h{i}"
        sp["heldout"].append(S.simulate_feat(a, P, 8, rng, device=DEV))
    G = dict(sp["cal"][0])
    G["labels"] = {**G["labels"], "family": "phys_pair"}
    sp["physical"].append(G)
    return sp


def test_structured_ensemble_mixture(splits, tmp_path):
    m = StructuredEnsemble(dict(n_members=2, device=DEV, steps=2, val_every=10 ** 9, log_every=10 ** 9))
    assert m.name == "structured_ensemble" and all(type(x).__name__ == "StructuredNPE" for x in m.members)
    assert len({x.cfg["seed"] for x in m.members}) == 2
    m.fit(D.batch_iter(splits["cal"], 16, 8, seed=0), None)
    v = D.apply_view(splits["heldout"][0], "E1", ("P1", "P5"))
    p = m.predict(v, 32)
    ps = [x.predict(v, 16) for x in m.members]
    assert torch.allclose(p["mode_prob"], torch.stack([q["mode_prob"] for q in ps]).mean(0), atol=1e-5)
    assert p["y_samples"].shape == (8, 32, 14) and p["y_q"].shape == (8, 14, 3)
    assert torch.allclose(p["mode_prob"].sum(-1), torch.ones(8, v["wheel_x"].shape[1]), atol=1e-4)
    m.save(tmp_path / "se.pt")
    m2 = StructuredEnsemble.load(tmp_path / "se.pt", device=DEV)
    assert torch.allclose(m2.predict(v, 32)["mode_prob"], p["mode_prob"], atol=1e-5)


@pytest.mark.parametrize("klass", [StructuredEnsemble, Blackbox])
def test_c2_pipeline(splits, tmp_path, klass):
    cfg = dict(device=DEV, n_members=2) if klass is StructuredEnsemble else dict(device=DEV)
    m = klass(cfg)
    out = C.run_method(m.name, splits, configs=[("E1", "P1P5"), ("E0", "P1P5P2")], model=m, out_root=tmp_path)
    res = out["configs"]["E1/P1P5"]
    assert res["raw"]["heldout/all"]["n_trials"] == 16 and "heldout/k5" in res["raw"]
    cf = res["conformal"]
    assert cf["n_cal"] == 48 and cf["n_cal_families"] == 6
    for k in ("raw", "split", "group_ss1", "group_pooled"):
        assert 0 <= cf[f"heldout/{k}/all"]["cover90"] <= 1
    # split CQR widens (or keeps) intervals when the raw model is under-covered on cal
    assert (klass is Blackbox) == ("heldout/aps/all" not in cf)
    pr = torch.load(tmp_path / f"preds_{m.name}_P1P5.pt", weights_only=False)
    assert pr["y_q"].shape == (16 + 48 + 8, 14, 3) and pr["mode_prob"].shape[1:] == (max(F["wheel_x"].shape[1] for v in splits.values() for F in v), 5)
    assert len(pr["family"]) == pr["y"].shape[0] and set(pr["set"]) == {"cal", "heldout", "physical"}
    assert (tmp_path / f"preds_{m.name}_P1P5P2_E0.pt").exists()
    json.loads((tmp_path / f"results_C2_{m.name}.json").read_text())
    assert set(res["lofo"]) == {"split", "group_ss1", "group_pooled"}
    lad, conf = C.merge(tmp_path)
    assert f"{m.name}/E1/P1P5" in lad and (tmp_path / "report_C2.md").exists()


def test_conformal_split_calibrates_in_distribution():
    """Exchangeable synthetic check: split CQR on y_q restores ~90 % coverage from an under-covering model."""
    rng = np.random.default_rng(0)
    N = 4000
    y = rng.normal(size=(N, 14))
    yq = np.stack([np.full((N, 14), -1.0), np.zeros((N, 14)), np.full((N, 14), 1.0)], -1)   # 68 % raw
    rec = lambda a, b: dict(y_q=torch.tensor(yq[a:b]), y=torch.tensor(y[a:b]), family=[f"f{i % 50}" for i in range(a, b)],
                            group_id=[f"g{i % 50}" for i in range(a, b)],
                            size=[4] * (b - a), consequential=np.zeros(b - a, bool),
                            mode_prob=torch.full((b - a, 4, 5), float("nan")), wheel_mask=torch.ones(b - a, 4, dtype=torch.bool))
    res = C.conformal_tables(rec(0, 2000), {"heldout": rec(2000, N)})
    assert abs(res["heldout/raw/all"]["cover90"] - 0.68) < 0.03
    assert abs(res["heldout/split/all"]["cover90"] - 0.90) < 0.02
    assert res["heldout/group_ss1/all"]["cover90"] > 0.85 and res["heldout/group_asm_pooled/all"]["cover90"] > 0.85


def test_load_never_raises_gpu_cap(tmp_path):
    """M1: Predictor.load lowers but never raises the process's CUDA memory cap (run in a fresh process)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    import subprocess
    import sys
    code = f"""
import torch
from capxfer.learn.base import current_mem_frac
from capxfer.learn.blackbox import Blackbox
m = Blackbox(dict(device="cuda", mem_frac=0.25)); m.save(r"{tmp_path}/bb.pt")
torch.cuda.set_per_process_memory_fraction(0.05)
Blackbox.load(r"{tmp_path}/bb.pt"); a = current_mem_frac()
Blackbox.load(r"{tmp_path}/bb.pt", mem_frac=0.5); b = current_mem_frac()
Blackbox.load(r"{tmp_path}/bb.pt", mem_frac=0.03); c = current_mem_frac()
assert abs(a - 0.05) < 1e-6 and abs(b - 0.05) < 1e-6 and abs(c - 0.03) < 1e-6, (a, b, c)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=55,
                       env={**__import__("os").environ, "PYTHONPATH": "."})
    assert r.returncode == 0, r.stderr[-2000:]


def test_retrain_after_sysid_versioned(splits, tmp_path):
    """Warm-start fine-tune from a base generation dir into models/<tag>/ with a manifest (M2)."""
    from capxfer.learn import registry as R
    from capxfer.learn.retrain import resolve_dr_spec, retrain_after_sysid
    base = tmp_path / "models" / "base"
    base.mkdir(parents=True)
    StructuredEnsemble(dict(n_members=2, device=DEV)).save(base / "structured_ensemble.pt")
    fit = {"best": {"mu": 0.55, "s0": 0.03}}
    dr = resolve_dr_spec(fit, 0.1)
    assert dr["mu"] == pytest.approx((0.495, 0.605))
    info = retrain_after_sysid(fit, base, ("structured_ensemble", "analytic_bayes"), tag="t1", feats=splits["cal"],
                               root=tmp_path, steps=3, per_asm=4, batch_size=8)
    out = tmp_path / "models" / "t1"
    man = json.loads((out / "manifest.json").read_text())
    e = man["methods"]["structured_ensemble"]
    assert e["sha256"] == R.sha256_file(out / "structured_ensemble.pt") and len(e["code_sha256"]) == 64
    assert e["base_sha256"] == R.sha256_file(base / "structured_ensemble.pt") and e["params_sha256"]
    assert info["methods"]["analytic_bayes"].startswith("skipped") and info["total_s"] > 0
    m = StructuredEnsemble.load(out / "structured_ensemble.pt", device=DEV)
    assert m.predict(D.apply_view(splits["heldout"][0], "E1"), 8)["mode_prob"] is not None

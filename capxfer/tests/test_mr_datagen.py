"""WS-MR / D17: datagen healthy-baseline labels (relative capability) and the measured-centred sensor DR."""
import numpy as np
import torch

from capxfer import datagen, sim


def test_healthy_baseline_labels(tmp_path):
    m = datagen.generate("hb", 96, {"pair": 1.0}, seed=3, sizes=(2,), phys_trials_per_asm=96, wheel_budget=1024,
                         engine="eager" if not torch.cuda.is_available() else "graph",
                         device="cuda" if torch.cuda.is_available() else "cpu", out_root=tmp_path, dr=sim.DR_MEASURED,
                         healthy_baseline=True, log=lambda s: None)
    assert m["args"]["healthy_baseline"] and m["args"]["sensor_dr"]["truth_latency"] == [0.048, 0.108]
    raws = list(datagen.iter_dataset(tmp_path / "hb"))
    lab = raws[0]["labels"]
    rh, rr, ra = lab["rho_healthy"], lab["rho_rel_twist"], lab["rho_twist"]
    assert rh.shape == ra.shape == (96, 14)
    h = (lab["mode"] == 0).all(1)
    assert h.any() and (~h).any()
    assert torch.allclose(ra[h], rh[h], atol=1e-6)                     # healthy trial == its own healthy twin
    assert torch.allclose(rr, ra / rh.clamp(min=datagen.REL_FLOOR))
    assert float(rh.min()) > 0.5                                       # confirmed firmware: healthy rho ~0.7-1.0
    # a faulted wheel never raises relative capability much above the healthy twin
    assert float(rr[~h].median()) < 1.0


def test_noload_range_covers_045_065_surface_speed():
    lo, hi = sim.DR_MEASURED["noload_scale"]
    from capxfer import assembly
    P = assembly.load_params("params/module_measured.yaml")
    nom = sim.nominal(P)
    v = lambda s: s * nom["omega0"] * nom["radius"] * nom["v_batt"] / nom["v0"]
    assert v(lo) <= 0.45 and v(hi) >= 0.65

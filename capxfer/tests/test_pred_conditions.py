"""WS-PRED: the frozen per-condition simulator prediction file (pred.csv) and the robot-view guard (CPU, synthetic)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from capxfer.analysis import realv2 as RV
from capxfer.analysis import realv2_design as D
from capxfer.field import predict as PR

SEED = bytes.fromhex("5e" * 32)


def _modes():
    return D.expand_schedule(SEED)["mode"]


def test_conditions_cover_the_mask_design():
    both = PR.conditions()
    assert len(both) == 2 * (36 + 16)                               # both passive modes when the modes are sealed
    sealed = PR.conditions(modes=_modes())
    assert len(sealed) == 36 + 16 and len({(b, m, w) for b, m, w, _ in sealed}) == 52
    assert all(fw == _modes()[f"{m}:{w}"] for _, m, w, fw in sealed)


def test_condition_predictions_small(tmp_path):
    torch.set_num_threads(4)
    conds = [c for c in PR.conditions() if c[0] in ("pair", "single_M3") and c[1] == "M3" and c[2] == "fl"]
    out = PR.write_pred(tmp_path / "pred_full.csv", conds, n_fault=1, n_healthy=2, seed=3, dt=0.01)
    f = pd.read_csv(out, keep_default_na=False)
    assert list(f.columns[:6]) == list(RV.PRED_COLS)
    assert set(PR.PROV_COLS) <= set(f.columns) and set(RV.RHO_COLS) <= set(f.columns)
    assert len(f) == 2 * (1 + 2)                                    # healthy row + coast + brake, per build
    assert (f["params_sha256"].str.len() == 64).all() and (f["code_sha256"].str.len() == 64).all()
    assert (f["lambda_sha256"] == "").all() and (f["theta_source"] == "nominal").all()
    lone = f[(f["build"] == "single_M3") & (f["module"] == "M3")]
    pair = f[(f["build"] == "pair") & (f["module"] == "M3")]
    assert (lone["L_pred"].astype(float) > 0.2).all() and (lone["z_pred"].astype(float) < 0.8).all()
    assert (pair["L_pred"].astype(float) < lone["L_pred"].astype(float).min()).all()       # masking in the assembly
    assert (pair["z_qs"].astype(float) > 0.9).all() and (lone["L_qs"].astype(float) > 0.2).all()
    assert all(k in RV.KEYS for k in f.loc[f["module"] != "", "k_star"])
    assert (tmp_path / "pred_full_rho.csv").is_file()
    # select: sealed mode -> exactly one row per condition, healthy rows kept
    sched = {"episodes": [{"faults": [{"mode": "brake", "wheel": "M3:fl"}]}]}
    sel = PR.select_modes(out, sched, tmp_path / "pred.csv")
    s = pd.read_csv(sel, keep_default_na=False)
    assert len(s) == 4 and set(s.loc[s["module"] != "", "firmware"]) == {"brake"}
    with pytest.raises(ValueError):
        PR.select_modes(out, {"episodes": [{"faults": [{"mode": "coast", "wheel": "M1:fl"}]}]}, tmp_path / "bad.csv")


def test_robot_view_drops_truth_and_evaluator_labels():
    raw = dict(t=torch.zeros(3), duty=torch.zeros(1, 3, 4), counts=torch.zeros(1, 3, 4), truth_pose=torch.ones(1, 3, 3),
               meta=dict(assembly="a", labels=dict(mode=torch.zeros(1, 4), g=torch.ones(1, 4), domain={},
                                                   schedule_trial=[1], mode_echo=torch.zeros(1, 4))))
    v = PR.robot_view(raw)
    assert "truth_pose" not in v and set(v["meta"]["labels"]) == {"schedule_trial"}
    assert "mode_echo" in PR.robot_view(raw, keep_echo=True)["meta"]["labels"]


def test_choice_rules():
    s = {"M2": 0.9, "M3": 0.9, "M4": 0.2}
    assert PR._choice(s, 0.5, "M3") == "M3" and PR._choice(s, 0.95, "M3") is None
    assert PR._choice(s, 0.9, "M2") is None and PR._choice(s, 0.9, "M2", strict=False) == "M2"
    assert PR._choice(s, -math.inf, "M4") == "M2"                   # tie coin outside the tie -> first in order
    assert np.isfinite(PR._loss([1, 1], [0.9, 0.5])[0]) and PR._loss([1, 1], [0.9, 0.5])[1] == RV.KEYS[1]

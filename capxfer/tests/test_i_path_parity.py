"""WS-DX regression: the field path (sim -> FieldLog session -> loader -> featurize) and the in-memory sim path
(sim -> sensors.to_raw -> featurize) must give the same features and targets for the same simulated plant.
This is the check that cleared the loader/featurize path as the cause of the dry-run coverage collapse."""
import numpy as np
import torch

from capxfer import assembly as A, featurize as FZ, sensors, sim as S
from capxfer.field import loader as LD, schedule as SCH, synth_session as SS
from capxfer.pipeline import adapters as AD, dryrun as DR


def test_session_path_matches_inmemory_path(tmp_path):
    prm = A.load_params(SS.Path(SS.__file__).resolve().parents[2] / "params" / "module_measured.yaml")
    sched = SCH.generate(f"{5:032x}", assemblies=("single",), params=prm,
                         counts={"F0": 1, "F1": 1, "F2": 1, "F3": 0, "F5": 0},
                         operator={"F4": (0, 1), "F6": (0, 1), "F7": (0, 1)})
    d, truth = SS.make_session(tmp_path, "single", n_trials=3, seed=3, schedule=sched, current_wired=True,
                               device="cpu", dom=lambda B, rng: DR.true_domain(B, rng))
    R = truth["R"]
    raw_a = sensors.to_raw(R, prm, truth["dom"], np.random.default_rng(0))
    raw_b = LD.load_session(d, allow_unverified=True, schedule=sched)[0]
    nums = raw_b["meta"]["labels"]["schedule_trial"]
    raw_b = AD.subset_raw(raw_b, [nums.index(t["trial"]) for t in truth["trials"]])
    Fa, Fb = FZ.featurize(raw_a, "E2"), FZ.featurize(raw_b, "E2")
    assert torch.equal(Fa["wheel_mask"], Fb["wheel_mask"]) and torch.equal(Fa["body_mask"], Fb["body_mask"])
    assert torch.allclose(Fa["wheel_geom"], Fb["wheel_geom"])
    for c in (1, 2):                                                     # enc_speed, duty: same sign/order/units
        a, b = Fa["wheel_x"][..., c].flatten(), Fb["wheel_x"][..., c].flatten()
        assert np.corrcoef(a, b)[0, 1] > 0.99
        assert (a - b).abs().mean() < 0.05 * a.abs().mean()
    rho, _ = S.targets_from_twist(R)
    for F in (Fa, Fb):                                                   # targets: no latency / frame / origin bias
        err = (F["y"] - rho.float()).abs()
        assert float(err.nanmean()) < 0.03
        assert float(err.nan_to_num(0).amax()) < 0.1

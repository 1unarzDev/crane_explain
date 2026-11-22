"""WS-R regression (DX side finding): the synthetic session's motor-current stream must follow sensors.to_raw's
current model (unidirectional CS: ~0 on BRAKE/COAST wheels, |I_free tanh + tau_em/kt| on driven wheels, per-trial
sync/gain/offset/noise), not a signed torque/kt. Checked through the full sim -> session -> loader path."""
import numpy as np
import torch

from capxfer import assembly as A, sensors, sim as S
from capxfer.field import loader as LD, schedule as SCH, synth_session as SS
from capxfer.pipeline import adapters as AD, dryrun as DR

EXACT = dict(current_noise_a=(0.0, 0.0), current_gain=(1.0, 1.0), current_offset_std=0.0, p_current_synced=1.0,
             stream_delay=(0.0, 0.0))


def _session(tmp_path, monkeypatch=None, n_trials=6, seed=4):
    prm = A.load_params(SS.Path(SS.__file__).resolve().parents[2] / "params" / "module_measured.yaml")
    sched = SCH.generate(f"{seed:032x}", assemblies=("single",), params=prm,
                         counts={"F0": 1, "F1": 3, "F2": 2, "F3": 0, "F5": 0},
                         operator={"F4": (0, 1), "F6": (0, 1), "F7": (0, 1)})
    d, truth = SS.make_session(tmp_path, "single", n_trials=n_trials, seed=seed, schedule=sched, current_wired=True,
                               device="cpu", dom=lambda B, rng: DR.true_domain(B, rng))
    raw = LD.load_session(d, allow_unverified=True, schedule=sched)[0]
    nums = raw["meta"]["labels"]["schedule_trial"]
    raw = AD.subset_raw(raw, [nums.index(t["trial"]) for t in truth["trials"]])
    return prm, truth, raw


def test_current_stream_equals_to_raw_model():
    """Noise-free sensor settings: the writer's current formula is exactly sensors.to_raw's."""
    prm = A.load_params(SS.Path(SS.__file__).resolve().parents[2] / "params" / "module_measured.yaml")
    asm = A.physical("pair", prm)
    B, n = 6, asm.n
    rng = np.random.default_rng(0)
    mode = torch.zeros(B, n, dtype=torch.long)
    g = torch.ones(B, n)
    mode[1, 0], mode[2, 1], mode[3, 2], mode[4, 3] = S.BRAKE, S.COAST, S.WEAK, S.ROLLER
    g[3, 2] = 0.5
    dom = S.sample_domain(B, rng)
    R = S.simulate(asm, prm, dom, mode, g, device="cpu", seed=0)
    ref = sensors.to_raw(R, prm, dom, np.random.default_rng(1), sensor_dr=EXACT)["current"]
    cur = torch.as_tensor(SS.current_stream(R, prm, np.random.default_rng(2), mode, sensor_dr=EXACT))
    assert torch.allclose(cur, ref.float(), atol=1e-5)
    assert float(cur[1, :, 0].abs().max()) == 0.0 and float(cur[2, :, 1].abs().max()) == 0.0
    assert float(cur.min()) >= 0.0


def test_session_current_matches_sensor_model(tmp_path, monkeypatch):
    for k, v in EXACT.items():
        monkeypatch.setitem(sensors.SENSOR_DR, k, v)
    prm, truth, raw = _session(tmp_path)
    R = truth["R"]
    ref = sensors.to_raw(R, prm, truth["dom"], np.random.default_rng(0), sensor_dr=EXACT)["current"].float()
    cur = raw["current"].float()
    T = min(cur.shape[1], ref.shape[1])
    cur, ref = cur[:, 20:T - 20], ref[:, 20:T - 20]
    assert float(cur.min()) >= 0.0                                       # unidirectional: never a signed torque/kt
    mode = torch.stack([t["mode"] for t in truth["trials"]])
    off = (mode == S.BRAKE) | (mode == S.COAST)
    if off.any():
        assert float(cur.permute(0, 2, 1)[off].abs().max()) < 1e-4       # BRAKE/COAST wheels read zero
    a, b = cur.flatten().numpy(), ref.flatten().numpy()
    assert np.corrcoef(a, b)[0, 1] > 0.97
    assert abs(a.mean() - b.mean()) < 0.05 * b.mean() + 0.01

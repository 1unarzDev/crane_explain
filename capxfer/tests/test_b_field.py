"""WS-B field tooling tests: sealed schedule, probe runner, FieldLog config, synthetic SIMULATION session + loader."""
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from capxfer import assembly as A
from capxfer import featurize as F
from capxfer.field import fieldlog_config as FC
from capxfer.field import probe_runner as PR
from capxfer.field import schedule as SCH

P = A.load_params("params/module_measured.yaml")
SMALL = dict(counts={"F0": 2, "F1": 3, "F2": 2, "F3": 2, "F5": 3}, operator={"F4": (1, 2), "F6": (1, 2), "F7": (1, 2)},
             auto_block=4)


# ----------------------------------------------------------------------------- schedule
def test_schedule_sealed_and_reproducible(tmp_path):
    r = SCH.write(tmp_path / "s", seed_dir=tmp_path / "sealed", **SMALL)
    raw = (tmp_path / "s" / "schedule.json").read_text()
    assert r["seed_hex"] not in raw and not (tmp_path / "s" / "seed.txt").exists()
    assert (tmp_path / "sealed" / "seed.txt").read_text().strip() == r["seed_hex"]
    assert SCH.verify(tmp_path / "s", r["seed_hex"])
    assert not SCH.verify(tmp_path / "s", "0" * 32)
    s = SCH.load(tmp_path / "s" / "schedule.json")
    assert s["n_trials"] == 4 * (12 + 6)
    assert sorted(s["assemblies_order"]) == sorted(SCH.ASSEMBLIES)
    for sess in s["sessions"]:
        asm = A.physical(sess["assembly"], P)
        kinds = [b["kind"] for b in sess["blocks"]]
        assert kinds.count("operator") == 3
        for b in sess["blocks"]:
            if b["kind"] == "operator":
                # fixed physical condition per block (F6w/F7w trials add one firmware wheel fault on top)
                assert len({json.dumps([f for f in t["faults"] if f["mode"] not in SCH.FW_MODES])
                            for t in b["trials"]}) == 1
            for t in b["trials"]:
                for f in t["faults"]:
                    if ":" in f["wheel"]:
                        assert f["wheel"] in asm.wheel_corner
                    if f["mode"] == "weak":
                        assert 0.3 <= f["g"] <= 0.9
                if t["fault_id"] == "F5":
                    assert len({f["wheel"] for f in t["faults"]}) == 2
                assert t["ood"] == {"F6": "payload", "F7": "low_mu"}.get(t["fault_id"], "none")
    nums = [t["trial"] for _, _, t in SCH.iter_trials(s)]
    assert nums == list(range(1, len(nums) + 1))
    # tampering is refused
    bad = tmp_path / "s" / "schedule.json"
    bad.chmod(0o644)
    bad.write_text(raw.replace('"F0"', '"F1"', 1))
    with pytest.raises(ValueError):
        SCH.load(bad)


def test_runsheet_readable_by_fieldlog(tmp_path):
    sys.path.insert(0, str(FC.FIELDLOG))
    import csvlog_v2
    r = SCH.write(tmp_path / "s", seed_hex="ab" * 16, **SMALL)
    rows = csvlog_v2.read_runsheet(tmp_path / "s" / "runsheet_pair.csv", "2.5")
    assert len(rows) == 18 and rows[0]["trial_id"] == "gnd_capxprobe_001"
    assert {int(x["portion_code"]) for x in rows} == {FC.PORTION_CODE}
    assert r["sha256"] == SCH.write(tmp_path / "s2", seed_hex="ab" * 16, **SMALL)["sha256"]


# ----------------------------------------------------------------------------- probe runner
class RecPub:
    def __init__(self):
        self.ev = []

    def marker(self, d):
        self.ev.append(("marker", None, d))

    def assembly(self, m, d):
        self.ev.append(("assembly", m, d))

    def fault(self, m, d):
        self.ev.append(("fault", m, d))

    def twist(self, mods, xi):
        self.ev.append(("twist", tuple(mods), xi))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.t += max(d, 1e-4)


def _sched():
    return SCH.generate("cd" * 16, assemblies=("pair",), params=P, **SMALL)


def test_runner_publishes_trial(tmp_path):
    s = _sched()
    sess, trials = PR.select_trials(s, "pair")
    tr = next(t for t in trials if t["fault_id"] == "F5")
    pub, clk = RecPub(), FakeClock()
    run = PR.Runner(pub, P, clock=clk, sleep=clk.sleep)
    assert run.run([tr], sess, gap=0.0) == 1
    kinds = [e[0] for e in pub.ev]
    assert pub.ev[0] == ("marker", None, {"trial": tr["trial"], "event": "start", "asm_type": "pair"})
    assert kinds.count("assembly") == 2 and kinds[1:3] == ["assembly", "assembly"]
    faults = [e for e in pub.ev if e[0] == "fault"]
    assert len(faults) == 4 and [f[2]["mode"] for f in faults[2:]] == ["ok", "ok"]
    tw = [e for e in pub.ev if e[0] == "twist"]
    T = PR.trial_plan(tr, sess, P)["T"]
    assert abs(len(tw) - (np.ceil(T * 50) + 2)) <= 1
    assert all(e[1] == ("M1", "M3") for e in tw)
    assert max(abs(e[2][1]) for e in tw) == pytest.approx(0.45, abs=1e-6)       # P6y hold
    assert pub.ev[-1][2]["event"] == "end" and tw[-1][2] == (0.0, 0.0, 0.0)


def test_runner_estop(tmp_path):
    s = _sched()
    sess, trials = PR.select_trials(s, "pair")
    pub, clk = RecPub(), FakeClock()
    run = PR.Runner(pub, P, clock=clk, sleep=clk.sleep)
    orig = pub.twist

    hit = []

    def twist(mods, xi):
        orig(mods, xi)
        if clk.t > 5.0 and not hit:
            hit.append(len(pub.ev))
            run.request_stop()
    pub.twist = twist
    assert run.run(trials[:3], sess, gap=1.0) == 0
    tw_idx = [i for i, e in enumerate(pub.ev) if e[0] == "twist"]
    abort = [i for i, e in enumerate(pub.ev) if e[0] == "marker" and e[2]["event"] == "abort"]
    assert len(abort) == 1
    stop_at = hit[0]
    assert abort[0] > stop_at
    assert all(pub.ev[i][2] == (0.0, 0.0, 0.0) for i in tw_idx if i >= stop_at)
    assert sum(1 for i in tw_idx if i >= stop_at) >= 10
    assert not any(e[0] == "marker" and e[2]["event"] == "start" and e[2]["trial"] != trials[0]["trial"] for e in pub.ev)


def test_runner_importable_and_dry_run(tmp_path):
    r = SCH.write(tmp_path / "s", seed_hex="ef" * 16, **SMALL)
    out = subprocess.run([sys.executable, "-c", "import sys; sys.modules['rclpy'] = None; import capxfer.field.probe_runner"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    out = subprocess.run([sys.executable, "-m", "capxfer.field.probe_runner", str(r["schedule"]), "--assembly", "L3", "--dry-run",
                          "--to-trial", "1000"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "P6y-" in out.stdout and "marker" in out.stdout and "total 18 trials" in out.stdout


# ----------------------------------------------------------------------------- FieldLog config
def test_config_file_and_overlay(tmp_path):
    assert FC.CONFIG_PATH.is_file()
    cfg = FC.load_config()
    assert cfg == yaml.safe_load(yaml.safe_dump(FC.build_config()))
    assert "TEAM-CONFIRM" in FC.CONFIG_PATH.read_text()
    srcs = {t["src"] for t in cfg["topics"]}
    assert srcs == set(FC.source_table())
    assert any(t["topic"].startswith("/fleet/truth/") for t in cfg["topics"])
    r = FC.render(cfg, {"M1": "alpha"})
    assert "/fleet/alpha/wheel_state" in [t["topic"] for t in r["topics"]]
    assert not any("{" in t["topic"] for t in r["topics"])
    sys.path.insert(0, str(FC.FIELDLOG))
    import csvlog_v2
    cb = csvlog_v2.Codebook(FC.overlay_codebook(tmp_path, cfg))
    tok, por = cb.portion_by_code("gnd", FC.PORTION_CODE)
    assert tok == FC.PORTION_TOKEN and len(por["columns"]) == len(set(por["columns"]))
    for t in cfg["topics"]:
        assert set(t["columns"]) <= set(cb.source("gnd", t["src"])["columns"])


# ----------------------------------------------------------------------------- synthetic session + loader
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from capxfer.field import synth_session as SS
    from capxfer.field import loader as L
    root = tmp_path_factory.mktemp("synth")
    s = SCH.generate("12" * 16, assemblies=("pair",), params=P, **SMALL)
    sess = s["sessions"][0]
    # a healthy, a weak+brake (F5) and a payload (F6) trial
    pick = [next(t for _, _, t in SCH.iter_trials(s) if t["fault_id"] == f) for f in ("F0", "F5", "F6")]
    sess["blocks"] = [{"block": 1, "kind": "auto", "fault_id": None, "condition": {}, "trials": pick}]
    d, truth = SS.make_session(root, "pair", n_trials=3, seed=7, schedule=s, current_wired=False)
    return d, truth, L.verify(d)


def test_synthetic_session_is_simulation(synth):
    d, truth, rep = synth
    meta = yaml.safe_load((d / "session.yaml").read_text())
    assert meta["label"] == "SIMULATION" and rep["verdict"] == "SIMULATION"


def test_loader_gate(synth):
    from capxfer.field import loader as L
    d, truth, rep = synth
    with pytest.raises(L.SessionRefused):
        L.load_session(d, verdict=rep)
    # even a (forged) FIELD_OK verdict never turns a SIMULATION-labelled session into FIELD data
    raws = L.load_session(d, verdict={"verdict": "FIELD_OK"})
    assert raws[0]["meta"]["source"] == "SIMULATION"


def test_loader_roundtrip(synth):
    from capxfer.field import loader as L
    d, truth, rep = synth
    raws = L.load_session(d, allow_unverified=True, verdict=rep)
    assert len(raws) == 1
    raw = raws[0]
    asm = raw["meta"]["assembly"]
    R = truth["R"]
    B, T, n = raw["duty"].shape
    assert (B, n) == (3, 8) and T == len(raw["t"]) and np.allclose(raw["t"][1] - raw["t"][0], 0.01)
    assert asm.wheel_corner == truth["asm"].wheel_corner
    assert raw["meta"]["source"] == "SIMULATION"
    assert raw["counts"].dtype == torch.long and torch.isnan(raw["current"]).all()
    assert torch.isfinite(raw["lidar_twist"]).all() and torch.isfinite(raw["truth_pose"]).all()
    lab = raw["meta"]["labels"]
    for b, tt in enumerate(truth["trials"]):
        assert torch.equal(lab["mode"][b], tt["mode"])
        assert torch.allclose(lab["g"][b], tt["g"], atol=1e-3)
    assert lab["ood_per_trial"] == ["none", "none", "payload"]
    # robot-visible streams agree with the simulator
    Tm = min(T, R["twist"].shape[1])
    assert (raw["gyro_z"][:, :Tm] - R["twist"][:, :Tm, 2]).abs().mean() < 0.01
    ang = raw["counts"][:, :Tm].float() * 2 * np.pi / raw["meta"]["cpr"]
    # receipt time = acquisition + ~2.5 ms, so the zero-order hold on the 100 Hz grid lags the truth by one sample
    assert (ang[:, 1:] - R["wheel_angle"][:, :Tm - 1]).abs().max() < 0.01
    # evaluator target from the loaded truth matches the simulator's true twist
    f = F.featurize(raw, "E2")
    sched = truth["schedule"]
    Lc = asm.char_length()
    W = np.array([1, 1, Lc ** 2])
    tw, tt = R["twist"].numpy(), R["t"].numpy()
    yref = np.stack([((tw[:, (tt >= ta) & (tt <= tb)].mean(1)) * c * W).sum(-1) / (c * c * W).sum()
                     for _, _, ta, tb, c in sched.windows()], 1)
    e = f["y"].numpy() - yref
    assert np.sqrt((e ** 2).mean()) < 0.025, e
    assert f["body_mask"].tolist() == [1.0] * 9

"""WS-D3: consequential OOD field specs (WS-A2) in the sealed schedule and the dry-run simulation."""
import numpy as np

from capxfer import assembly as A
from capxfer.field import schedule as SCH
from capxfer.pipeline import dryrun as DR


def test_d3_specs_and_allocation():
    s = SCH.generate(f"{21:032x}")
    assert s == SCH.generate(f"{21:032x}")
    assert s["n_trials"] == 1420
    for sess in s["sessions"]:
        conds = [t["condition"] for b in sess["blocks"] for t in b["trials"]]
        single = sess["assembly"] == "single"
        assert len(conds) == (370 if single else 350)                       # PROTOCOL section 3 / 4.1 (D3b)
        assert sum(c == "F5" for c in conds) == 60 and sum(c == "F1" for c in conds) == (40 if single else 60)
        n6 = sum(c.startswith("F6") for c in conds)
        assert n6 == (50 if single else 30 if sess["assembly"] in SCH.F6_TYPES else 0)
        assert sum(c.startswith("F7") for c in conds) == (50 if single else 60 - n6)
        assert sum(c.endswith("w") for c in conds) == (90 if single else 30)
        for b in sess["blocks"]:
            c = b["condition"]
            if c.get("kind") == "payload":
                assert SCH.PAYLOAD_KG[0] <= c["mass_kg"] <= 6.0 and 0.08 <= c["offset_m"] <= 0.12
                assert abs(np.hypot(c["dx_m"], c["dy_m"]) - c["offset_m"]) < 1e-3
                assert c["module"] in ("M1", "M3")
            if c.get("kind") == "low_mu":
                assert c["region"] == "left" and SCH.SHEET_MU[0] <= c["mu"] <= SCH.SHEET_MU[1]


def test_mixed_type_field_sessions_runsheets_and_verify(tmp_path):
    """D3c: every field session holds blocks of all four types; each type's cells are spread over sessions."""
    w = SCH.write(tmp_path / "sched", seed_hex=f"{22:032x}", seed_dir=tmp_path / "sealed")
    s = SCH.load(w["schedule"])
    fs = SCH.field_sessions(s)
    assert len(fs) == SCH.N_SESSIONS >= 8 and sum(f["n_trials"] for f in fs) == s["n_trials"]
    for f in fs:
        assert f["assemblies"] == sorted(SCH.ASSEMBLIES)
        assert f["n_reassemblies"] == sum(r["reassemble"] for r in f["blocks"]) >= 4
    # every operator cell of a type lands in as many sessions as it has blocks (stratified dealing)
    for sess in s["sessions"]:
        for fid in ("F4", "F6", "F7"):
            bl = [b for b in sess["blocks"] if b["fault_id"] == fid]
            assert len({b["field_session"] for b in bl}) == min(len(bl), SCH.N_SESSIONS)
    seen = set()
    for f in fs:
        sub = SCH.sub_schedule(s, f["id"])
        tr = [t["trial"] for _, _, t in SCH.iter_trials(sub)]
        assert len(tr) == f["n_trials"] and not (seen & set(tr))
        seen |= set(tr)
        rows = list(open(tmp_path / "sched" / f"runsheet_{f['id']}.csv"))
        assert len(rows) == f["n_trials"] + 1
        blocks = list(open(tmp_path / "sched" / f"blocks_{f['id']}.csv"))
        assert len(blocks) == len(f["blocks"]) + 1
        for a in f["assemblies"]:
            assert SCH.sub_schedule(s, f["id"], a)["sessions"][0]["assembly"] == a
    assert len(seen) == s["n_trials"]
    assert SCH.verify(tmp_path / "sched", f"{22:032x}")


def test_dryrun_sim_faults_per_wheel_sheet_and_payload_offset():
    params = A.load_params(SCH.Path(SCH.__file__).resolve().parents[2] / "params" / "module_measured.yaml")
    asm = A.physical("single", params)
    trials = [{"faults": [{"wheel": "none", "mode": "low_mu", "region": "left", "mu": 0.22}]},
              {"faults": [{"wheel": "none", "mode": "payload", "module": "M1", "mass_kg": 5.0, "dx_m": 0.07,
                           "dy_m": -0.07}, {"wheel": "M1:fl", "mode": "coast"}, {"wheel": "M1:rr", "mode": "brake"}]}]
    mode, g, pay, mu = DR.sim_faults_d3(trials, asm)
    left = asm.wheel_pos[:, 1] > 0
    assert mu.shape == (2, 4) and np.allclose(mu[0, left].numpy(), 0.22 / DR.TRUE_DOMAIN["mu"])
    assert np.allclose(mu[0, ~left].numpy(), 1.0) and np.allclose(mu[1].numpy(), 1.0)
    assert abs(float(pay["x"][1]) - 0.07) < 1e-6 and abs(float(pay["y"][1]) + 0.07) < 1e-6
    assert mode[1].tolist() == [2, 0, 0, 3]

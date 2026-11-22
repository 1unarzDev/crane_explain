"""WS-R / D3: F6w/F7w (prereg section 3): 5 of every 10 F6/F7 block trials also carry two passive firmware wheel faults
(coast/brake on distinct wheels; D3 pre-freeze change)."""
import numpy as np

from capxfer.field import loader as LD, schedule as SCH, synth_session as SS
from capxfer.pipeline import adapters as AD

ZERO = {"F0": 0, "F1": 0, "F2": 0, "F3": 0, "F5": 0}


def test_f6w_f7w_allocation():
    s = SCH.generate(f"{11:032x}", counts={"F0": 2, "F1": 2, "F2": 2, "F3": 2, "F5": 2},
                     operator={"F4": (1, 10), "F6": (2, 10), "F7": (2, 10)})
    assert s == SCH.generate(f"{11:032x}", counts={"F0": 2, "F1": 2, "F2": 2, "F3": 2, "F5": 2},
                             operator={"F4": (1, 10), "F6": (2, 10), "F7": (2, 10)})          # deterministic
    for sess in s["sessions"]:
        wheels = set()
        for b in sess["blocks"]:
            for t in b["trials"]:
                wf = [f for f in t["faults"] if f["mode"] in SCH.FW_MODES]
                if t["fault_id"] in ("F6", "F7"):
                    assert t["condition"] == (t["fault_id"] + "w" if wf else t["fault_id"])
                    assert len(wf) in (0, SCH.OOD_WHEEL_FAULT_N) and t["ood"] in ("payload", "low_mu")
                    assert all(f["mode"] in SCH.OOD_WHEEL_FAULT_MODES for f in wf)
                    assert len({f["wheel"] for f in wf}) == len(wf)
                    wheels |= {f["wheel"] for f in wf}
                else:
                    assert t["condition"] == t["fault_id"]
            if b.get("fault_id") in ("F6", "F7"):
                assert sum(t["condition"].endswith("w") for t in b["trials"]) == 5
            if b.get("fault_id") == "F4":
                assert all(len(t["faults"]) == 1 for t in b["trials"])
        assert len(wheels) >= 2


def test_f6w_session_roundtrip(tmp_path):
    sched = SCH.generate(f"{12:032x}", assemblies=("pair",), counts=ZERO,
                         operator={"F4": (0, 1), "F6": (1, 4), "F7": (1, 4)})
    n = sum(len(b["trials"]) for b in sched["sessions"][0]["blocks"])
    d, truth = SS.make_session(tmp_path, "pair", n_trials=n, seed=2, schedule=sched, current_wired=False, device="cpu")
    raw = LD.load_session(d, allow_unverified=True, schedule=sched)[0]
    meta = AD.trial_meta(raw, AD.schedule_index(sched))                 # raises on any label/schedule mismatch
    sidx = AD.schedule_index(sched)
    mode = raw["meta"]["labels"]["mode"].numpy()
    nw = 0
    for i, tid in enumerate(meta["trial_id"]):
        tr = sidx[("pair", AD.trial_num(tid))]
        assert meta["fault"][i] == tr["fault_id"] and meta["ood"][i] == tr["ood"]
        assert int((mode[i] != 0).sum()) == (SCH.OOD_WHEEL_FAULT_N if tr["condition"].endswith("w") else 0)
        assert meta["condition"][i] == tr["condition"]
        nw += tr["condition"].endswith("w")
    assert nw == 4

#!/usr/bin/env python3
"""Generate every run sheet the team needs from a team-drawn seed. Wraps capxfer/field/schedule.py (never edits it);
the designs of the extra protocols come from ../packet_config.yaml (the one file to edit when the design changes).

  make_run_sheets.py main  OUT_DIR --seed-file SEALED/seed_main.txt [options]     sealed main collection (prereg §8)
  make_run_sheets.py extra KIND OUT_DIR --seed-file SEALED/seed_KIND.txt [options] one extra protocol (packet_config kinds:
        shakedown hooktest pilot sysid simval apriltag natural floor2, plus docking; WS-P2 / PROTOCOL_V2 (all from
        capxfer.analysis.realv2_design.expand_schedule of ONE 256-bit seed): masking (card 09, MASK1/MASK2, links in
        episodes.csv), choice (card 10, DEMO paired episodes), simval2 (SIMVAL day 2), tagstatic (AprilTag static))
  make_run_sheets.py plan                                                         timeline + hours from packet_config.yaml

Options: --ids M1=m1,M2=m2,M3=m3,M4=m4 (fleet ids, TEAM-CONFIRM)  --data-dir /data/capxfer/sessions  --operator-codes A,B
  --site-code N  --prereg-sha256 HEX  --study-code 15  --truth-id capxfer_asm  --module Mk (per-module kinds)  --day 1|2
  --session-order block|segment (main; default from packet_config, PENDING OWNER DECISION)  --fieldlog per_session|per_segment
  --naive-arm (docking)  --sealed-dir DIR (docking arm key)  --force (overwrite an existing schedule in OUT_DIR)

Output (OUT_DIR): schedule.json / schedule.sha256 / commit.json (no seed inside), schedule_trials.csv, fieldlog_runsheets.csv
(one row per FieldLog run sheet with its sha256 and trial order; the validator matches sessions by it), SUMMARY.md (trials,
rebuilds, hours, and the other session-order variant for comparison), codebook_overlay/, and per scheduled session
<SID>/SESSION_CARD.md, <SID>/blocks_<SID>.csv and either <SID>/log/ (per_session: one FieldLog session, the probe runner in
--session mode relays the active IMU/rf2o) or <SID>/segNN_<type>/ (per_segment) with runsheet.csv, plan.yaml,
fieldlog_config.yaml and COMMANDS.txt.

The seed: draw it yourself on the robot computer, once per schedule, and keep it sealed (prereg §8):
  head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \\n' > SEALED/seed_main.txt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import packet_lib as PL  # noqa: E402

from capxfer import assembly as A  # noqa: E402
from capxfer.probes import Schedule  # noqa: E402
from capxfer.field import fieldlog_config as FC  # noqa: E402
from capxfer.field import schedule as SCH  # noqa: E402

PARAMS = PL.REPO / "params" / "module_measured.yaml"
CONFIG = PL.PACKET / "packet_config.yaml"
ORDINAL_BASE = {"main": 0, "shakedown": 2000, "hooktest": 2100, "pilot": 2200, "sysid": 2300, "simval": 2400,
                "apriltag": 2500, "natural": 2600, "floor2": 2700, "docking": 2800, "masking": 3000, "choice": 3100,
                "simval2": 3200, "tagstatic": 3300}
NATURAL_TEXT = {
    "jammed_roller": "ONE roller of wheel {w}: one zip tie between that roller and the side plate so it cannot spin; the other rollers stay free",
    "debris_wrap": "50 cm cotton thread, 10 tight turns around one roller axle end of wheel {w}",
    "worn_roller": "replace one roller of wheel {w} with the spare that has a ~3 mm flat (skip the block if there is no spare)",
    "loose_hub": "loosen the set screw / clamp of wheel {w} by 1/4-1/2 turn; the retaining nut or circlip stays in place",
    "low_battery": "pre-discharged pack (~10.8 V resting) in module {m}; stop at 10.5 V under load",
}


def load_config():
    return yaml.safe_load(CONFIG.read_text())


def read_seed(p):
    p = Path(p)
    if not p.is_file():
        sys.exit(f"seed file {p} not found. Draw it on the robot computer (keep it sealed):\n"
                 f"  head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \\n' > {p}")
    s = p.read_text().strip().lower()
    if len(s) < 32 or any(c not in "0123456789abcdef" for c in s):
        sys.exit(f"{p}: the seed must be >= 32 hex characters")
    return s


def parse_kv(s, default=None):
    d = dict(default or {})
    for kv in (s or "").split(","):
        if kv.strip():
            k, v = kv.split("=", 1)
            d[k.strip()] = v.strip()
    return d


# ------------------------------------------------------------------------------------------------ extra schedules
def session_entry(name, spec, params):
    asm = A.build(name, [tuple(m) for m in spec], params)
    frames = [{"module": m, "x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "yaw_deg": 90 * int(yk)}
              for (m, _, _, yk), p in zip(asm.modules, asm.module_pos)]
    imu_m, lidar_m = SCH.sensor_modules(asm)
    return {"assembly": name, "modules": [list(m) for m in asm.modules], "frames": frames, "imu_module": imu_m,
            "lidar_module": lidar_m, "blocks": []}, asm


def fault(rng, wheel, mode, g=None):
    f = {"wheel": wheel, "mode": mode}
    if mode == "weak":
        f["g"] = round(float(g if g is not None else rng.uniform(0.3, 0.9)), 3)
    return f


def firmware_faults(rng, spec, wheels, wheel_choice):
    """spec: ok | F0 | F1 | F2 | F3 | F5 | weak:<g> | coast | brake | passive2 (same semantics as capxfer.field.schedule)."""
    spec = str(spec or "ok")
    pick = lambda: wheel_choice() if wheel_choice else str(rng.choice(wheels))      # noqa: E731
    if spec in ("ok", "F0"):
        return [], "F0"
    if spec.startswith("weak:"):
        return [fault(rng, pick(), "weak", float(spec.split(":")[1]))], "F1"
    if spec in ("coast", "brake", "weak"):
        return [fault(rng, pick(), spec)], {"weak": "F1", "coast": "F2", "brake": "F3"}[spec]
    if spec in ("F1", "F2", "F3"):
        return [fault(rng, pick(), {"F1": "weak", "F2": "coast", "F3": "brake"}[spec])], spec
    ws = rng.choice(wheels, size=2, replace=False)
    if spec == "F5":
        return [fault(rng, str(w), str(rng.choice(("weak", "coast", "brake")))) for w in ws], "F5"
    if spec == "passive2":
        return [fault(rng, str(w), str(rng.choice(("coast", "brake")))) for w in ws], "F5"
    raise ValueError(f"unknown firmware spec {spec!r}")


class Builder:
    """Builds a schedule in the capxfer.fault_schedule/v1 layout (what probe_runner and the loader read).
    WS-P2: one assembly type may appear with several module layouts (e.g. M1 alone, then M2 alone after a split): the
    session entries are keyed by (type, layout); a change of layout is a rebuild. `row` adds fields to the block's
    field-session row (masking: split/phase/round); `fs` puts the block in a named field session (default: one)."""

    def __init__(self, kind, params, trial_base):
        self.kind, self.params = kind, params
        self.trial_no = int(trial_base)
        self.block_no = 1
        self.sessions = {}
        self.order = []

    def add_block(self, name, spec, trials, fault_id=None, instruction="automated: no physical change", kind=None,
                  extra=None, row=None, fs=None):
        key = (name, json.dumps([list(m) for m in spec]))
        if key not in self.sessions:
            self.sessions[key], _ = session_entry(name, spec, self.params)
        tl = []
        for t in trials:
            t = dict(t)
            t["trial"] = self.trial_no
            t.setdefault("ood", {"F6": "payload", "F7": "low_mu"}.get(t["fault_id"], "none"))
            t.setdefault("condition", t["fault_id"])
            self.trial_no += 1
            tl.append(t)
        b = {"block": self.block_no, "kind": kind or ("auto" if fault_id is None else "operator"), "fault_id": fault_id,
             "condition": dict({"kind": "firmware_auto" if fault_id is None else "operator", "instruction": instruction},
                               **(extra or {})), "trials": tl}
        self.block_no += 1
        self.sessions[key]["blocks"].append(b)
        self.order.append((key, b, dict(row or {}), fs))
        return b

    def schedule(self, session_id):
        groups = {}
        for key, b, row, fs in self.order:
            groups.setdefault(fs or session_id, []).append((key, b, row))
        fss = []
        for sid, items in groups.items():
            rows, prev = [], None
            for pos, (key, b, row) in enumerate(items, start=1):
                rows.append(dict({"pos": pos, "block": b["block"], "assembly": key[0], "kind": b["kind"],
                                  "fault_id": b["fault_id"], "n_trials": len(b["trials"]), "reassemble": key != prev}, **row))
                prev = key
            fss.append({"id": sid, "blocks": rows, "n_trials": sum(r["n_trials"] for r in rows),
                        "assemblies": sorted({r["assembly"] for r in rows}),
                        "n_reassemblies": sum(r["reassemble"] for r in rows)})
        return {"schema": SCH.SCHEMA, "generator": f"capxfer/field_packet/tools/make_run_sheets.py extra {self.kind}",
                "packet_kind": self.kind, "packet_config_sha256": PL.sha256_file(CONFIG),
                "assemblies_order": list(dict.fromkeys(k[0] for k in self.sessions)),
                "n_trials": sum(len(b["trials"]) for _, b, _, _ in self.order),
                "sessions": list(self.sessions.values()), "field_sessions": fss, "n_sessions": len(fss),
                "session_order": "block"}


# ------------------------------------------------------------------------------------------------ WS-P2 masking / choice
FID_OF_MODE = {"weak": "F1", "coast": "F2", "brake": "F3", "roller_locked": "F4"}


def fid_of(faults):
    """Fault id of a trial's wheel faults: F0 none; F1-F4 one mode (F4 also for 2 zip-tied wheels, condition F4x2d);
    F5 two firmware faults."""
    modes = [f["mode"] for f in faults]
    if not modes:
        return "F0"
    if all(m == "roller_locked" for m in modes):
        return "F4"
    if len(modes) == 1:
        return FID_OF_MODE[modes[0]]
    return "F5"


def fault_text(faults):
    return ";".join(f"{f['wheel']}:{f['mode']}" + (f":g={f['g']}" if "g" in f else "") for f in faults)


def _realv2(seed_hex):
    """PROTOCOL_V2 sealed expansion (capxfer/analysis/realv2_design.expand_schedule) of the team's 256-bit seed."""
    RD = PL.load_realv2_design()
    seed = bytes.fromhex(seed_hex)
    if len(seed) < 32:
        raise SystemExit("PROTOCOL_V2 needs a 256-bit seed (64 hex characters)")
    X = RD.expand_schedule(seed)
    return RD, X


def _realv2_meta(RD, X):
    return {"commitment": X["commitment"], "realv2_schedule_sha256": RD.schedule_sha256(X), "version": X["version"]}


def _build_spec(build):
    """'square4' / 'single_M2' -> (assembly type, module layout)."""
    if build.startswith("single_"):
        return "single", [[build.split("_", 1)[1], 0, 0, 0]]
    return build, [list(m) for m in A.PHYSICAL_TYPES[build]]


def build_masking(cfg, seed_hex, params):
    """MASK (PROTOCOL_V2 §2.3; card 09) from realv2_design.expand_schedule: builds square4 -> the four lone modules (sealed
    order) -> L3 / pair (sealed order); per build 6 healthy + every (module, wheel) of the build x 3 repeats with its
    sealed passive firmware mode (coast/brake), in the sealed trial order. Every trial is a full probe-library run.
    Link: the condition (module, wheel) is the episode; its lone trials (phase split, single_<module>) link to all its
    parent-assembly trials. Field sessions: packet_config masking.sessions (build -> MASKk)."""
    RD, X = _realv2(seed_hex)
    mc = cfg["masking"]
    fs_of = {b: sid for sid, bl in mc["sessions"].items() for b in bl}
    b = Builder("masking", params, mc["trial_base"])
    by_build = {}
    for t in X["mask_trials"]:
        by_build.setdefault(t["build"], []).append(t)
    units, eps = [], {}
    for build in X["mask_order"]:
        name, spec = _build_spec(build)
        mods = [m for m, *_ in spec]
        assert tuple(sorted(mods)) == tuple(sorted(RD.TYPE_MODULES.get(name, mods))) or name == "single"
        lone = name == "single"
        tl = []
        for t in by_build[build]:
            fl = [] if t["module"] is None else [{"wheel": f"{t['module']}:{t['wheel']}", "mode": t["firmware"]}]
            ep = f"{t['module']}:{t['wheel']}" if t["module"] else f"H_{build}"
            tl.append({"fault_id": fid_of(fl), "faults": fl,
                       "condition": f"MASK:{ep}" if t["module"] else "MASK:healthy",
                       "episode": ep, "phase": "split" if lone else "assembly", "round": build,
                       "rep": int(t.get("rep", -1)) + 1, "episode_modules": "+".join(mods),
                       "fault_carrier": t["module"] or "none", "episode_faults": fault_text(fl),
                       "r3_split": t["r3_split"], "realv2_seq": int(t["seq"])})
        if lone:
            instr = (f"SPLIT -> {mods[0]} ALONE ({build}): undock; ONLY {mods[0]} on the start mark, every other module parked "
                     f">= 1 m away and not commanded; truth tag on {mods[0]} (offset -> session_index.csv); firmware faults "
                     f"are injected automatically")
        else:
            instr = (f"BUILD {name} ({'+'.join(mods)}): dock the modules; truth tag at the assembly tag position (offset -> "
                     f"session_index.csv); no physical fault: firmware faults are injected automatically")
        blk = b.add_block(name, spec, tl, instruction=instr, extra={"kind": "masking", "build": build},
                          row={"split": lone, "phase": "split" if lone else "assembly", "round": build},
                          fs=fs_of[build])
        units.append({"episode": build, "phase": "split" if lone else "assembly",
                      "module": mods[0] if lone else "all", "trials": [x["trial"] for x in blk["trials"]],
                      "fault_carrier": "none", "zip_wheels": []})
        for x in blk["trials"]:
            if x["fault_carrier"] == "none":
                continue
            e = eps.setdefault(x["episode"], {"episode": x["episode"], "design": x["faults"][0]["mode"],
                                              "fault_carrier": x["fault_carrier"], "faults": x["faults"],
                                              "r3_split": x["r3_split"], "parts": {}})
            e["parts"].setdefault(build, []).append(x["trial"])
    sched = b.schedule("MASK")
    sched["episodes"] = sorted(eps.values(), key=lambda e: e["episode"])
    sched["link_units"] = units
    sched["realv2"] = dict(_realv2_meta(RD, X), part="mask", mask_order=X["mask_order"])
    return sched


def demo_l3_expand(seed_hex, cfg):
    """DEMO draws (SPINE D15, docs/ideas/CHOICE.md §4) from realv2_design.expand_schedule(seed)["demo"]: per episode the
    two faulted L3 modules (balanced 7/7/6 over the 3 pairs) with one firmware COAST wheel each, the random-arm module,
    the tie coins, the dock / lone-probe order of the three modules and a start mark per module. The third module is
    the only capable one (= the oracle's module by design; the analysis may also use the measured z_alone)."""
    RD, X = _realv2(seed_hex)
    mods = list(RD.DEMO_MODULES)
    eps = []
    for e in X["demo"]:
        fm = list(e["faulted_modules"])
        eps.append(dict(episode=e["episode"], faulted_modules=fm, coast_wheels=dict(e["faulted_wheels"]),
                        capable_module=next(m for m in mods if m not in fm), random_coin=e["random_module"],
                        cert_tie_coin=e["cert_tie_coin"], carry_over_tie_coin=e["carry_over_tie_coin"],
                        oracle_tie_coin=e["oracle_tie_coin"], dock_order=list(e["dock_order"]),
                        offset_m=dict(e["offset_m"])))
    return RD, X, eps, "realv2_design"


def build_choice(cfg, seed_hex, params):
    """DEMO (SPINE D11, D12, D15; CHOICE.md §4; PROTOCOL_V2 R4, being updated by WS-PR2; card 10): 20 episodes on L3 with
    one firmware COAST wheel on each of two sealed-random modules, narrow bay (sideways entry only). Per episode: build
    L3, one full-library probe (the robot uses P1+P5+P2), choose_module logs every arm's module and prints the
    deduplicated dock list, split, one full-library probe of each module alone (sealed order), then every module of the
    dock list docks sideways (sealed order, sealed start marks). No zip ties. Returns (schedule, operator rows)."""
    RD, X, demo, source = demo_l3_expand(seed_hex, cfg)
    cc = cfg["choice"]
    name = RD.DEMO_ASSEMBLY
    spec = [list(m) for m in A.PHYSICAL_TYPES[name]]
    mods = [m for m, *_ in spec]
    assert tuple(mods) == tuple(RD.DEMO_MODULES)
    asm = A.build(name, [tuple(m) for m in spec], params)
    ys = {m: float(p[1]) for (m, *_), p in zip(asm.modules, asm.module_pos)}
    top = max(ys.values())
    nearest = sorted(m for m in mods if abs(ys[m] - top) < 1e-9)      # bb nearest-goal rule: largest y (bay opens in +y)
    b = Builder("choice", params, cc["trial_base"])
    rows, units, episodes = [], [], []
    frow = 0
    for e in demo:
        k = int(e["episode"]) + 1
        ep = f"D{k:02d}"
        fl = [{"wheel": f"{m}:{e['coast_wheels'][m]}", "mode": "coast"} for m in e["faulted_modules"]]
        order = list(e["dock_order"])
        link = {"episode": ep, "episode_modules": "+".join(mods), "fault_carrier": "+".join(e["faulted_modules"]),
                "episode_faults": fault_text(fl), "choice_block": k}
        instr = (f"EPISODE {ep}: BUILD {name} ({'+'.join(mods)}) in the open area; truth tag at the assembly position; "
                 f"no physical fault (firmware coast faults are injected automatically)")
        blk = b.add_block(name, spec, [dict(link, fault_id=fid_of(fl), faults=[dict(f) for f in fl], condition="DEMO:F2|F2",
                                            phase="assembly", rep=1)],
                          instruction=instr, extra={"kind": "demo", "episode": ep},
                          row={"split": False, "phase": "assembly", "choice_block": k})
        asm_trial = blk["trials"][0]["trial"]
        units.append({"episode": ep, "phase": "assembly", "module": "all", "trials": [asm_trial], "fault_carrier": "none",
                      "zip_wheels": []})
        lone = {}
        for j, m in enumerate(order):
            mf = [dict(f) for f in fl if f["wheel"].startswith(m + ":")]
            ins = (("after choose_module printed the dock list: SPLIT; " if j == 0 else "") +
                   f"ONLY {m} on the open-area start mark (other modules parked outside the arena); truth tag on {m}")
            bl = b.add_block("single", [[m, 0, 0, 0]], [dict(link, fault_id=fid_of(mf), faults=mf,
                                                              condition="DEMO:lone", phase="split", rep=1)],
                             instruction=ins, kind="operator", extra={"kind": "demo", "episode": ep, "module": m},
                             row={"split": True, "phase": "split", "choice_block": k, "module": m})
            lone[m] = bl["trials"][0]["trial"]
            units.append({"episode": ep, "phase": "split", "module": m, "trials": [lone[m]], "fault_carrier": "none",
                          "zip_wheels": []})
        pose = {m: offsets_index(cc, e["offset_m"][m]) for m in mods}
        frow += 1
        rows.append({"episode": ep, "step": "PROBE_ASSEMBLY", "fieldlog_row": frow, "module": name, "dock_slot": "",
                     "start_pose_id": "", "start_x_offset_m": "", "schedule_trial": asm_trial, "instruction": instr})
        rows.append({"episode": ep, "step": "CHOOSE", "fieldlog_row": "", "module": "", "dock_slot": "",
                     "start_pose_id": "", "start_x_offset_m": "", "schedule_trial": "",
                     "instruction": "after 'e', BEFORE the split: run choose_module (COMMANDS.txt); it logs the "
                                    "pre-split arms (sealed)"})
        for m in order:
            frow += 1
            rows.append({"episode": ep, "step": "PROBE_LONE", "fieldlog_row": frow, "module": m, "dock_slot": "",
                         "start_pose_id": "", "start_x_offset_m": "", "schedule_trial": lone[m],
                         "instruction": f"{m} alone, open area (split; the other modules parked)"})
        rows.append({"episode": ep, "step": "FINALIZE", "fieldlog_row": "", "module": "", "dock_slot": "",
                     "start_pose_id": "", "start_x_offset_m": "", "schedule_trial": "",
                     "instruction": "after the last lone probe's 'e': run choose_module --finalize; write down the DOCK "
                                    "LIST it prints (1-3 modules, in order, with start marks)"})
        for j in range(1, len(mods) + 1):
            frow += 1
            rows.append({"episode": ep, "step": "DOCK", "fieldlog_row": frow, "module": f"dock-list #{j}", "dock_slot": j,
                         "start_pose_id": "", "start_x_offset_m": "", "schedule_trial": "",
                         "instruction": f"dock the #{j} module of the printed dock list sideways into the narrow bay from "
                                        f"its start mark (choose_module prints it); SKIP this row if the list is shorter"})
        episodes.append({"episode": ep, "choice_block": k, "assembly": name, "modules": mods,
                         "faulted_modules": list(e["faulted_modules"]), "capable_module": e["capable_module"],
                         "faults": fl, "assembly_trial": asm_trial, "lone_trials": lone, "dock_order": order,
                         "offset_m": {m: float(e["offset_m"][m]) for m in mods}, "start_pose_id": pose,
                         "bb_nearest_goal": nearest,
                         "coins": {c: e[c] for c in ("random_coin", "cert_tie_coin", "carry_over_tie_coin",
                                                     "oracle_tie_coin")}})
    sched = b.schedule(cc["session_id"])
    sched["choice"] = {"episodes": episodes, "tau": cc["tau"], "alpha": cc["alpha"], "bay": dict(cc["bay"]),
                       "demo_source": source,
                       "arms": ["certificate", "carry_over", "random", "oracle"]}
    sched["link_units"] = units
    sched["realv2"] = dict(_realv2_meta(RD, X), part="demo", demo_source=source)
    return sched, rows


def offsets_index(cc, off):
    return list(cc["start_offsets_m"]).index(float(off)) + 1


def build_simval2(cfg, seed_hex, params):
    """SIMVAL day 2 (PROTOCOL_V2 §2.3/§5): M1 alone, 5 healthy + 5 of one sealed M1 wheel in its sealed MASK mode, shuffled;
    repeats the MASK lone-M1 conditions >= 1 day later."""
    RD, X = _realv2(seed_hex)
    sc = cfg["simval2"]
    b = Builder("simval2", params, sc["trial_base"])
    tl = []
    for t in X["simval_day2"]:
        fl = [] if t["wheel"] is None else [{"wheel": f"M1:{t['wheel']}", "mode": t["firmware"]}]
        tl.append({"fault_id": fid_of(fl), "faults": fl, "condition": "SIMVAL2", "realv2_seq": int(t["seq"])})
    b.add_block("single", [["M1", 0, 0, 0]], tl, instruction="M1 alone at the MASK start mark; firmware faults automated")
    sched = b.schedule(sc["session_id"])
    sched["realv2"] = dict(_realv2_meta(RD, X), part="simval_day2")
    return sched


def build_tagstatic(cfg, seed_hex, params):
    """AprilTag static check (PROTOCOL_V2 §5; card 05 C) in the sealed pose order: 5 static trials, no probe runner."""
    RD, X = _realv2(seed_hex)
    sc = cfg["tagstatic"]
    b = Builder("tagstatic", params, sc["trial_base"])
    order = [int(k) + 1 for k in X["apriltag_pose_order"]]
    tl = [{"fault_id": "F0", "faults": [], "condition": f"STATIC_POSE{k}"} for k in order]
    b.add_block("single", [["M1", 0, 0, 0]], tl,
                instruction="STATIC: do NOT run the probe runner. Rows in order = reference poses " + ", ".join(map(str, order))
                            + "; place M1 on that pose, start the row, 15 s hands off, end it")
    sched = b.schedule(sc["session_id"])
    sched["static"] = True
    sched["realv2"] = dict(_realv2_meta(RD, X), part="apriltag", pose_order=order)
    return sched


def _subst(x, mapping):
    if isinstance(x, str):
        for k, v in mapping.items():
            x = x.replace("{" + k + "}", str(v))
        return x
    if isinstance(x, list):
        return [_subst(v, mapping) for v in x]
    if isinstance(x, dict):
        return {k: _subst(v, mapping) for k, v in x.items()}
    return x


def expand_block(bs, rng, fixed_rng, params, fixed_cache, tag):
    """One config block -> [(assembly, spec, trials, fault_id, instruction, extra)] (several for mix / per_wheel / natural)."""
    name = bs["assembly"]
    spec = bs.get("modules") or A.PHYSICAL_TYPES[name]
    asm = A.build(name, [tuple(m) for m in spec], params)
    wheels = list(asm.wheel_corner)
    mods = [m for m, *_ in asm.modules]
    if bs.get("per_wheel"):
        out = []
        for w in wheels:
            b2 = _subst({k: v for k, v in bs.items() if k != "per_wheel"}, {"wheel": w})
            out += expand_block(b2, rng, fixed_rng, params, fixed_cache, tag + ":" + w)
        return out
    if "mix" in bs:
        cells = [f for f, c in bs["mix"].items() for _ in range(int(c))]
        cells = [cells[int(i)] for i in rng.permutation(len(cells))]
        size = int(bs.get("block_size", 20))
        out = []
        for k in range(0, len(cells), size):
            trials = []
            for f in cells[k:k + size]:
                fl, fid = firmware_faults(rng, f, wheels, None)
                trials.append({"fault_id": fid, "faults": fl})
            out.append((name, spec, trials, None, bs.get("instruction", "automated: no physical change"), {}))
        return out
    n = int(bs.get("n", 1))

    def fixed(key, choices):
        k = (tag, key)
        if k not in fixed_cache:
            fixed_cache[k] = str(fixed_rng.choice(choices))
        return fixed_cache[k]
    if "natural" in bs:
        nk = bs["natural"]
        if nk == "low_battery":
            m = str(rng.choice(mods))
            where, w = f"module {m}", "all"
        else:
            w = str(rng.choice(wheels))
            m, where = w.split(":")[0], f"wheel {w}"
        nat = (name, spec, [{"fault_id": "F0", "faults": [], "condition": f"N:{nk}"} for _ in range(n)], f"N_{nk}",
               f"NATURAL FAULT '{nk}' (card 06): " + NATURAL_TEXT[nk].format(w=w, m=m) + "; firmware healthy",
               {"kind": "natural", "natural": nk, "where": w})
        out = [nat]
        if bs.get("pair_healthy"):
            hea = (name, spec, [{"fault_id": "F0", "faults": [], "condition": f"N:healthy_pair:{nk}"}
                                for _ in range(int(bs["pair_healthy"]))], "N_healthy",
                   f"paired HEALTHY block for '{nk}' on {where}: condition removed, same module, same pack",
                   {"kind": "natural_healthy", "natural": nk, "where": w})
            out = [nat, hea] if rng.random() < 0.5 else [hea, nat]
        return out
    tspecs = bs.get("trials") or [bs.get("trial", {})] * n
    if bs.get("trials") is None:
        tspecs = [bs.get("trial", {})] * n
    block_wheel = {}
    trials = []
    for ts in tspecs:
        ts = ts or {}
        wsel = ts.get("wheel", "random")
        if wsel == "random":
            chooser = None
        elif wsel == "block":
            chooser = lambda: block_wheel.setdefault("w", str(rng.choice(wheels)))     # noqa: E731
        elif wsel == "fixed":
            chooser = lambda: fixed("fw_wheel", wheels)                                   # noqa: E731
        else:
            chooser = lambda w=wsel: w                                                    # noqa: E731
        fl, fid = firmware_faults(rng, ts.get("firmware", "ok"), wheels, chooser)
        trials.append({"fault_id": fid, "faults": fl})
    op = bs.get("operator")
    fid_block, extra, instr = None, {}, bs.get("instruction", "automated: no physical change")
    if op == "roller_locked":
        ow = bs.get("op_wheels", "block")
        if ow == "block":
            lw = [str(rng.choice(wheels))]
        elif ow == "fixed":
            lw = [fixed("op_wheel", wheels)]
        elif ow == "diag":
            m = str(rng.choice(mods))
            lw = [f"{m}:fl", f"{m}:rr"] if rng.random() < 0.5 else [f"{m}:fr", f"{m}:rl"]
        else:
            lw = list(ow)
        for t in trials:
            t["faults"] = [{"wheel": w, "mode": "roller_locked"} for w in lw] + t["faults"]
            t["fault_id"] = "F4" if t["fault_id"] == "F0" else "F5"
        fid_block = "F4"
        instr = bs.get("instruction", "zip-tie the rollers of wheel " + " AND ".join(lw)).replace("{op_wheel}", " AND ".join(lw))
        extra = {"kind": "roller_locked", "wheels": lw}
    elif op == "payload":
        kg = round(float(rng.uniform(*bs.get("payload_kg", [4.0, 6.0]))), 2)
        r = round(float(rng.uniform(*bs.get("offset_m", [0.08, 0.12]))), 3)
        if kg > 6.0 or r > 0.12:
            raise ValueError("payload outside the safety envelope (> 6 kg or > 12 cm)")
        pm = bs.get("payload_module", mods[0])
        corner = str(rng.choice(PL.CORNERS))
        for t in trials:
            t["faults"] = [{"wheel": "none", "mode": "payload", "module": pm, "mass_kg": kg}] + t["faults"]
            t["fault_id"], t["ood"] = "F6", "payload"
        fid_block = "F6"
        instr = bs.get("instruction", f"strap {kg} kg on the lifter deck of {pm} (lifter stowed), CG {100 * r:.1f} cm from "
                                      f"the module centre toward its {corner} corner" +
                       ("; firmware adds wheel faults" if any(len(t["faults"]) > 1 for t in trials) else ""))
        extra = {"kind": "payload", "mass_kg": kg, "offset_m": r, "corner": corner, "module": pm}
    elif op == "low_mu":
        mu = round(float(rng.uniform(*bs.get("mu", [0.12, 0.35]))), 3)
        for t in trials:
            t["faults"] = [{"wheel": "none", "mode": "low_mu", "region": "left", "mu": mu}] + t["faults"]
            t["fault_id"], t["ood"] = "F7", "low_mu"
        fid_block = "F7"
        instr = bs.get("instruction", f"low-friction sheet (target mu {mu:.2f}; use the nearest measured sheet), edge under the "
                                      f"centre line, covering the LEFT wheels" +
                       ("; firmware adds wheel faults" if any(len(t["faults"]) > 1 for t in trials) else ""))
        extra = {"kind": "low_mu", "mu": mu}
    if bs.get("condition"):
        for t in trials:
            t["condition"] = bs["condition"]
    return [(name, spec, trials, fid_block, instr, extra)]


def build_extra(kind, cfg, seed_hex, params, module="M1", day=1):
    kc = cfg["kinds"][kind]
    s63 = int(seed_hex, 16) % (2 ** 63)
    salt = PL.MODULES.index(module) if kc.get("per_module") else 0
    rng = np.random.default_rng([s63, int(day) if kc.get("per_day") else 1, salt])
    fixed_rng = np.random.default_rng([s63, 99])                 # draws shared by every day / module of this kind
    base = int(kc["trial_base"]) + (1000 * salt if kc.get("per_module") else 0) + (500 * (int(day) - 1) if kc.get("per_day") else 0)
    sub = {"module": module, "day": day}
    blocks_cfg = _subst(kc["blocks"], sub)
    groups = []
    fixed_cache = {}
    for i, bs in enumerate(blocks_cfg):
        g = expand_block(bs, rng, fixed_rng, params, fixed_cache, f"b{i}")
        if bs.get("probes"):                                  # WS-P2: block runs a probe subset (e.g. P1+P5+P2)
            pids = list(SCH.trial_pids({"probes": bs["probes"]}))
            for x in g:
                for t in x[2]:
                    t["probes"] = pids
        groups.append((bs["assembly"], g, "mix" in bs))
    if kc.get("order", "fixed") == "shuffle":
        types = list(dict.fromkeys(t for t, _, _ in groups))
        types = [types[int(i)] for i in rng.permutation(len(types))]
        flat = []
        for t in types:
            units = []
            for tt, g, split in groups:                       # a mix splits into independent blocks; a natural
                if tt == t:                                   # block and its paired healthy block stay together
                    units += [[x] for x in g] if split else [g]
            for j in rng.permutation(len(units)):
                flat += units[int(j)]
    else:
        flat = [x for _, g, _ in groups for x in g]
    b = Builder(kind, params, base)
    for name, spec, trials, fid, instr, extra in flat:
        b.add_block(name, spec, trials, fault_id=fid, instruction=instr, extra=extra,
                    kind="operator" if (fid or "").startswith(("F4", "F6", "F7", "N_")) else None)
    sched = b.schedule(_subst(kc["session_id"], sub))
    if kc.get("static"):
        sched["static"] = True
    return sched


# ------------------------------------------------------------------------------------------------ writing
def canonical(sched):
    return json.dumps(sched, sort_keys=True, separators=(",", ":")).encode()


def write_schedule(out, sched, seed_hex, force):
    if (out / "schedule.json").exists() and not force:
        old = json.loads((out / "schedule.json").read_text())
        if canonical(old) != canonical(sched):
            sys.exit(f"{out}/schedule.json exists and differs from what this seed generates. Refusing to overwrite a "
                     f"(possibly sealed) schedule; use a new OUT_DIR, or --force if it was never used.")
    raw = canonical(sched)
    (out / "schedule.json").write_bytes(raw)
    h = hashlib.sha256(raw).hexdigest()
    (out / "schedule.sha256").write_text(f"{h}  schedule.json\n")
    commit = {"schedule_sha256": h, "seed_sha256": hashlib.sha256(seed_hex.encode()).hexdigest(), "schema": SCH.SCHEMA}
    (out / "commit.json").write_text(json.dumps(commit, indent=1) + "\n")
    return h


def asm_diagram(spec):
    cells = {(int(i), int(j)): m for m, i, j, _ in spec}
    mi, mj = max(i for i, _ in cells), max(j for _, j in cells)
    lines = []
    for i in range(mi, -1, -1):
        lines.append("    " + " ".join(f"[{cells[(i, j)]}]" if (i, j) in cells else "    " for j in range(mj, -1, -1)))
    return "\n".join(lines) + "\n    (front of every module = up the page = assembly +x; assembly +y = left)"


def render_config(values):
    cfg = FC.render(FC.load_config(FC.CONFIG_PATH), values)
    raw = yaml.safe_dump(cfg, sort_keys=False, width=140)
    return raw, hashlib.sha256(raw.encode()).hexdigest()


def run_rows(sched, fs):
    """[(row dict, assembly, block row, trial)] of one field session in run order (FieldLog run-sheet layout)."""
    byb = PL.blocks_by_key(sched)
    out, k = [], 0
    for r in fs["blocks"]:
        for t in byb[(r["assembly"], r["block"])]["trials"]:
            k += 1
            out.append((PL.runsheet_row(k, t["fault_id"], r["pos"]), r["assembly"], r, t))
    return out


def hours(n_trials, n_builds, rates):
    tph, rb = rates["trials_per_hour"], rates["rebuild_min"]
    return n_trials / tph[0] + n_builds * rb[0] / 60, n_trials / tph[1] + n_builds * rb[1] / 60


LINK_COLS = ("episode", "phase", "round", "rep", "episode_modules", "fault_carrier", "episode_faults", "choice_block",
             "r3_split", "realv2_seq")


def sched_hours(sched, rates, extra=(0.0, 0.0)):
    """(lo, hi) robot hours of a schedule: each trial at the full or short-probe rate, each rebuild to a multi-module
    layout at rebuild_min, each split set-up (one module alone, masking) at split_min."""
    tph, stph = rates["trials_per_hour"], rates.get("short_trials_per_hour", rates["trials_per_hour"])
    rb, sp = rates["rebuild_min"], rates.get("split_min", rates["rebuild_min"])
    byb = PL.blocks_by_key(sched)
    lo = hi = 0.0
    for fs in PL.field_sessions(sched):
        for r in fs["blocks"]:
            for t in byb[(r["assembly"], r["block"])]["trials"]:
                short = SCH.trial_pids(t) != SCH.PIDS
                lo += 1 / (stph[0] if short else tph[0])
                hi += 1 / (stph[1] if short else tph[1])
            if r["reassemble"]:
                m = sp if r.get("split") else rb
                lo += m[0] / 60
                hi += m[1] / 60
    return lo + extra[0], hi + extra[1]


def write_all(out, sched, a, kind, cfg):
    params = A.load_params(PARAMS)
    ids = parse_kv(a.ids, {m: m.lower() for m in PL.MODULES})
    vcodes = parse_kv(a.vehicle_codes, {m: str(k + 1) for k, m in enumerate(PL.MODULES)})
    cb_path = FC.overlay_codebook(out / "codebook_overlay")
    ops = (a.operator_codes or "<OPA>,<OPB>").split(",")
    data = a.data_dir
    mode = a.fieldlog or cfg["main"].get("fieldlog", "per_session")
    static = bool(sched.get("static"))
    sheets, trial_rows, summary = [], [], []
    byb = PL.blocks_by_key(sched)
    for t_s in sched["sessions"]:
        for blk in t_s["blocks"]:
            for t in blk["trials"]:
                trial_rows.append({"trial": t["trial"], "assembly": t_s["assembly"],
                                   "modules": "+".join(m[0] for m in t_s["modules"]), "block": blk["block"],
                                   "fault_id": t["fault_id"], "condition": t.get("condition", t["fault_id"]),
                                   "ood": t.get("ood", "none"),
                                   "faults": ";".join(f"{f['wheel']}:{f['mode']}" + "".join(f":{k}={f[k]}" for k in
                                                      ("g", "mass_kg", "mu") if k in f) for f in t["faults"]),
                                   "probes": "+".join(SCH.trial_pids(t)),
                                   **{k: t.get(k, "") for k in LINK_COLS}})
    all_mods = sorted({m[0] for s in sched["sessions"] for m in s["modules"]})
    vals = {m: ids[m] for m in PL.MODULES}
    vals.update(LEAD=ids[all_mods[0]], IMU_MODULE=ids[all_mods[0]], LIDAR_MODULE=ids[all_mods[0]], TRUTH_ID=a.truth_id)
    raw_cfg, cfg_sha = render_config(vals)
    for si, fs in enumerate(PL.field_sessions(sched), start=1):
        sid = fs["id"]
        sd = out / sid
        sd.mkdir(parents=True, exist_ok=True)
        if hasattr(SCH, "write_block_sheet") and "field_sessions" in sched:
            SCH.write_block_sheet(sd / f"blocks_{sid}.csv", sched, sid)
        rows = run_rows(sched, fs)
        segs = PL.segments(sched, fs)
        units = [(f"log", rows, None)] if mode == "per_session" else \
            [(f"seg{s['seg']:02d}_{s['assembly']}", [x for x in rows if x[2]["pos"] in {b['pos'] for b in s['blocks']}], s)
             for s in segs]
        card = [f"# Session card {sid} ({kind})", "",
                f"Schedule sha256 `{(out / 'schedule.sha256').read_text().split()[0]}` · {len(rows)} trials · "
                f"{len(segs)} build(s) including the first · FieldLog: {mode.replace('_', ' ')}.", "",
                "Operators: A ______ (logger, trial tool)  B ______ (e-stop, video).  Date ______  Floor ______", "",
                "Before the first build: OPERATOR_CHECKLIST.md part 1. Phone video running continuously from now on.", ""]
        for uname, urows, seg in units:
            ud = sd / uname
            ud.mkdir(parents=True, exist_ok=True)
            k0 = urows[0][0]["row"] - 1
            rs_rows = []
            for r, *_ in urows:
                r = dict(r, row=r["row"] - k0, nn=r["nn"] - k0, trial_id=f"gnd_capxprobe_{r['row'] - k0:03d}")
                rs_rows.append(r)
            rs = PL.write_runsheet(ud / "runsheet.csv", rs_rows)
            rs_sha = PL.sha256_file(rs)
            (ud / "fieldlog_config.yaml").write_text(raw_cfg)
            asms = list(dict.fromkeys(x[1] for x in urows))
            ordinal = (100 * si + (seg["seg"] if seg else 0)) if kind == "main" else \
                ORDINAL_BASE.get(kind, 2900) + 10 * (si - 1) + (seg["seg"] if seg else 0)
            lead = PL.session_of(sched, asms[0])["modules"][0][0]
            plan = {"study_codes": [int(a.study_code)], "session_ordinal": int(ordinal), "session_role_code": 1,
                    "vehicle_code": int(vcodes[lead]), "vehicle_config_code": int(PL.ASM_CODE.get(asms[0], 9)) if len(asms) == 1 else 9,
                    "prereg_sha256": a.prereg_sha256, "runsheet_path": str(rs.resolve()), "runsheet_sha256": rs_sha,
                    "capture_tier_code": 3, "gps_vel_frame_code": 9, "timesync_mode_code": 0,
                    "holdout_vehicle_config_codes": [], "holdout_site_codes": [], "split_m": {}}
            (ud / "plan.yaml").write_text("# FieldLog v2 plan, generated by make_run_sheets.py (do not edit; regenerate)\n"
                                          + yaml.safe_dump(plan, sort_keys=False))
            members = sorted({m[0] for x in urows for m in PL.session_of(sched, x[1], x[2]["block"])["modules"]})
            idlist = ",".join(f"{m}={ids[m]}" for m in members)
            pos = sorted({x[2]["pos"] for x in urows})
            cmds = [f"# {sid} {uname}: FieldLog run-sheet rows 1-{len(urows)}, block positions {pos[0]}-{pos[-1]}, "
                    f"types {', '.join(asms)}",
                    "# Terminal A, logger (runs for the whole unit; Ctrl-C closes the FieldLog session cleanly):",
                    "source /opt/ros/jazzy/setup.bash",
                    f"python3 {PL.FIELDLOG}/ros2_csvlog_v2.py --config {ud.resolve()}/fieldlog_config.yaml "
                    f"--plan {ud.resolve()}/plan.yaml --out {data} --operator-code {ops[0]}:1 --operator-code {ops[-1]}:2 "
                    f"--site-code {a.site_code} --field --codebook {cb_path.resolve()}",
                    "# Terminal B, trial tool (s = start a row BEFORE Enter in the runner; e = end AFTER the runner prints the end):",
                    f"python3 {PL.FIELDLOG}/csvlog_trial.py {data}/<NEW_SESSION_FOLDER> --runsheet {rs.resolve()} "
                    f"--codebook {cb_path.resolve()}"]
            if static:
                cmds.append("# Terminal C: NO probe runner (static trials; see the session card)")
            else:
                cmds += ["# Terminal C, probe runner: walks the blocks in order, prompts every rebuild and block change, relays",
                         "# the active IMU/rf2o (resume after a stop: add --from-block POS or --from-trial N):",
                         "source /opt/ros/jazzy/setup.bash",
                         f"PYTHONPATH={PL.REPO} python3 -m capxfer.field.probe_runner {out.resolve()}/schedule.json "
                         f"--session {sid} --ids {idlist} --confirm" + (f" --from-block {pos[0]} --to-block {pos[-1]}" if seg else ""),
                         "# dry run of the same (no ROS):",
                         f"PYTHONPATH={PL.REPO} python3 -m capxfer.field.probe_runner {out.resolve()}/schedule.json "
                         f"--session {sid} --dry-run" + (f" --from-block {pos[0]} --to-block {pos[-1]}" if seg else "")]
            (ud / "COMMANDS.txt").write_text("\n".join(cmds) + "\n")
            sheets.append({"sched_session": sid, "unit": uname, "assemblies": "+".join(asms), "n_trials": len(urows),
                           "first_pos": pos[0], "last_pos": pos[-1], "runsheet_sha256": rs_sha, "config_sha256": cfg_sha,
                           "session_ordinal": ordinal, "dir": str(ud.relative_to(out)),
                           "trial_order": " ".join(str(x[3]["trial"]) for x in urows)})
        card.append(f"FieldLog: start the logger and trial tool **once** from `{units[0][0]}/COMMANDS.txt`"
                    if mode == "per_session" else "FieldLog: a new logger session at every build (`segNN_*/COMMANDS.txt`).")
        card.append("")
        for s in segs:
            spec = PL.session_of(sched, s["assembly"], s["first_block"])["modules"]
            card += [f"## Build {s['seg']}: **{s['assembly']}**", "", "```", asm_diagram(spec), "```", "",
                     "- [ ] Rebuilt (runner prompt), latches seated, play ____ mm; tag offset → session_index.csv (row per build)",
                     "- [ ] Lowest pack ____ V (swap < 11.1 V)", "",
                     "| ✓ | pos | block | run-sheet rows | schedule trials | condition / instruction |", "|---|---|---|---|---|---|"]
            for bl in s["blocks"]:
                rr = [x[0]["row"] for x in rows if x[2]["pos"] == bl["pos"]]
                tn = [t["trial"] for t in bl["trials"]]
                card.append(f"| ☐ | {bl['pos']} | {bl['block']} | {min(rr)}–{max(rr)} | {min(tn)}–{max(tn)} | "
                            f"{bl['fault_id'] or 'auto'}: {bl['condition'].get('instruction', '')} |")
            card.append("")
        card += ["- [ ] End: Ctrl-C the logger, validate (OPERATOR_CHECKLIST part 3), block_log.csv complete", ""]
        (sd / "SESSION_CARD.md").write_text("\n".join(card) + "\n")
        summary.append((sid, len(rows), len(segs), sorted({s["assembly"] for s in segs})))
    with open(out / "fieldlog_runsheets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheets[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(sheets)
    with open(out / "schedule_trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trial_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(trial_rows)
    rates = cfg["main"]
    tot = sum(x[1] for x in summary)
    builds = sum(x[2] for x in summary)
    short = any(SCH.trial_pids(t) != SCH.PIDS for _, _, t in SCH.iter_trials(sched))
    if short:                      # WS-P2: per-trial rates (probe subsets) and split set-ups
        fsd = {f["id"]: f for f in PL.field_sessions(sched)}
        hours = lambda n, nb, r, sid=None: sched_hours(dict(sched, field_sessions=[fsd[sid]]) if sid else sched, r)  # noqa
    else:
        hours = globals()["hours"]
    lo, hi = hours(tot, builds, rates)
    tph, rb = rates["trials_per_hour"], rates["rebuild_min"]
    lines = [f"# Run-sheet summary ({kind})", "",
             f"Generated by `make_run_sheets.py {kind}` from `{a.seed_file}` (seed sha256 in commit.json), packet_config sha256 "
             f"`{PL.sha256_file(CONFIG)[:12]}`. Schedule sha256 `{(out / 'schedule.sha256').read_text().split()[0]}`.", "",
             f"| session | trials | builds (incl. the first) | types | probing h @{tph[0]}/h | total h ({tph[0]}/h + {rb[0]} min/build … "
             f"{tph[1]}/h + {rb[1]} min/build) |", "|---|---|---|---|---|---|"]
    for sid, n, nb, ty in summary:
        h0, h1 = hours(n, nb, rates, sid) if short else hours(n, nb, rates)
        lines.append(f"| {sid} | {n} | {nb} | {', '.join(ty)} | {n / tph[0]:.2f} | {h0:.2f} – {h1:.2f} |")
    if short:
        lines += ["", f"**Total: {tot} trials, {builds} builds (assembly rebuilds + split set-ups) = {lo:.1f}–{hi:.1f} h** "
                      f"(trial pace and minutes per rebuild / split are TEAM-CONFIRM).", ""]
    else:
        lines += ["", f"**Total: {tot} trials, {builds} builds, {tot / tph[0]:.1f} h of probing + {builds * rb[0] / 60:.1f}–"
                  f"{builds * rb[1] / 60:.1f} h of rebuilding = {lo:.1f}–{hi:.1f} h** (trial pace and minutes per rebuild are TEAM-CONFIRM).", ""]
    if kind == "main":
        alt = "segment" if sched.get("session_order", "block") == "block" else "block"
        try:
            s2 = SCH.generate(read_seed(a.seed_file), session_order=alt)
            b2 = sum(f["n_reassemblies"] for f in s2["field_sessions"])
            l2, h2 = hours(s2["n_trials"], b2, rates)
            lines += [f"Session order used: **{sched.get('session_order')}** (PENDING OWNER DECISION, packet_config.yaml "
                      f"`main.session_order`). The other variant, `{alt}`, with the same seed: {s2['n_trials']} trials, {b2} "
                      f"builds, {l2:.1f}–{h2:.1f} h.", ""]
        except TypeError:
            pass
    lines.append(f"Trial duration {Schedule(A.physical('single', params).char_length()).T:.1f} s + 3 s gap.")
    if short:
        stph, sp = rates.get("short_trials_per_hour", tph), rates.get("split_min", rb)
        lines += ["", f"**Probe-subset trials (WS-P2):** {'+'.join(SCH.trial_pids(next(SCH.iter_trials(sched))[2]))} "
                      f"({Schedule(A.physical('single', params).char_length(), pids=SCH.SHORT_PIDS).T:.1f} s) at "
                      f"{stph[0]}–{stph[1]} trials/h; a split set-up (one module alone) takes {sp[0]}–{sp[1]} min, a rebuild "
                      f"to an assembly {rb[0]}–{rb[1]} min. The total-hours column and the bold total use these rates "
                      f"(the probing column assumes full-library trials)."]
    if sched.get("episodes"):
        with open(out / "episodes.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["episode", "fault", "fault_carrier", "r3_split", "parts", "trials"])
            for e in sched["episodes"]:
                w.writerow([e["episode"], fault_text(e["faults"]), e["fault_carrier"], e.get("r3_split", ""),
                            "+".join(e["parts"]), ";".join(f"{k}:" + " ".join(map(str, v)) for k, v in e["parts"].items())])
        lines += ["", f"Links: `episodes.csv` ({len(sched['episodes'])} (module, wheel) conditions; each condition's trials in "
                      f"every parent assembly and on the module alone). One `manual/episode_log.csv` row per build."]
    if sched.get("realv2"):
        lines += ["", f"PROTOCOL_V2 seed commitment `{sched['realv2']['commitment']}`; realv2 expansion sha256 "
                      f"`{sched['realv2']['realv2_schedule_sha256']}` (both must equal the freeze record)."]
        commit = json.loads((out / "commit.json").read_text())
        commit.update(realv2_commitment=sched["realv2"]["commitment"],
                      realv2_schedule_sha256=sched["realv2"]["realv2_schedule_sha256"])
        (out / "commit.json").write_text(json.dumps(commit, indent=1) + "\n")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return tot, builds, lo, hi


def docking(out, seed_hex, a, cfg):
    """Sealed docking schedule (DOCKING_PROTOCOL §5): blocks of all arms + blocks of two arms, one start pose per block;
    the operator sees arm CODES only, the code->arm key goes to the sealed folder."""
    dc = cfg["docking"]
    rng = np.random.default_rng(int(seed_hex, 16) % (2 ** 63))
    arms = list(dc["arms"]) + (["naive_detector"] if a.naive_arm else [])
    codes = "ABCDE"[:len(arms)]
    perm = [arms[int(i)] for i in rng.permutation(len(arms))]
    key = dict(zip(codes, perm))
    inv = {v: k for k, v in key.items()}
    offsets = dc["start_offsets_m"]
    rows, run, row = [], 0, 0
    nblock = int(dc["blocks_all_arms"]) + int(dc["blocks_two_arms"])
    for blk in range(1, nblock + 1):
        sel = arms if blk <= int(dc["blocks_all_arms"]) else list(dc["two_arms"])
        order = [sel[int(i)] for i in rng.permutation(len(sel))]
        pose = int(rng.integers(len(offsets))) + 1
        row += 1
        rows.append({"row": row, "block": blk, "run_in_block": 0, "run_global": "", "arm_code": "", "start_pose_id": "",
                     "start_x_offset_m": "", "what": "PROBE"})
        for j, arm in enumerate(order, start=1):
            run += 1
            row += 1
            rows.append({"row": row, "block": blk, "run_in_block": j, "run_global": run, "arm_code": inv[arm],
                         "start_pose_id": pose, "start_x_offset_m": offsets[pose - 1], "what": "DOCK"})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dock_runsheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    params = A.load_params(PARAMS)
    b = Builder("docking", params, dc["trial_base"])
    lock = [{"wheel": w, "mode": "roller_locked"} for w in dc["probe_faults"]]
    for blk in range(1, nblock + 1):
        b.add_block(dc["assembly"], A.PHYSICAL_TYPES[dc["assembly"]], [{"fault_id": "F4", "condition": "F4x2d", "faults": list(lock)}],
                    fault_id="F4", instruction="docking block-start probe: full probe library in the open area")
    sched = b.schedule("DOCK")
    write_schedule(out, sched, seed_hex, a.force)
    ud = out / "DOCK" / "log"
    ud.mkdir(parents=True, exist_ok=True)
    fl = []
    for r in rows:
        fl.append(PL.runsheet_row(r["row"], "F4", r["block"]))
        fl[-1]["arm_code"] = " ABCDE".index(r["arm_code"]) if r["arm_code"] else 0
    rs = PL.write_runsheet(ud / "runsheet.csv", fl)
    ids = parse_kv(a.ids, {m: m.lower() for m in PL.MODULES})
    raw, cfg_sha = render_config({**{m: ids[m] for m in PL.MODULES}, "LEAD": ids["M1"], "IMU_MODULE": ids["M1"],
                                  "LIDAR_MODULE": ids["M1"], "TRUTH_ID": a.truth_id})
    (ud / "fieldlog_config.yaml").write_text(raw)
    plan = {"study_codes": [13, int(a.study_code)], "session_ordinal": ORDINAL_BASE["docking"] + 1, "session_role_code": 1,
            "vehicle_code": 1, "vehicle_config_code": 1, "prereg_sha256": a.prereg_sha256, "runsheet_path": str(rs.resolve()),
            "runsheet_sha256": PL.sha256_file(rs), "capture_tier_code": 3, "gps_vel_frame_code": 9, "timesync_mode_code": 0,
            "holdout_vehicle_config_codes": [], "holdout_site_codes": [], "split_m": {}}
    (ud / "plan.yaml").write_text("# FieldLog v2 plan (docking), generated by make_run_sheets.py\n" + yaml.safe_dump(plan, sort_keys=False))
    cb_path = FC.overlay_codebook(out / "codebook_overlay")
    ops = (a.operator_codes or "<OPA>,<OPB>").split(",")
    base = int(dc["trial_base"]) - 1
    (ud / "COMMANDS.txt").write_text("\n".join([
        "# Docking session (card 04). One FieldLog row per block-start probe AND per docking run (rows outside a FieldLog",
        "# trial are dropped by the logger): start a row before every probe and every dock run.",
        "source /opt/ros/jazzy/setup.bash",
        f"python3 {PL.FIELDLOG}/ros2_csvlog_v2.py --config {ud.resolve()}/fieldlog_config.yaml --plan {ud.resolve()}/plan.yaml "
        f"--out {a.data_dir} --operator-code {ops[0]}:1 --operator-code {ops[-1]}:2 --site-code {a.site_code} --field "
        f"--codebook {cb_path.resolve()}",
        f"python3 {PL.FIELDLOG}/csvlog_trial.py {a.data_dir}/<NEW_SESSION_FOLDER> --runsheet {rs.resolve()} --codebook {cb_path.resolve()}",
        "# block-start probe of block k (k = 1.." + str(nblock) + "):",
        f"PYTHONPATH={PL.REPO} python3 -m capxfer.field.probe_runner {out.resolve()}/schedule.json --assembly {dc['assembly']} "
        f"--from-trial $(({base} + k)) --to-trial $(({base} + k)) --ids M1={ids['M1']} --confirm",
        "# docking run: start the dock executor with the arm CODE of the sheet (executor: WS-F, TEAM-CONFIRM; not in this packet)",
    ]) + "\n")
    with open(out / "fieldlog_runsheets.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["sched_session", "unit", "assemblies", "n_trials", "first_pos", "last_pos", "runsheet_sha256",
                    "config_sha256", "session_ordinal", "dir", "trial_order"])
        w.writerow(["DOCK", "log", dc["assembly"], len(rows), 1, nblock, PL.sha256_file(rs), cfg_sha, plan["session_ordinal"],
                    "DOCK/log", " ".join(str(base + int(r["block"])) for r in rows if r["what"] == "PROBE")])
    sealed = Path(a.sealed_dir) if a.sealed_dir else Path(a.seed_file).resolve().parent
    sealed.mkdir(parents=True, exist_ok=True)
    kp = sealed / "dock_arm_key.csv"
    with open(kp, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["arm_code", "arm"])
        for c in codes:
            w.writerow([c, key[c]])
    commit = json.loads((out / "commit.json").read_text())
    commit.update(dock_runsheet_sha256=PL.sha256_file(out / "dock_runsheet.csv"), dock_arm_key_sha256=PL.sha256_file(kp))
    (out / "commit.json").write_text(json.dumps(commit, indent=1) + "\n")
    print(f"docking: {run} runs in {nblock} blocks; arm key sealed in {kp} (do not open before the analysis)")
    return 0


def choice(out, seed_hex, a, cfg):
    """Sealed DEMO (card 10; SPINE D15): schedule.json (sealed: faulted modules, coins, orders, marks; operators follow the
    sheet, not this file), choice_runsheet.csv (operator sheet per episode: PROBE_ASSEMBLY, CHOOSE, PROBE_LONE x3,
    DOCK slots x3 = the printed dock list), one FieldLog run sheet with a row per probe AND per dock slot (dock rows:
    trial_order 0; unused slots are simply not run), COMMANDS.txt, SESSION_CARD.md. No arm identity on any sheet."""
    params = A.load_params(PARAMS)
    cc = cfg["choice"]
    sched, rows = build_choice(cfg, seed_hex, params)
    out.mkdir(parents=True, exist_ok=True)
    h = write_schedule(out, sched, seed_hex, a.force)
    with open(out / "choice_runsheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    sid = cc["session_id"]
    ud = out / sid / "log"
    ud.mkdir(parents=True, exist_ok=True)
    trial_fid = {t["trial"]: t["fault_id"] for _, _, t in SCH.iter_trials(sched)}
    blk_of = {t["trial"]: r["pos"] for f_ in PL.field_sessions(sched) for r in f_["blocks"]
              for t in PL.blocks_by_key(sched)[(r["assembly"], r["block"])]["trials"]}
    fl, order = [], []
    for r in rows:
        if r["fieldlog_row"] == "":
            continue
        if r["step"].startswith("PROBE"):
            fl.append(PL.runsheet_row(int(r["fieldlog_row"]), trial_fid[r["schedule_trial"]], blk_of[r["schedule_trial"]]))
            order.append(str(r["schedule_trial"]))
        else:
            fl.append(PL.runsheet_row(int(r["fieldlog_row"]), "F0", 0))
            fl[-1]["arm_code"] = 0
            order.append("0")
    rs = PL.write_runsheet(ud / "runsheet.csv", fl)
    ids = parse_kv(a.ids, {m: m.lower() for m in PL.MODULES})
    mods = sched["choice"]["episodes"][0]["modules"]
    nm = len(mods)
    lead = mods[0]
    raw, cfg_sha = render_config({**{m: ids[m] for m in PL.MODULES}, "LEAD": ids[lead], "IMU_MODULE": ids[lead],
                                  "LIDAR_MODULE": ids[lead], "TRUTH_ID": a.truth_id})
    (ud / "fieldlog_config.yaml").write_text(raw)
    plan = {"study_codes": [int(a.study_code)], "session_ordinal": ORDINAL_BASE["choice"] + 1, "session_role_code": 1,
            "vehicle_code": 1, "vehicle_config_code": 9, "prereg_sha256": a.prereg_sha256,
            "runsheet_path": str(rs.resolve()), "runsheet_sha256": PL.sha256_file(rs), "capture_tier_code": 3,
            "gps_vel_frame_code": 9, "timesync_mode_code": 0, "holdout_vehicle_config_codes": [], "holdout_site_codes": [],
            "split_m": {}}
    (ud / "plan.yaml").write_text("# FieldLog v2 plan (DEMO, card 10), generated by make_run_sheets.py\n"
                                  + yaml.safe_dump(plan, sort_keys=False))
    cb_path = FC.overlay_codebook(out / "codebook_overlay")
    ops = (a.operator_codes or "<OPA>,<OPB>").split(",")
    idlist = ",".join(f"{m}={ids[m]}" for m in mods)
    neps = len(sched["choice"]["episodes"])
    bundle = a.bundle or "<CHOICE_BUNDLE_DIR from the freeze record>"
    (ud / "COMMANDS.txt").write_text("\n".join([
        f"# DEMO (card 10): {neps} episodes. One FieldLog row per probe (L3, then each module alone) AND per dock-list",
        "# slot (up to 3; skip unused slots): start a row ('s') before each, end it ('e') after.",
        "# Terminal A, logger (whole session):",
        "source /opt/ros/jazzy/setup.bash",
        f"python3 {PL.FIELDLOG}/ros2_csvlog_v2.py --config {ud.resolve()}/fieldlog_config.yaml --plan {ud.resolve()}/plan.yaml "
        f"--out {a.data_dir} --operator-code {ops[0]}:1 --operator-code {ops[-1]}:2 --site-code {a.site_code} --field "
        f"--codebook {cb_path.resolve()}",
        "# Terminal B, trial tool:",
        f"python3 {PL.FIELDLOG}/csvlog_trial.py {a.data_dir}/<NEW_SESSION_FOLDER> --runsheet {rs.resolve()} --codebook {cb_path.resolve()}",
        f"# Terminal C, episode k (k = 1..{neps}): the {nm + 1} probes (L3, then each module alone in sheet order); the",
        "# runner prompts before each; run the choice (Terminal D) BEFORE confirming the split prompt:",
        f"PYTHONPATH={PL.REPO} python3 -m capxfer.field.probe_runner {out.resolve()}/schedule.json --session {sid} "
        f"--from-block $(({nm + 1}*k-{nm})) --to-block $(({nm + 1}*k)) " + f"--ids {idlist} --confirm",
        "# Terminal D, step 1 BEFORE the split, on THIS laptop, from the L3-probe trial file just closed ('e'):",
        f"PYTHONPATH={PL.REPO} python3 -m capxfer.field.choose_module {a.data_dir}/<NEW_SESSION_FOLDER> --episode $k "
        f"--sheets {out.resolve()} --bundle {bundle} --lambda <lambda_v2.json from the freeze record> "
        f"--log {out.resolve()}/choices/choices.csv",
        "# Terminal D, step 2 after the last lone probe ('e'): the oracle from the lone probes' truth + the DOCK LIST:",
        f"PYTHONPATH={PL.REPO} python3 -m capxfer.field.choose_module {a.data_dir}/<NEW_SESSION_FOLDER> --episode $k "
        f"--sheets {out.resolve()} --log {out.resolve()}/choices/choices.csv --finalize",
        "# then, for each module Mk of the printed dock list, in its order (L = Mk's lone-probe trial, printed with the list):",
        f"PYTHONPATH={PL.REPO} python3 -m capxfer.field.probe_runner {out.resolve()}/schedule.json --set-faults $L --ids {idlist}",
        "#   ... dock Mk with the dock executor (WS-F, TEAM-CONFIRM; not in this packet), FieldLog row 's' ... 'e' ...",
        f"PYTHONPATH={PL.REPO} python3 -m capxfer.field.probe_runner {out.resolve()}/schedule.json --clear-faults $L --ids {idlist}",
    ]) + "\n")
    with open(out / "fieldlog_runsheets.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["sched_session", "unit", "assemblies", "n_trials", "first_pos", "last_pos", "runsheet_sha256",
                    "config_sha256", "session_ordinal", "dir", "trial_order"])
        w.writerow([sid, "log", f"{sched['choice']['episodes'][0]['assembly']}+single", len(fl), 1, (nm + 1) * neps, PL.sha256_file(rs), cfg_sha, plan["session_ordinal"],
                    f"{sid}/log", " ".join(order)])
    card = [f"# Session card {sid} (DEMO, card 10)", "",
            f"Schedule sha256 `{h}` · {neps} episodes · {len(fl)} FieldLog rows. Arm identities are never shown. Narrow bay "
            f"{cc['bay']['width_m']:.2f} m wide x {cc['bay']['depth_m']:.2f} m deep, opening in +y (sideways entry only).", "",
            "Operators: A ______ (logger, trial tool, choose_module)  B ______ (e-stop, video, physical changes)", ""]
    for e in sched["choice"]["episodes"]:
        marks = ", ".join(f"{m} mark {e['start_pose_id'][m]}" for m in e["modules"])
        card += [f"## Episode {e['episode']}", "",
                 f"- [ ] Build {e['assembly']}; probe it (row "
                 f"{next(r['fieldlog_row'] for r in rows if r['episode'] == e['episode'] and r['step'] == 'PROBE_ASSEMBLY')})",
                 "- [ ] choose_module (pre-split) logged ✓",
                 f"- [ ] Split; probe alone: {', '.join(e['dock_order'])}",
                 "- [ ] choose_module --finalize; DOCK LIST: #1 ____  #2 ____  #3 ____",
                 f"- [ ] Docks in list order (start marks: {marks}); outcomes ____ / ____ / ____", ""]
    (out / sid).mkdir(exist_ok=True)
    (out / sid / "SESSION_CARD.md").write_text("\n".join(card) + "\n")
    commit = json.loads((out / "commit.json").read_text())
    commit.update(choice_runsheet_sha256=PL.sha256_file(out / "choice_runsheet.csv"),
                  realv2_commitment=sched["realv2"]["commitment"], realv2_schedule_sha256=sched["realv2"]["realv2_schedule_sha256"])
    (out / "commit.json").write_text(json.dumps(commit, indent=1) + "\n")
    lo, hi = demo_hours(cfg)
    ed = round(demo_expected_docks(), 2)
    (out / "SUMMARY.md").write_text(
        f"# Run-sheet summary (DEMO)\n\nSchedule sha256 `{h}`; PROTOCOL_V2 seed commitment `{sched['realv2']['commitment']}`; "
        f"DEMO draws: {sched['choice']['demo_source']}. {neps} episodes on {sched['choice']['episodes'][0]['assembly']}: "
        f"{(nm + 1) * neps} probes (full library) + ~{ed * neps:.0f} docks (expected {ed} per episode, max {nm}), "
        f"{len(fl)} FieldLog rows reserved, ~{lo:.1f}–{hi:.1f} h (realv2_design.hours_table, TEAM-CONFIRM minutes).\n")
    print(f"choice (DEMO): {neps} episodes, ~{ed * neps:.0f} docks; schedule {out}/schedule.json sha256 {h}")
    return 0


def demo_hours(cfg=None):
    """DEMO hours = realv2_design.hours_table() (episodes x (fixed minutes + expected deduplicated docks x DOCK_MIN))."""
    RD = PL.load_realv2_design()
    return tuple(RD.hours_table()["rows"]["demo"])


def demo_expected_docks():
    RD = PL.load_realv2_design()
    return RD.expected_docks()["mean"]


def plan_table(cfg):
    """Timeline rows with trials and hours (dummy seed; counts do not depend on the seed)."""
    params = A.load_params(PARAMS)
    rates = cfg["main"]
    rows = []
    for st in cfg["timeline"]:
        if not st.get("enabled", True):
            continue
        k, rep = st.get("kind"), int(st.get("repeat", 1))
        trials, builds, h = "–", 0, st.get("hours")
        if k == "main":
            s = SCH.generate("ab" * 16, session_order=rates.get("session_order", "block"))
            n = s["n_trials"]
            builds = sum(f["n_reassemblies"] for f in s["field_sessions"])
            trials = f"{n}"
            h = hours(n, builds, rates)
        elif k in ("masking", "simval2", "tagstatic"):
            s = {"masking": build_masking, "simval2": build_simval2, "tagstatic": build_tagstatic}[k](cfg, "ab" * 32, params)
            n = s["n_trials"]
            builds = sum(f["n_reassemblies"] for f in s["field_sessions"])
            trials = f"{n}" + (f" ({len(s['episodes'])} (module, wheel) conditions)" if s.get("episodes") else "")
            h = sched_hours(s, rates, st.get("extra_hours", [0, 0])) if k != "tagstatic" else tuple(st["hours"])
        elif k == "choice":
            RD = PL.load_realv2_design()
            n, nm, ed = RD.DEMO_EPISODES, len(RD.DEMO_MODULES), demo_expected_docks()
            trials = f"{n} episodes ({RD.DEMO_ASSEMBLY}): {(nm + 1) * n} probes + ~{ed * n:.0f} docks ({ed:.2f}/episode)"
            h = demo_hours(cfg)
        elif k in cfg.get("kinds", {}):
            s = build_extra(k, cfg, "ab" * 16, params)
            n = s["n_trials"]
            builds = s["field_sessions"][0]["n_reassemblies"]
            trials = f"{n}" if rep == 1 else f"{rep} × {n}"
            h0, h1 = [rep * x for x in sched_hours(s, rates)]          # per-trial rate (full library or probe subset)
            ex = st.get("extra_hours", [0, 0])
            h = (h0 + ex[0], h1 + ex[1])
        elif k == "docking":
            dc = cfg["docking"]
            nb = int(dc["blocks_all_arms"]) + int(dc["blocks_two_arms"])
            trials = f"{nb} probes + {int(dc['blocks_all_arms']) * len(dc['arms']) + int(dc['blocks_two_arms']) * len(dc['two_arms'])} runs"
        rows.append(dict(st, trials=trials, builds=builds, h=h))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    cfg = load_config()
    for name in ("main", "extra"):
        p = sub.add_parser(name)
        if name == "extra":
            p.add_argument("kind", choices=sorted(cfg["kinds"]) + ["docking", "masking", "choice", "simval2", "tagstatic"])
        p.add_argument("out_dir")
        p.add_argument("--seed-file", required=True)
        p.add_argument("--ids", default="", help="slot=fleet id, e.g. M1=m1,M2=m2,M3=m3,M4=m4 (TEAM-CONFIRM)")
        p.add_argument("--vehicle-codes", default="", help="slot=roster vehicle code (default M1=1..M4=4)")
        p.add_argument("--truth-id", default="capxfer_asm", help="AprilTag truth id of the robot tag (TEAM-CONFIRM)")
        p.add_argument("--data-dir", default="/data/capxfer/sessions")
        p.add_argument("--operator-codes", default="")
        p.add_argument("--site-code", type=int, default=1)
        p.add_argument("--prereg-sha256", default=None)
        p.add_argument("--study-code", default="15")
        p.add_argument("--module", default="M1", choices=PL.MODULES)
        p.add_argument("--day", type=int, default=1, choices=[1, 2])
        p.add_argument("--session-order", choices=["block", "segment"], default=None)
        p.add_argument("--fieldlog", choices=["per_session", "per_segment"], default=None)
        p.add_argument("--naive-arm", action="store_true")
        p.add_argument("--sealed-dir", default=None)
        p.add_argument("--bundle", default=None, help="choice: the frozen choice-bundle folder (written into COMMANDS.txt)")
        p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        rows = plan_table(cfg)
        print("| # | step | trials | builds | hours |")
        print("|---|---|---|---|---|")
        for i, r in enumerate(rows):
            h = r["h"]
            print(f"| {i} | {r['title']} | {r['trials']} | {r['builds'] or ''} | {h[0]:.1f}–{h[1]:.1f} |")
        return 0
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = read_seed(a.seed_file)
    if a.cmd == "main":
        order = a.session_order or cfg["main"].get("session_order", "block")
        sched = SCH.generate(seed, session_order=order)          # schedule.py DEFAULTS (prereg §3 allocation + overrides)
        kind = "main"
    else:
        kind = a.kind
        if kind == "docking":
            return docking(out, seed, a, cfg)
        if kind == "choice":
            return choice(out, seed, a, cfg)
        if kind in ("masking", "simval2", "tagstatic"):
            sched = {"masking": build_masking, "simval2": build_simval2, "tagstatic": build_tagstatic}[kind](
                cfg, seed, A.load_params(PARAMS))
        else:
            sched = build_extra(kind, cfg, seed, A.load_params(PARAMS), module=a.module, day=a.day)
    h = write_schedule(out, sched, seed, a.force)
    tot, builds, lo, hi = write_all(out, sched, a, kind, cfg)
    print(f"{kind}: schedule {out}/schedule.json sha256 {h}; {tot} trials, {builds} builds, ~{lo:.1f}-{hi:.1f} h. "
          f"See {out}/SUMMARY.md and {out}/<SESSION>/SESSION_CARD.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

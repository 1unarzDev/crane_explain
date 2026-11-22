"""Shared helpers for the capxfer field-packet tools (WS-P). Standard library + numpy + PyYAML only (no torch, no ROS).

Every tool in this folder is READ-ONLY with respect to FieldLog session folders: it never creates, edits, renames or
deletes anything inside a session. Reports go to stdout or to a path the operator names outside the session.

Paths: the capxfer repository is found from this file (../../..) or $CAPXFER_REPO; FieldLog from $FIELDLOG_DIR or
~/crane_ml/Tools/FieldLog.
"""
from __future__ import annotations

import csv
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKET = HERE.parent
REPO = Path(os.environ.get("CAPXFER_REPO", PACKET.parents[1]))
FIELDLOG = Path(os.environ.get("FIELDLOG_DIR", Path.home() / "crane_ml" / "Tools" / "FieldLog"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ASM_CODE = {"single": 1, "pair": 2, "L3": 3, "square4": 4}
ASM_OF_CODE = {v: k for k, v in ASM_CODE.items()}
CORNERS = ("fl", "fr", "rl", "rr")
MODULES = ("M1", "M2", "M3", "M4")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(path, folder):
    """True if `path` lies inside `folder` (both resolved)."""
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
        return True
    except ValueError:
        return False


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        rows = list(rd)
    if not rows:
        return [], []
    return rows[0], rows[1:]


# ------------------------------------------------------------------------------------------------ manual CSV schemas
# Every CSV the team fills in by hand. (column, type, unit, required, allowed, description)
# type: str | int | float | bool01 | enum | date | time | sha256. allowed: tuple of enum values, or (lo, hi) for numbers.
# The templates in ../templates/ are generated from this table (make_templates.py), the validator checks against it
# and CSV_SPEC.md documents it. Change it only here.

FLOOR_CODES = ("sealed_concrete", "bare_concrete", "tile", "vinyl", "carpet", "asphalt", "other")
SESSION_KINDS = ("main", "shakedown", "hooktest", "pilot", "sysid", "simval", "apriltag", "natural", "floor2",
                 "tetromino", "docking", "dockpilot", "masking", "choice")
NOTE_CATEGORIES = ("intervention", "anomaly", "contact", "wheel_slip", "latch", "cable", "estop", "video_gap",
                   "reassembly", "battery", "logger", "other")
DOCK_OUTCOMES = ("success", "collision", "pose_error", "timeout", "infeasible", "abort", "invalid")
EXEC_STATUS = ("done", "infeasible", "timeout", "abort", "crash", "not_started")
INVALID_REASONS = ("none", "executor_crash", "planner_crash", "comms_loss", "apriltag_coverage", "fault_dislodged_before",
                   "fault_dislodged_after", "person_in_arena", "other")
NATURAL_KINDS = ("jammed_roller", "debris_wrap", "worn_roller", "loose_hub", "low_battery", "healthy_pair")
PHASES = ("assembly", "split")
DOCK_OFFSETS = (-0.10, -0.05, 0.0, 0.05, 0.10)          # start marks 1..5 (= realv2_design.DOCK_OFFSETS_M)

SCHEMAS = {
    "session_index": [
        ("fieldlog_dir", "str", "", True, None, "FieldLog session folder name, e.g. gnd_20261201T140312Z_a1b2c3"),
        ("kind", "enum", "", True, SESSION_KINDS, "which protocol card this session belongs to"),
        ("sched_session", "str", "", True, None, "scheduled session id from the run-sheet folder (S01..S12, PILOT, SYSID_M2, ...)"),
        ("segment", "int", "", True, (1, 99), "build number k inside the scheduled session ('Build k' on the session card); one row per build"),
        ("assembly", "enum", "", True, ("single", "pair", "L3", "square4", "tetromino", "none"), "assembly type built"),
        ("module_slots", "str", "", True, None, "slot=fleet id pairs used in the rendered config, ';'-separated: M1=m1;M3=m3"),
        ("module_serials", "str", "", True, None, "slot=physical serial/label pairs, ';'-separated (which physical module sat in each slot)"),
        ("floor_id", "enum", "", True, ("A", "B"), "A = system-ID floor (main); B = second floor (card 07)"),
        ("floor_code", "enum", "", True, FLOOR_CODES, "floor material"),
        ("date_local", "date", "", True, None, "YYYY-MM-DD"),
        ("operator_a_code", "int", "", True, (1, 999), "roster code of operator A (lead/logger)"),
        ("operator_b_code", "int", "", True, (1, 999), "roster code of operator B (e-stop/video)"),
        ("tag_dx_m", "float", "m", True, (-1.0, 1.0), "robot AprilTag centre minus assembly origin, assembly x (origin = mean of module centres)"),
        ("tag_dy_m", "float", "m", True, (-1.0, 1.0), "same, assembly y"),
        ("tag_dyaw_deg", "float", "deg", True, (-180.0, 180.0), "tag x-axis yaw relative to assembly +x"),
        ("config_sha256", "sha256", "", True, None, "sha256 of the rendered FieldLog config used (fieldlog_config.yaml in the run-sheet folder log/ or segNN_*/)"),
        ("runsheet_sha256", "sha256", "", True, None, "sha256 of the run sheet used (must equal session.yaml runsheet_sha256)"),
        ("video_files", "str", "", True, None, "video file names covering this session, ';'-separated (videos stay with the team)"),
        ("pack_codes", "str", "", False, None, "battery pack codes installed at start, slot=code ';'-separated"),
        ("air_temp_c", "float", "degC", False, (-10.0, 50.0), "room temperature"),
        ("notes", "str", "", False, None, "free text"),
    ],
    "block_log": [
        ("fieldlog_dir", "str", "", True, None, "FieldLog session folder of this block"),
        ("sched_session", "str", "", True, None, "S01..S12 or extra id"),
        ("pos", "int", "", True, (1, 999), "block position in the session (blocks_Sxx.csv pos)"),
        ("block", "int", "", True, (1, 9999), "schedule block number (blocks_Sxx.csv block)"),
        ("assembly", "enum", "", True, ("single", "pair", "L3", "square4", "tetromino"), "assembly type"),
        ("fault_id", "str", "", False, None, "operator block fault id (F4/F6/F7) or empty for automated blocks"),
        ("t_start_local", "time", "", True, None, "hh:mm wall clock at the first trial of the block"),
        ("t_end_local", "time", "", True, None, "hh:mm wall clock after the last trial"),
        ("reassembled_before", "bool01", "", True, None, "1 if the robot was rebuilt before this block"),
        ("reassembly_min", "float", "min", False, (0.0, 120.0), "minutes spent on the rebuild (0 if none)"),
        ("latch_play_mm", "float", "mm", False, (0.0, 20.0), "play between docked modules after the rebuild (multi-module only)"),
        ("zip_tie_wheel", "str", "", False, None, "F4: wheel whose rollers are zip-tied, e.g. M1:fl (must equal the block instruction)"),
        ("zip_ties_intact_end", "bool01", "", False, None, "F4: 1 if all zip-ties were still in place after the last trial"),
        ("sheet_id", "str", "", False, None, "F7: label of the low-friction sheet used"),
        ("sheet_mu_target", "float", "1", False, (0.0, 1.0), "F7: mu in the block instruction"),
        ("sheet_mu_measured", "float", "1", False, (0.0, 1.0), "F7: measured wheel-sheet mu of the sheet used (MEASUREMENTS_SHEET F-items)"),
        ("sheet_edge_ok", "bool01", "", False, None, "F7: 1 if the sheet edge ran under the assembly centre line, left wheels covered"),
        ("payload_module", "str", "", False, None, "F6: module carrying the payload (M1/M3)"),
        ("payload_mass_kg", "float", "kg", False, (0.0, 6.0), "F6: weighed payload mass incl. tray and straps (never > 6)"),
        ("payload_corner", "enum", "", False, ("fl", "fr", "rl", "rr", ""), "F6: corner the payload CG was shifted toward"),
        ("payload_offset_m", "float", "m", False, (0.0, 0.12), "F6: measured CG offset from the module centre (never > 0.12)"),
        ("v_batt_start_min_v", "float", "V", True, (9.0, 13.0), "lowest module pack voltage at block start (multimeter or BAT stream)"),
        ("v_batt_end_min_v", "float", "V", True, (9.0, 13.0), "lowest module pack voltage at block end"),
        ("packs_swapped_before", "bool01", "", True, None, "1 if any battery was swapped before this block"),
        ("n_trials_run", "int", "", True, (0, 999), "trials completed (outcome 1)"),
        ("n_aborted", "int", "", True, (0, 999), "trials aborted / voided"),
        ("notes", "str", "", False, None, "free text"),
    ],
    "operator_notes": [
        ("fieldlog_dir", "str", "", True, None, "FieldLog session folder"),
        ("trial_id", "str", "", False, None, "gnd_capxprobe_NNN if the note is about one trial, else empty"),
        ("t_local", "time", "", True, None, "hh:mm:ss wall clock"),
        ("operator_code", "int", "", True, (1, 999), "roster code of the writer"),
        ("category", "enum", "", True, NOTE_CATEGORIES, "what kind of note"),
        ("note", "str", "", True, None, "free text (FieldLog files accept codes only, so text goes here)"),
    ],
    "video_index": [
        ("video_file", "str", "", True, None, "file name on the team's video storage (the video itself is NOT returned)"),
        ("sha256", "sha256", "", True, None, "sha256sum of the video file"),
        ("size_bytes", "int", "B", True, (1, 10 ** 12), "file size"),
        ("device_code", "int", "", True, (1, 99), "phone/camera code"),
        ("start_local", "str", "", True, None, "YYYY-MM-DD hh:mm:ss start time from the file metadata"),
        ("duration_s", "float", "s", True, (1.0, 86400.0), "duration"),
        ("fieldlog_dirs", "str", "", True, None, "FieldLog session folders it covers, ';'-separated"),
        ("notes", "str", "", False, None, "gaps, battery changes of the phone, etc."),
    ],
    "docking_runs": [
        ("fieldlog_dir", "str", "", True, None, "FieldLog session folder of the docking session"),
        ("block", "int", "", True, (1, 99), "dock schedule block"),
        ("run_in_block", "int", "", True, (1, 8), "run index inside the block"),
        ("run_global", "int", "", True, (1, 999), "dock schedule run number"),
        ("arm_code", "enum", "", True, ("A", "B", "C", "D", "E"), "arm code shown on the dock run sheet (arm names stay sealed)"),
        ("start_pose_id", "int", "", True, (1, 5), "start mark used (1..5 = x offset -0.10..+0.10 m)"),
        ("t_start_local", "time", "", True, None, "hh:mm:ss when the executor was started"),
        ("executor_status", "enum", "", True, EXEC_STATUS, "what the executor printed at the end"),
        ("operator_outcome", "enum", "", True, DOCK_OUTCOMES, "operator's call (the automatic tag-based call is computed later and wins)"),
        ("contact_seen", "bool01", "", True, None, "1 if the robot touched a wall"),
        ("fault_in_place_before", "bool01", "", True, None, "zip-ties checked in place before the run"),
        ("fault_in_place_after", "bool01", "", True, None, "zip-ties still in place after the run"),
        ("invalid_reason", "enum", "", True, INVALID_REASONS, "technical invalidity (DOCKING_PROTOCOL §7); none if valid"),
        ("repeat_of", "int", "", False, (1, 999), "run_global this run repeats (technical repeat), else empty"),
        ("v_batt_min_v", "float", "V", False, (9.0, 13.0), "pack voltage after the run"),
        ("notes", "str", "", False, None, "free text"),
    ],
    "episode_log": [     # WS-P2 masking/split card 09: one row per phase of every episode (assembly, then each module alone)
        ("fieldlog_dir", "str", "", True, None, "FieldLog session folder"),
        ("sched_session", "str", "", True, None, "MASK1 / MASK2 / DEMO"),
        ("episode", "str", "", True, None, "MASK: the build (square4, single_M2, L3, pair); DEMO: the episode (D07)"),
        ("phase", "enum", "", True, PHASES, "assembly = the whole assembly; split = one module alone (single_Mk builds, DEMO lone probes)"),
        ("module", "enum", "", True, MODULES + ("all",), "split: the module driving alone; assembly: all"),
        ("module_serial", "str", "", True, None, "physical label of that module (assembly: ';'-separated slot=serial list)"),
        ("block_pos", "int", "", True, (1, 999), "block position on the session card (runner prompt 'block pos')"),
        ("first_trial", "int", "", True, (1, 10 ** 6), "first schedule trial number of this build / phase"),
        ("last_trial", "int", "", True, (1, 10 ** 6), "last schedule trial number of this build / phase"),
        ("fault_carrier_seen", "str", "", True, None, "module carrying a PHYSICAL fault (zip ties) as checked by both people: DEMO the faulted module, MASK none (firmware faults only)"),
        ("zip_tie_wheels_seen", "str", "", False, None, "zip-tied wheels on the probed robot, checked before the first trial, ';'-separated (M3:fl;M3:rr); empty if none"),
        ("zip_ties_intact_end", "bool01", "", False, None, "1 if every zip tie of the episode was still in place after its last trial (empty if none)"),
        ("tag_module", "enum", "", True, MODULES + ("assembly",), "where the truth AprilTag sat: on a module (split) or the assembly tag position"),
        ("others_parked", "bool01", "", True, None, "split: 1 if the other modules were >= 1 m away and not commanded (assembly: 1)"),
        ("notes", "str", "", False, None, "free text"),
    ],
    "choice_runs": [     # WS-P2 DEMO (card 10, SPINE D15): one row per dock (1-3 per episode: the printed dock list)
        ("fieldlog_dir", "str", "", True, None, "FieldLog session folder of the DEMO session"),
        ("episode", "str", "", True, None, "DEMO episode id from the sheet (D01..D20)"),
        ("dock_in_episode", "int", "", True, (1, 3), "position in the DOCK LIST choose_module printed (1..3)"),
        ("module", "enum", "", True, MODULES, "module that docked (= that position of the printed dock list)"),
        ("run_global", "int", "", True, (1, 999), "running dock number in the session (repeats get a new number)"),
        ("start_pose_id", "int", "", True, (1, 5), "start mark used (1..5 = x offset -0.10..+0.10 m; printed with the dock list)"),
        ("t_start_local", "time", "", True, None, "hh:mm:ss when the executor was started"),
        ("executor_status", "enum", "", True, EXEC_STATUS, "what the executor printed at the end"),
        ("operator_outcome", "enum", "", True, DOCK_OUTCOMES, "operator's call (the automatic tag-based call is computed later and wins)"),
        ("contact_seen", "bool01", "", True, None, "1 if the robot touched a bay wall"),
        ("fault_in_place_before", "bool01", "", True, None, "1 if the module's firmware fault state was as scheduled before the dock (fault_state echo: coast on its sealed wheel, or ok)"),
        ("fault_in_place_after", "bool01", "", True, None, "same after the dock"),
        ("choice_logged", "bool01", "", True, None, "1 if choose_module logged this episode's choice before the split"),
        ("invalid_reason", "enum", "", True, INVALID_REASONS, "technical invalidity (DOCKING_PROTOCOL §7); none if valid"),
        ("repeat_of", "int", "", False, (1, 999), "run_global this dock repeats (technical repeat), else empty"),
        ("v_batt_min_v", "float", "V", False, (9.0, 13.0), "pack voltage after the dock"),
        ("notes", "str", "", False, None, "free text"),
    ],
    "natural_faults": [
        ("fieldlog_dir", "str", "", True, None, "FieldLog session folder"),
        ("block", "int", "", True, (1, 999), "natural-fault block number from the run sheet folder"),
        ("first_trial_id", "str", "", True, None, "gnd_capxprobe_NNN of the first trial with this condition"),
        ("last_trial_id", "str", "", True, None, "gnd_capxprobe_NNN of the last trial"),
        ("assembly", "enum", "", True, ("single", "pair", "square4"), "assembly type"),
        ("module", "enum", "", True, MODULES, "module carrying the condition"),
        ("wheel", "enum", "", True, CORNERS + ("all",), "wheel (all = whole module, e.g. low battery)"),
        ("kind", "enum", "", True, NATURAL_KINDS, "condition (healthy_pair = the paired healthy block)"),
        ("rollers_affected", "int", "", False, (0, 9), "number of rollers affected (jammed/debris/worn)"),
        ("how_induced", "str", "", True, None, "exactly what was done (material, length, turns, screw turns)"),
        ("free_spin_check", "str", "", False, None, "hand check result: roller turns freely / stiff / locked; hub slips at ...)"),
        ("v_batt_rest_v", "float", "V", False, (9.0, 13.0), "low_battery: resting pack voltage before the block"),
        ("photo_file", "str", "", False, None, "photo file name (kept with the videos)"),
        ("safety_ok", "bool01", "", True, None, "1 = both operators checked the card's safety limits before the block"),
        ("notes", "str", "", False, None, "free text"),
    ],
    "apriltag_reference": [
        ("fieldlog_dir", "str", "", True, None, "FieldLog session folder of the static check"),
        ("trial_id", "str", "", True, None, "gnd_capxprobe_NNN during which the robot sat on this pose"),
        ("pose_id", "int", "", True, (1, 20), "reference pose number"),
        ("x_m", "float", "m", True, (-10.0, 10.0), "reference x of the assembly origin in the arena frame (tape/laser)"),
        ("y_m", "float", "m", True, (-10.0, 10.0), "reference y"),
        ("yaw_deg", "float", "deg", True, (-180.0, 180.0), "reference heading"),
        ("method", "enum", "", True, ("tape", "laser", "jig"), "how the reference was measured"),
        ("pos_uncertainty_mm", "float", "mm", True, (0.0, 50.0), "honest uncertainty of the reference position"),
        ("notes", "str", "", False, None, "free text"),
    ],
    "hook_test_manual": [
        ("module", "enum", "", True, MODULES, "module slot under test"),
        ("serial", "str", "", True, None, "physical module label"),
        ("wheel", "enum", "", True, CORNERS, "wheel"),
        ("mode", "enum", "", True, ("ok", "weak", "coast", "brake"), "fault mode set with fault_cmd"),
        ("g", "float", "1", False, (0.0, 1.0), "weak only"),
        ("slp_pin_v", "float", "V", True, (0.0, 5.5), "G2 SLP pin of that channel to GND (coast must be ~0, others high)"),
        ("pwm_pin_v", "float", "V", False, (0.0, 5.5), "G2 PWM pin, motor commanded but robot lifted (brake: 0)"),
        ("out_a_v", "float", "V", False, (-1.0, 15.0), "MxA to GND with the command held at zero speed"),
        ("out_b_v", "float", "V", False, (-1.0, 15.0), "MxB to GND"),
        ("backdrive_force_n", "float", "N", False, (0.0, 200.0), "spring-scale force at the tyre to turn the wheel slowly (~1 rev in 5 s)"),
        ("feel", "enum", "", True, ("free", "stiff", "very_stiff", "locked", "driven"), "how the wheel feels when turned by hand"),
        ("echo_seen", "bool01", "", True, None, "1 if the fault_state echo appeared (ros2 topic echo)"),
        ("pass_by_eye", "bool01", "", True, None, "1 if the behaviour matched FIRMWARE_HOOKS.md table"),
        ("operator_code", "int", "", True, (1, 999), "roster code"),
        ("notes", "str", "", False, None, "free text"),
    ],
}

MEASUREMENT_COLUMNS = [
    ("item_id", "str", "", True, None, "row id from MEASUREMENTS_SHEET.md (do not change)"),
    ("module", "str", "", True, None, "M1..M4, ALL or ARENA"),
    ("quantity", "str", "", True, None, "what is measured (do not change)"),
    ("value", "float", "", False, None, "the measured number (empty only if status is NOT_MEASURED or NA)"),
    ("unit", "str", "", True, None, "unit of value (do not change)"),
    ("method", "str", "", True, None, "how to measure (do not change)"),
    ("instrument", "str", "", False, None, "scale/tape/multimeter model"),
    ("n_repeats", "int", "", False, (0, 100), "repeats averaged into value"),
    ("spread", "float", "", False, None, "max - min over repeats, same unit"),
    ("status", "enum", "", True, ("TODO", "MEASURED", "NOT_MEASURED", "NA"), "TODO until done"),
    ("operator_code", "int", "", False, (1, 999), "roster code"),
    ("date_local", "date", "", False, None, "YYYY-MM-DD"),
    ("params_key", "str", "", True, None, "params/module_measured.yaml key this feeds (do not change)"),
    ("team_confirm", "str", "", True, None, "1 = depends on a hardware fact the team must confirm"),
    ("note", "str", "", False, None, "free text"),
]


def header(name):
    cols = MEASUREMENT_COLUMNS if name == "measurements" else SCHEMAS[name]
    return [c[0] for c in cols]


def check_cell(spec, v):
    """None if OK, else a reason string."""
    col, typ, unit, req, allowed, _ = spec
    v = (v or "").strip()
    if v == "":
        return "required value missing" if req else None
    try:
        if typ == "int":
            x = int(v)
            if allowed and not (allowed[0] <= x <= allowed[1]):
                return f"{x} outside [{allowed[0]}, {allowed[1]}]"
        elif typ == "float":
            x = float(v)
            if x != x or x in (float("inf"), float("-inf")):
                return "not a finite number"
            if allowed and not (allowed[0] <= x <= allowed[1]):
                return f"{x} outside [{allowed[0]}, {allowed[1]}]"
        elif typ == "bool01":
            if v not in ("0", "1"):
                return "must be 0 or 1"
        elif typ == "enum":
            if v not in allowed:
                return f"{v!r} not one of {list(allowed)}"
        elif typ == "date":
            import datetime
            datetime.date.fromisoformat(v)
        elif typ == "time":
            p = v.split(":")
            if not (2 <= len(p) <= 3 and all(q.isdigit() for q in p) and int(p[0]) < 24 and int(p[1]) < 60):
                return "must be hh:mm or hh:mm:ss"
        elif typ == "sha256":
            if len(v) != 64 or any(c not in "0123456789abcdef" for c in v.lower()):
                return "not a 64-hex sha256"
    except ValueError as e:
        return f"cannot parse {v!r} as {typ} ({e})"
    return None


# ------------------------------------------------------------------------------------------------ run-sheet segments
RUNSHEET_HEADER = ["row", "trial_id", "portion_code", "nn", "split_code", "arm_code", "block_code", "amp", "sign", "dur_s",
                   "gap_s", "extra_wait_s", "axis_code", "thr_idx", "n_steps"]
FAULT_CODE = {f"F{k}": k for k in range(8)}


def runsheet_row(k, fault_id, pos):
    return {"row": k, "trial_id": f"gnd_capxprobe_{k:03d}", "portion_code": 67, "nn": k, "split_code": 1,
            "arm_code": FAULT_CODE.get(str(fault_id).rstrip("w*")[:2], 0), "block_code": int(pos), "amp": 0, "sign": 0,
            "dur_s": 0, "gap_s": 0, "extra_wait_s": 0, "axis_code": 0, "thr_idx": 0, "n_steps": 0}


def field_sessions(sched):
    """Run order of a schedule: [{id, blocks: [{pos, block, assembly, kind, fault_id, n_trials, reassemble}]}]."""
    try:
        from capxfer.field import schedule as SCH
        if hasattr(SCH, "field_sessions"):
            return SCH.field_sessions(sched)
    except Exception:
        pass
    if sched.get("field_sessions"):
        return sched["field_sessions"]
    return [{"id": s["assembly"], "blocks": [{"pos": i + 1, "block": b["block"], "assembly": s["assembly"], "kind": b["kind"],
                                               "fault_id": b.get("fault_id"), "n_trials": len(b["trials"]), "reassemble": i == 0}
                                              for i, b in enumerate(s["blocks"])]} for s in sched["sessions"]]


def blocks_by_key(sched):
    return {(s["assembly"], b["block"]): b for s in sched["sessions"] for b in s["blocks"]}


def session_of(sched, assembly, block=None):
    """Schedule session entry of an assembly type; with `block`, the entry holding that block (WS-P2: one type may
    appear with several layouts, e.g. M1 alone and M2 alone after a split)."""
    if block is not None:
        return next(s for s in sched["sessions"] if s["assembly"] == assembly and any(b["block"] == block for b in s["blocks"]))
    return next(s for s in sched["sessions"] if s["assembly"] == assembly)


def segments(sched, fs):
    """Split one field session into FieldLog segments: maximal runs of consecutive blocks of one assembly type.
    A new FieldLog session is started for every segment (the IMU/lidar module and the tag offset change with the
    assembly type; FieldLog logs one IMU and one rf2o stream per session)."""
    byb = blocks_by_key(sched)
    segs = []
    for r in fs["blocks"]:
        if not segs or segs[-1]["assembly"] != r["assembly"] or (r.get("split") is not None and r["reassemble"]):
            segs.append({"assembly": r["assembly"], "blocks": [], "first_block": r["block"]})
        b = byb[(r["assembly"], r["block"])]
        segs[-1]["blocks"].append(dict(r, trials=b["trials"], condition=b.get("condition", {})))
    for i, s in enumerate(segs, start=1):
        s["seg"] = i
        k = 0
        s["rows"] = []
        for b in s["blocks"]:
            b["first_row"] = k + 1
            for t in b["trials"]:
                k += 1
                s["rows"].append((runsheet_row(k, t["fault_id"], b["pos"]), t))
            b["last_row"] = k
    return segs


def expected_trial_order(sched):
    """{runsheet_sha256-independent} map: list of (field_session, seg, assembly, [schedule trial numbers in order])."""
    out = []
    for fs in field_sessions(sched):
        for s in segments(sched, fs):
            out.append((fs["id"], s["seg"], s["assembly"], [int(t["trial"]) for _, t in s["rows"]]))
    return out


def write_runsheet(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNSHEET_HEADER, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


# ------------------------------------------------------------------------------------------------ PROTOCOL_V2 design
def load_realv2_design():
    """capxfer/analysis/realv2_design.py (PROTOCOL_V2 sealed schedule expansion). On the robot laptop's system python
    (numpy + PyYAML only) the package import fails (pandas / scipy are analysis-side); the file is then loaded on its
    own with inert stand-ins for `scipy.stats` and the sibling `stats` module, which only its power functions use.
    expand_schedule / commitment / the design constants are pure numpy + hashlib, so the result is identical."""
    try:
        from capxfer.analysis import realv2_design as RD
        return RD
    except ImportError:
        pass
    import importlib.util
    import types
    path = REPO / "capxfer" / "analysis" / "realv2_design.py"
    saved = {k: sys.modules.get(k) for k in ("scipy", "scipy.stats", "_cx_realv2", "_cx_realv2.stats")}
    try:
        if saved["scipy"] is None:
            sc = types.ModuleType("scipy")
            sc.stats = types.ModuleType("scipy.stats")
            sys.modules["scipy"], sys.modules["scipy.stats"] = sc, sc.stats
        pkg = types.ModuleType("_cx_realv2")
        pkg.__path__ = []
        sys.modules["_cx_realv2"] = pkg
        sys.modules["_cx_realv2.stats"] = types.ModuleType("_cx_realv2.stats")
        spec = importlib.util.spec_from_file_location("_cx_realv2.realv2_design", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

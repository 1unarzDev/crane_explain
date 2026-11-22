#!/usr/bin/env python3
"""Validate returned capxfer CSV data. READ-ONLY: never writes inside a session folder.

  validate_csv_session.py PATH [PATH ...] [--sheets RUN_SHEET_DIR ...] [--schedule schedule.json] [--manual DIR]
                          [--expect field|simulation] [--out report.txt] [--csv report.csv] [--no-loader] [--no-verify]

PATH = one FieldLog v2 session folder (has session.yaml), or a return-package root (sessions/*/ and manual/ inside).
--sheets: the folder(s) make_run_sheets.py wrote; sessions are matched to their run sheet (fieldlog_runsheets.csv) by
          the run-sheet sha256 and checked against that schedule and trial order. Without it, give --schedule.
--expect field (default): a SIMULATION label is a FAIL. --expect simulation: for dry runs.

Checks (C1-C15 per session, M1-M3 for manual CSVs; CSV_SPEC.md §6 lists them):
  C1 files  C2 SHA256SUMS  C3 session.yaml  C4 trials.csv  C5 trial-file schema/dtypes  C6 trial markers
  C7 assembly frames (+ WS-P2: the logged modules = the scheduled layout)  C8 fault echoes vs schedule  C9 run order
  vs run sheet (choice demo: dock rows have no markers)  C10 rates and gaps (prereg §7 d, truth coverage)  C11 liveness
  (encoders, duty, IMU, tag)  C12 housekeeping/battery  C13 events.csv  C14 FieldLog verify_session  C15 capxfer loader
  (skipped if torch is not installed)  E1 (WS-P2) masking episode links complete over all sessions given
  M4 (WS-P2) episode_log.csv / choice_runs.csv agree with the sealed masking / choice schedules
Verdict: FAIL if any check fails, WARN if any warns, else PASS. Exit code 0 PASS, 1 WARN, 2 FAIL.
Needs only Python 3.10+, numpy and PyYAML (system python on the robot laptop is fine).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import packet_lib as PL  # noqa: E402

from capxfer import assembly as A  # noqa: E402
from capxfer.probes import Schedule  # noqa: E402
from capxfer.field import fieldlog_config as FC  # noqa: E402
from capxfer.field import schedule as SCHF  # noqa: E402

PARAMS = PL.REPO / "params" / "module_measured.yaml"
REQUIRED_FILES = ("session.yaml", "trials.csv", "events.csv", "housekeeping.csv", "SHA256SUMS")
ALLOWED_FILES = set(REQUIRED_FILES) | {"README_SESSION.txt"}
LEAD = ["t_mono_ns", "t_src_ns", "seq", "src"]
# expected and minimum rates (Hz) inside a probe trial; CUR only if present; LODOM only if the assembly has a lidar
RATES = {"CTL": (50, 40), "WHL": (100, 50), "IMU": (100, 50), "TRUTH": (30, 20), "LODOM": (10, 8), "CUR": (100, 50)}
GAP_ROBOT_S = 0.100            # prereg §7(d)
TRUTH_MISSING_MAX = 0.20       # prereg §7 capability-target rule
MODE_CODE = {"ok": 0, "weak": 1, "coast": 2, "brake": 3, "roller_locked": 4, "payload": 5, "low_mu": 6}
OPERATOR_MODES = {"roller_locked", "payload", "low_mu"}
SRC_NAME = {v: k for k, v in FC.SRC.items()}
LEVELS = ("PASS", "WARN", "FAIL")


class Report:
    def __init__(self):
        self.items = []

    def add(self, level, check, scope, msg):
        self.items.append((level, check, scope, msg))

    def ok(self, check, scope, msg="ok"):
        self.add("PASS", check, scope, msg)

    def warn(self, check, scope, msg):
        self.add("WARN", check, scope, msg)

    def fail(self, check, scope, msg):
        self.add("FAIL", check, scope, msg)

    def verdict(self, scope_prefix=None):
        lv = [i[0] for i in self.items if scope_prefix is None or i[2].startswith(scope_prefix)]
        return "FAIL" if "FAIL" in lv else ("WARN" if "WARN" in lv else "PASS")


# ------------------------------------------------------------------------------------------------ reading
def read_trial(path, rep, scope):
    """{src: (t_mono int64 (k,), {col: float (k,)})}, header, n_rows. Checks dtypes on the fly (C5)."""
    table = FC.source_table()
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        hdr = next(rd, [])
        rows = list(rd)
    ci = {c: i for i, c in enumerate(hdr)}
    need = LEAD + FC.portion_columns()
    miss = [c for c in need if c not in ci]
    if hdr[:4] != LEAD:
        rep.fail("C5", scope, f"first columns are {hdr[:4]}, expected {LEAD}")
    if miss:
        rep.fail("C5", scope, f"{len(miss)} required column(s) missing, e.g. {miss[:6]}")
    extra = [c for c in hdr if c not in need]
    if extra:
        rep.warn("C5", scope, f"{len(extra)} unexpected column(s): {extra[:6]}")
    per = defaultdict(list)
    bad_text, bad_src, stray, nonmono = 0, 0, 0, 0
    last_t = None
    examples = []
    for r in rows:
        if len(r) != len(hdr):
            bad_text += 1
            examples.append(f"row with {len(r)} cells")
            continue
        try:
            t = int(r[0])
            s = int(r[ci["src"]])
        except (ValueError, KeyError):
            bad_text += 1
            examples.append(f"bad t_mono_ns/src {r[:4]}")
            continue
        if last_t is not None and t < last_t - 1_000_000:
            nonmono += 1
        last_t = t if last_t is None else max(last_t, t)
        if s not in table:
            bad_src += 1
            continue
        per[s].append((t, r))
    if bad_src:
        rep.warn("C5", scope, f"{bad_src} row(s) with a src code that is not a capxfer source")
    if nonmono:
        rep.warn("C5", scope, f"t_mono_ns goes backwards by > 1 ms {nonmono} time(s)")
    out = {}
    for s, rr in per.items():
        cols = [c for c in table[s][1] if c in ci]
        own = set(cols) | set(LEAD)
        tm = np.array([t for t, _ in rr], dtype=np.int64)
        d = {}
        for c in cols:
            j = ci[c]
            v = np.full(len(rr), np.nan)
            for k, (_, r) in enumerate(rr):
                x = r[j]
                if x == "":
                    continue
                try:
                    f = float(x)
                except ValueError:
                    bad_text += 1
                    if len(examples) < 3:
                        examples.append(f"{c}={x!r}")
                    continue
                if not math.isfinite(f) or x.strip().lower() in ("nan", "inf", "-inf"):
                    bad_text += 1
                    if len(examples) < 3:
                        examples.append(f"{c}={x!r}")
                    continue
                v[k] = f
            d[c] = v
        # cells filled outside the row's own src columns
        others = [ci[c] for c in hdr if c not in own and c in ci]
        for _, r in rr[:2000]:
            if any(r[j] != "" for j in others):
                stray += 1
        out[s] = (tm, d)
    if bad_text:
        rep.fail("C5", scope, f"{bad_text} cell(s) not parseable as finite numbers (e.g. {examples[:3]})")
    if stray:
        rep.warn("C5", scope, f"{stray} row(s) fill columns of another source")
    return out, hdr, len(rows)


def window(streams, t0, t1):
    o = {}
    for s, (tm, d) in streams.items():
        m = (tm >= t0) & (tm <= t1)
        o[s] = (tm[m], {c: v[m] for c, v in d.items()})
    return o


def markers(streams):
    s = FC.SRC["TRIAL"]
    if s not in streams:
        return []
    tm, d = streams[s]
    ev = []
    for k in np.argsort(tm, kind="stable"):
        ev.append((int(tm[k]), int(d["trial_num"][k]) if np.isfinite(d["trial_num"][k]) else -1,
                   int(d["trial_event_code"][k]) if np.isfinite(d["trial_event_code"][k]) else -1,
                   int(d["asm_type_code"][k]) if np.isfinite(d["asm_type_code"][k]) else -1))
    return ev


def build_assembly(streams, asm_code, params):
    side = float(params["module"]["side_x_m"])
    frames = {}
    for m in FC.MODULES:
        s = FC.SRC[f"ASM_{m}"]
        if s in streams and len(streams[s][0]):
            d = streams[s][1]
            frames[m] = (d[f"asm_x_m_{FC.mk(m)}"][-1], d[f"asm_y_m_{FC.mk(m)}"][-1], d[f"asm_yaw_deg_{FC.mk(m)}"][-1])
    if not frames:
        raise ValueError("no assembly frame (/fleet/<id>/assembly) logged at trial start")
    mods = sorted(frames)
    xs = np.array([frames[m][0] for m in mods])
    ys = np.array([frames[m][1] for m in mods])
    spec = [(m, int(round((frames[m][0] - xs.min()) / side)), int(round((frames[m][1] - ys.min()) / side)),
             int(round(frames[m][2] / 90.0)) % 4) for m in mods]
    name = {1: "single", 2: "pair", 3: "L3", 4: "square4"}.get(asm_code, "custom")
    asm = A.build(name, spec, params)
    if not np.allclose(asm.module_pos, np.stack([xs, ys], 1), atol=0.02):
        raise ValueError("logged module frames are not on the module grid (side length / frames)")
    return asm, spec, frames


def echoes(streams, t_lo, t_hi):
    """[(t, module, wheel_code, mode_code, g, trial)] fault_state echoes in [t_lo, t_hi]."""
    out = []
    for m in FC.MODULES:
        s = FC.SRC[f"FLT_{m}"]
        if s not in streams:
            continue
        tm, d = streams[s]
        for k in range(len(tm)):
            if t_lo <= tm[k] <= t_hi:
                out.append((int(tm[k]), m, d[f"fs_wheel_code_{FC.mk(m)}"][k], d[f"fs_mode_code_{FC.mk(m)}"][k],
                            d[f"fs_g_{FC.mk(m)}"][k], d[f"fs_trial_{FC.mk(m)}"][k]))
    return sorted(out)


def rate_and_gaps(tm, t0, t1):
    if len(tm) < 2:
        return None, None
    t = np.sort(tm)
    dt = np.diff(t) / 1e9
    dur = (t1 - t0) / 1e9
    return len(t) / max(dur, 1e-9), float(dt.max()) if len(dt) else None


def max_gap_in(tm, a, b):
    """Largest interval (s) without a sample inside [a, b] (ns), counting the edges."""
    t = np.sort(tm[(tm >= a) & (tm <= b)])
    pts = np.concatenate([[a], t, [b]])
    return float(np.diff(pts).max() / 1e9)


# ------------------------------------------------------------------------------------------------ session checks
def load_sheets(dirs):
    seg_by_hash, scheds = {}, {}
    for d in dirs or []:
        d = Path(d)
        p = d / "fieldlog_runsheets.csv"
        if not p.is_file():
            p = d / "segments.csv"
        if not p.is_file():
            continue
        sch = json.loads((d / "schedule.json").read_text()) if (d / "schedule.json").is_file() else None
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                r["_sheets"] = str(d)
                seg_by_hash[r["runsheet_sha256"]] = r
        if sch is not None:
            scheds[str(d)] = sch
    return seg_by_hash, scheds


def sched_modules(sched):
    """{trial number: sorted module ids of its scheduled layout}."""
    return {int(t["trial"]): sorted(m[0] for m in s["modules"]) for s in sched["sessions"] for b in s["blocks"]
            for t in b["trials"]}


def check_episodes(summ, rep):
    """E1 (WS-P2, PROTOCOL_V2 MASK): every (module, wheel) condition with any completed trial has completed trials in EVERY
    part it is scheduled in (each parent assembly and the module alone); the links live in the sealed schedule."""
    by_sched = {}
    for s in summ:
        if s and s.get("sched") is not None and s["sched"].get("episodes"):
            by_sched.setdefault(id(s["sched"]), (s["sched"], set()))[1].update(n for n, ok in s["done"] if ok == "ok")
    for sched, done in by_sched.values():
        bad, full, seen = [], 0, 0
        for e in sched["episodes"]:
            parts = e["parts"]
            if not {x for v in parts.values() for x in v} & done:
                continue
            seen += 1
            miss = [k for k, v in parts.items() if not set(v) & done]
            if miss:
                bad.append(f"{e['episode']}: no completed trial in {', '.join(miss)}")
            else:
                full += 1
        if bad:
            rep.warn("E1", "episodes", f"{len(bad)} condition(s) with an incomplete assembly->alone link (validate all MASK "
                                       f"sessions together): {bad[:4]}")
        rep.ok("E1", "episodes", f"{full} of {seen} started condition(s) linked across every part "
                                 f"({len(sched['episodes'])} in the schedule)")


def sched_index(sched):
    return {(s["assembly"], int(t["trial"])): t for s in sched["sessions"] for b in s["blocks"] for t in b["trials"]}


def check_session(d, rep, a, params, seg_by_hash, scheds):
    sc = d.name
    # C1 files
    names = sorted(p.name for p in d.iterdir())
    subdirs = [p.name for p in d.iterdir() if p.is_dir()]
    for f in REQUIRED_FILES:
        if f not in names:
            rep.fail("C1", sc, f"{f} missing")
    if subdirs:
        rep.fail("C1", sc, f"sub-folder(s) inside the session folder: {subdirs} (move derived files to a sibling derived/)")
    trial_files = sorted(p for p in d.glob("gnd_capxprobe_*.csv"))
    odd = [n for n in names if n not in ALLOWED_FILES and not n.startswith("gnd_capxprobe_") and n not in subdirs]
    if odd:
        rep.fail("C1", sc, f"files FieldLog does not write: {odd[:5]} (nothing may be added to a session folder)")
    if not d.name.startswith("gnd_"):
        rep.warn("C1", sc, "folder name does not start with gnd_ (renamed?)")
    if not trial_files:
        rep.fail("C1", sc, "no trial files gnd_capxprobe_NNN.csv")
    # C2 checksums
    sums = {}
    if (d / "SHA256SUMS").is_file():
        for ln in (d / "SHA256SUMS").read_text().splitlines():
            if ln.strip():
                h, n = ln.split(None, 1)
                sums[n.strip().lstrip("*")] = h
        bad = [n for n, h in sums.items() if not (d / n).is_file() or PL.sha256_file(d / n) != h]
        unl = [n for n in names if n not in sums and n != "SHA256SUMS" and n not in subdirs]
        if bad:
            rep.fail("C2", sc, f"checksum mismatch or missing file: {bad[:5]} (a file was edited or copied incompletely)")
        else:
            rep.ok("C2", sc, f"{len(sums)} files match SHA256SUMS")
        if unl:
            rep.warn("C2", sc, f"not covered by SHA256SUMS: {unl[:5]}")
    # C3 session.yaml
    try:
        meta = yaml.safe_load((d / "session.yaml").read_text())
    except Exception as e:
        rep.fail("C3", sc, f"session.yaml unreadable: {e}")
        return None
    if meta.get("schema_version") != 2:
        rep.fail("C3", sc, "not a FieldLog v2 session (schema_version != 2)")
    label = meta.get("label")
    if a.expect == "field":
        if label != "FIELD":
            rep.fail("C3", sc, f"label {label} (not FIELD): reasons {meta.get('simulation_reasons')}")
        if not meta.get("field_flag") or meta.get("use_sim_time") is not False or not str(meta.get("vehicle_id") or ""):
            rep.fail("C3", sc, "FIELD needs --field, use_sim_time false and a roster vehicle id")
        ops = {o.get("operator_code") for o in meta.get("operators") or []}
        if len(ops) < 2:
            rep.fail("C3", sc, f"{len(ops)} operator code(s): the provenance gate needs two people")
        if meta.get("prereg_sha256") in (None, "") and meta.get("session_ordinal", 0) < 2000:
            rep.warn("C3", sc, "prereg_sha256 is empty on a main-collection session (pass --prereg-sha256 to make_run_sheets)")
        if not (meta.get("clock", {}).get("start", {}).get("locked") and meta.get("clock", {}).get("end", {}).get("locked")):
            rep.warn("C3", sc, "chrony was not locked at start/end (wall clock stamps unreliable)")
    else:
        rep.warn("C3", sc, f"label {label}: NOT field data (expected for a dry run)")
    if "capxfer" not in str(meta.get("codebook_version", "")):
        rep.warn("C3", sc, f"codebook {meta.get('codebook_version')} has no capxfer overlay (pass --codebook)")
    seg = seg_by_hash.get(meta.get("runsheet_sha256"))
    sched = None
    if seg is not None:
        rep.ok("C3", sc, f"run sheet = {seg['sched_session']}/{seg.get('unit', seg.get('segment'))} "
                         f"({seg.get('assemblies', seg.get('assembly'))}) of {seg['_sheets']}")
        sched = scheds.get(seg["_sheets"])
        if meta.get("config_sha256") and seg.get("config_sha256") and meta["config_sha256"] != seg["config_sha256"]:
            rep.warn("C3", sc, "FieldLog config differs from the rendered segment config (edited or another file used)")
    else:
        if seg_by_hash:
            rep.warn("C3", sc, "run sheet not found in the given --sheets folders: run order cannot be checked")
    if sched is None and a.schedule_obj is not None:
        sched = a.schedule_obj
    sidx = sched_index(sched) if sched is not None else None
    # C4 trials.csv
    hdr, trows = PL.read_csv(d / "trials.csv") if (d / "trials.csv").is_file() else ([], [])
    tr = [dict(zip(hdr, r)) for r in trows]
    by_id = {}
    for r in tr:
        if r.get("replaces_nn", "0") not in ("", "0"):
            pass
        by_id.setdefault(r["trial_id"], []).append(r)
    no_video = [r["trial_id"] for r in tr if r.get("outcome_code") not in ("0", "") and r.get("video_ok") != "1"]
    if no_video:
        (rep.warn if a.expect == "simulation" else rep.warn)("C4", sc, f"{len(no_video)} trial(s) with video_ok = 0 "
                                                                         f"(each needs the slate on video): {no_video[:4]}")
    oc = Counter(r.get("outcome_code") for r in tr)
    if oc.get("5") or oc.get("6"):
        rep.warn("C4", sc, f"{oc.get('5', 0)} aborted and {oc.get('6', 0)} voided trial(s): give the reason in operator_notes.csv")
    # per trial
    asm_seen = Counter()
    executed = []
    want_all = [int(x) for x in seg["trial_order"].split()] if seg is not None else None   # 0 = dock row (choice demo)
    row_of = {r["trial_id"]: int(r.get("runsheet_row") or 0) for r in tr}
    mods_of = sched_modules(sched) if sched is not None else {}
    n_dock = 0
    hs = {}
    for p in trial_files:
        scope = f"{sc}/{p.stem}"
        streams, thdr, nrows = read_trial(p, rep, scope)
        rws = by_id.get(p.stem, [])
        if not rws:
            rep.fail("C4", scope, "trial file has no row in trials.csv")
        elif str(nrows) != rws[-1].get("rows_n"):
            rep.fail("C4", scope, f"{nrows} data rows but trials.csv rows_n = {rws[-1].get('rows_n')}")
        ev = markers(streams)
        starts = [e for e in ev if e[2] == FC.TRIAL_EVENT_CODES["start"]]
        ends = [e for e in ev if e[2] in (FC.TRIAL_EVENT_CODES["end"], FC.TRIAL_EVENT_CODES["abort"])]
        k_row = row_of.get(p.stem, 0)
        if not ev and want_all and 1 <= k_row <= len(want_all) and want_all[k_row - 1] == 0:
            n_dock += 1                                   # WS-P2 choice demo: a dock run row (no probe markers)
            continue
        if a.static_ok and not ev:
            rep.ok("C6", scope, "no probe markers (static / manual trial)")
            continue
        if len(starts) != 1 or len(ends) != 1:
            rep.warn("C6", scope, f"{len(starts)} start and {len(ends)} end/abort marker(s): the loader keeps only complete "
                                  f"start..end pairs (was the probe runner started inside this FieldLog trial?)")
            if not starts or not ends:
                continue
        t0, num, _, asm_code = starts[0]
        t1, num1, evc, _ = ends[-1]
        if num1 != num:
            rep.fail("C6", scope, f"start marker trial {num} but end marker trial {num1}")
        if evc == FC.TRIAL_EVENT_CODES["abort"]:
            rep.warn("C6", scope, f"trial {num} aborted (e-stop): excluded by prereg §7(c); note the reason")
            executed.append((p.stem, num, "abort"))
            continue
        executed.append((p.stem, num, "ok"))
        # C7 assembly
        try:
            asm, spec, frames = build_assembly(window(streams, t0 - int(0.2e9), t1), asm_code, params)
        except Exception as e:
            rep.fail("C7", scope, str(e))
            continue
        asm_seen[asm.name] += 1
        want_m = mods_of.get(int(num))
        got_m = sorted(m for m, *_ in spec)
        if want_m is not None and got_m != want_m:
            rep.fail("C7", scope, f"trial {num} logged with module(s) {'+'.join(got_m)}, the schedule says "
                                  f"{'+'.join(want_m)} (wrong module driving alone after a split?)")
        tm_, dm_ = streams.get(FC.SRC["TRIAL"], (np.zeros(0, np.int64), {}))
        k_ = [i for i in range(len(tm_)) if tm_[i] == t0]
        if k_ and "imu_slot" in dm_ and np.isfinite(dm_["imu_slot"][k_[0]]):          # session-mode markers (WS-P)
            from capxfer.field.schedule import sensor_modules
            im, lm = sensor_modules(asm)
            want = (int(im[1:]) if im else 0, int(lm[1:]) if lm else 0)
            got = (int(dm_["imu_slot"][k_[0]]), int(np.nan_to_num(dm_["lidar_slot"][k_[0]])))
            if got != want:
                rep.fail("C7", scope, f"runner relayed IMU/lidar of slots {got}, the assembly needs {want}")
        dur = (t1 - t0) / 1e9
        t_s = sidx.get((asm.name, num)) if sidx is not None else None
        sch = SCHF.trial_schedule(asm, t_s, duration_s=dur)          # WS-P2: full library or the trial's probe subset
        if abs(dur - sch.T) > 0.3:
            rep.warn("C6", scope, f"trial lasted {dur:.2f} s, its probes ({'+'.join(SCHF.trial_pids(t_s)) if t_s else 'inferred'}) "
                                  f"take {sch.T:.2f} s")
        # C8 echoes vs schedule
        st = window(streams, t0 - int(0.2e9), t1 + int(0.5e9))
        ctl = st.get(FC.SRC["CTL"], (np.zeros(0, np.int64), {}))
        mv = [ctl[0][k] for k in range(len(ctl[0])) if any(abs(ctl[1][c][k]) > 1e-6 for c in FC.CTL_COLS
                                                           if np.isfinite(ctl[1][c][k]))]
        t_motion = int(min(mv)) if mv else t1
        ech = echoes(st, t0 - int(0.2e9), t_motion)
        restore = echoes(st, t_motion, t1 + int(0.5e9))
        if sidx is not None:
            t = sidx.get((asm.name, num))
            if t is None:
                rep.fail("C8", scope, f"trial {num} ({asm.name}) is not in the schedule: the loader will refuse the session")
            else:
                why = []
                for f in t["faults"]:
                    if f["wheel"] == "none" or ":" not in f["wheel"]:
                        mod = f.get("module") or asm.modules[0][0]
                        hit = [e for e in ech if e[1] == mod and int(e[3]) == MODE_CODE[f["mode"]]]
                        if not hit and f["mode"] not in OPERATOR_MODES:
                            why.append(f"no echo for {f['mode']}")
                        continue
                    m, c = f["wheel"].split(":")
                    hit = [e for e in ech if e[1] == m and np.isfinite(e[2]) and int(e[2]) == PL.CORNERS.index(c)]
                    modes = {int(e[3]) for e in hit if np.isfinite(e[3])}
                    if f["mode"] in OPERATOR_MODES:
                        if modes - {MODE_CODE[f["mode"]], 0}:
                            why.append(f"{f['wheel']}: echo {modes} contradicts {f['mode']}")
                        continue
                    if MODE_CODE[f["mode"]] not in modes:
                        why.append(f"{f['wheel']}: scheduled {f['mode']}, echoes {sorted(modes) or 'none'}")
                    elif f["mode"] == "weak":
                        gs = [e[4] for e in hit if int(e[3]) == 1]
                        if not any(np.isfinite(gv) and abs(gv - f["g"]) <= 1e-3 for gv in gs):
                            why.append(f"{f['wheel']}: g echo {gs} != {f['g']}")
                    if not any(e[1] == m and np.isfinite(e[2]) and int(e[2]) == PL.CORNERS.index(c) and int(e[3]) == 0
                               for e in restore):
                        why.append(f"{f['wheel']}: no 'ok' echo after the trial (fault cleared?)")
                sched_w = {f["wheel"] for f in t["faults"] if ":" in f["wheel"]}
                extra = [e for e in ech if np.isfinite(e[3]) and int(e[3]) in (1, 2, 3) and np.isfinite(e[2])
                         and f"{e[1]}:{PL.CORNERS[int(e[2])] if int(e[2]) < 4 else '?'}" not in sched_w]
                if extra:
                    why.append(f"unscheduled firmware fault echoes: {extra[:2]}")
                badtrial = [e for e in ech if np.isfinite(e[5]) and int(e[5]) != num]
                if badtrial:
                    why.append(f"echo trial field {int(badtrial[0][5])} != {num}")
                if why:
                    rep.fail("C8", scope, "fault echo does not match the schedule (prereg §7 b exclusion): " + "; ".join(why[:4]))
        # C10 rates and gaps
        problems, warns = [], []
        for name_, srcs in [("CTL", [FC.SRC["CTL"]]), ("IMU", [FC.SRC["IMU"]]), ("TRUTH", [FC.SRC["TRUTH"]]),
                            ("WHL", [FC.SRC[f"WHL_{m}"] for m, *_ in spec]),
                            ("LODOM", [FC.SRC["LODOM"]] if asm.has_lidar() else []),
                            ("CUR", [FC.SRC[f"CUR_{m}"] for m, *_ in spec if FC.SRC[f"CUR_{m}"] in streams])]:
            exp, mn = RATES[name_]
            for s in srcs:
                tm = st.get(s, (np.zeros(0, np.int64), {}))[0]
                tm = tm[(tm >= t0) & (tm <= t1)]
                r, gmax = rate_and_gaps(tm, t0, t1)
                lab = SRC_NAME.get(s, s)
                if r is None:
                    if name_ in ("CTL", "WHL", "TRUTH", "IMU"):
                        problems.append(f"{lab}: no samples")
                    elif name_ == "LODOM":
                        warns.append(f"{lab}: no lidar odometry (E2 rung unavailable for this trial)")
                    continue
                if r < mn:
                    problems.append(f"{lab}: {r:.1f} Hz < minimum {mn}")
                elif r < 0.9 * exp:
                    warns.append(f"{lab}: {r:.1f} Hz (expected {exp})")
                if gmax is not None and gmax > 5.0 / exp and name_ != "LODOM":
                    warns.append(f"{lab}: gap {gmax * 1000:.0f} ms")
        # prereg 7(d): robot-visible gap > 100 ms inside P1/P5 holds; truth coverage of untested legs
        robot = [FC.SRC[f"WHL_{m}"] for m, *_ in spec] + [FC.SRC["IMU"]] + ([FC.SRC["LODOM"]] if asm.has_lidar() else [])
        lat = float(params["truth"]["apriltag_latency_s"])
        excl, undef = [], []
        for pid, leg, ta, tb, _ in sch.windows():
            a_ns, b_ns = t0 + int(ta * 1e9), t0 + int(tb * 1e9)
            key = f"{pid}{'+' if leg > 0 else '-'}"
            if pid in ("P1", "P5"):
                for s in robot:
                    tm = st.get(s, (np.zeros(0, np.int64), {}))[0]
                    lim = 0.25 if s == FC.SRC["LODOM"] else GAP_ROBOT_S          # rf2o is 10 Hz by design
                    if max_gap_in(tm, a_ns, b_ns) > lim:
                        excl.append(f"{SRC_NAME.get(s, s)} in {key}")
            else:
                tm = st.get(FC.SRC["TRUTH"], (np.zeros(0, np.int64), {}))[0]
                a2, b2 = a_ns + int(lat * 1e9), b_ns + int(lat * 1e9)
                n = int(((tm >= a2) & (tm <= b2)).sum())
                want = (tb - ta) * RATES["TRUTH"][0]
                if n < (1 - TRUTH_MISSING_MAX) * want:
                    undef.append(f"{key} ({n}/{want:.0f})")
        if problems:
            rep.fail("C10", scope, "; ".join(problems[:5]))
        if warns:
            rep.warn("C10", scope, "; ".join(warns[:5]))
        if excl:
            rep.warn("C10", scope, f"robot-visible gap > 100 ms in a P1/P5 hold ({', '.join(excl[:4])}): trial excluded (prereg §7 d)")
        if undef:
            rep.warn("C10", scope, f"AprilTag missing > 20 % of hold window: {', '.join(undef[:4])} (target undefined)")
        # C11 liveness
        moving = any(abs(ctl[1]["cmd_vx"][k]) > 0.1 for k in range(len(ctl[0])) if np.isfinite(ctl[1]["cmd_vx"][k]))
        if moving:
            for m, *_ in spec:
                s = FC.SRC[f"WHL_{m}"]
                if s not in st:
                    continue
                dd = st[s][1]
                cnt = np.stack([dd[f"cnt_{c}_{FC.mk(m)}"] for c in PL.CORNERS], 1)
                dut = np.stack([dd[f"duty_{c}_{FC.mk(m)}"] for c in PL.CORNERS], 1)
                span = np.nanmax(cnt, 0) - np.nanmin(cnt, 0) if np.isfinite(cnt).any() else np.zeros(4)
                dead = [c for c, sp in zip(PL.CORNERS, span) if not sp > 0]
                t_ = sidx.get((asm.name, num)) if sidx else None
                blocked = {f["wheel"].split(":")[1] for f in (t_ or {}).get("faults", [])
                           if f.get("mode") in ("coast", "brake", "roller_locked") and f["wheel"].startswith(m + ":")}
                dead = [c for c in dead if c not in blocked]
                if dead:
                    rep.fail("C11", scope, f"{m}: encoder counts never change on {dead} although the robot was commanded to move")
                if np.isfinite(dut).any() and np.nanmax(np.abs(dut)) > 1.0 + 1e-6:
                    rep.fail("C11", scope, f"{m}: duty outside [-1, 1] (effort must be the PI duty before fault scaling)")
                if not (np.isfinite(dut).any() and np.nanmax(np.abs(dut)) > 0.01):
                    rep.warn("C11", scope, f"{m}: duty is always ~0 while moving (effort[] not filled?)")
            if FC.SRC["IMU"] in st and len(st[FC.SRC["IMU"]][0]):
                wz = st[FC.SRC["IMU"]][1]["wz_radps"]
                if np.nanstd(wz) < 1e-4:
                    rep.warn("C11", scope, "IMU yaw rate is constant (wrong module's IMU, or IMU frozen)")
            if FC.SRC["TRUTH"] in st and len(st[FC.SRC["TRUTH"]][0]) > 2:
                gx = st[FC.SRC["TRUTH"]][1]["gt_x_m"]
                gy = st[FC.SRC["TRUTH"]][1]["gt_y_m"]
                if np.nanmax(np.hypot(gx - np.nanmedian(gx), gy - np.nanmedian(gy))) < 0.02:
                    rep.warn("C11", scope, "AprilTag pose never moves > 2 cm although the robot moved (wrong tag id?)")
        hs[p.stem] = num
    for chk, what in (("C5", "trial-file schema and dtypes"), ("C6", "trial markers"), ("C7", "assembly frames"),
                      ("C8", "fault echoes match the schedule" if sidx is not None else "no schedule given: echoes not checked"),
                      ("C10", "rates, gaps, truth coverage"), ("C11", "encoders, duty, IMU, tag alive")):
        if not any(i[1] == chk and i[2].startswith(sc) for i in rep.items):
            (rep.ok if (chk != "C8" or sidx is not None) else rep.warn)(chk, sc, f"{what}: all {len(trial_files)} trial file(s)")
    # C9 order vs run sheet
    if seg is not None:
        want = [int(x) for x in seg["trial_order"].split()]
        rowmap = {r["trial_id"]: int(r.get("runsheet_row") or 0) for r in tr}
        wrong = []
        for stem, num, _ in executed:
            k = rowmap.get(stem, 0)
            if 1 <= k <= len(want) and want[k - 1] != num:
                wrong.append(f"row {k} ran trial {num}, sheet says {want[k - 1]}")
            elif not (1 <= k <= len(want)):
                wrong.append(f"{stem}: run-sheet row {k} out of range")
        if wrong:
            rep.warn("C9", sc, f"{len(wrong)} trial(s) out of run-sheet order: {wrong[:3]} (labels follow the markers; note why)")
        else:
            rep.ok("C9", sc, f"{len(executed)} trials in run-sheet order")
        n_want = sum(1 for x in want if x != 0)
        if len([e for e in executed if e[2] == "ok"]) < n_want:
            rep.warn("C9", sc, f"{len([e for e in executed if e[2] == 'ok'])} of {n_want} scheduled trials completed")
        if n_dock:
            rep.ok("C9", sc, f"{n_dock} dock-run row(s) (choice demo; no probe markers expected)")
    # C12 housekeeping
    hh, hrows = PL.read_csv(d / "housekeeping.csv") if (d / "housekeeping.csv").is_file() else ([], [])
    if hh:
        hi = {c: i for i, c in enumerate(hh)}
        vb = [float(r[hi["v_batt_v"]]) for r in hrows if "v_batt_v" in hi and r[hi["v_batt_v"]] not in ("",)]
        if not vb:
            rep.warn("C12", sc, "no battery voltage stream: fill v_batt_* in block_log.csv by multimeter")
        elif min(vb) < 10.8:
            rep.warn("C12", sc, f"pack voltage fell to {min(vb):.2f} V (< 10.8 V: swap packs, card rule)")
        if not hrows:
            rep.warn("C12", sc, "housekeeping.csv has no clock rows")
    # C13 events
    eh, erows = PL.read_csv(d / "events.csv") if (d / "events.csv").is_file() else ([], [])
    if eh != ["t_mono_ns", "t_wall_ns", "trial_id", "event_code", "value"]:
        rep.fail("C13", sc, f"events.csv header {eh}")
    else:
        codes = Counter(r[3] for r in erows)
        if not codes.get("1") or not codes.get("2"):
            rep.warn("C13", sc, "session_start or session_end event missing (logger killed?)")
        if codes.get("7"):
            rep.warn("C13", sc, f"{codes['7']} bad_input event(s) (inbox line refused: wrong computer or typo?)")
        lw = Counter(r[4] for r in erows if r[3] == "8")
        if lw:
            rep.warn("C13", sc, f"logger warnings (value: 1 rate, 2 disk, 3 missing publisher, 29 chrony override): {dict(lw)}")
    # C14 FieldLog verifier
    if not a.no_verify:
        try:
            if str(PL.FIELDLOG) not in sys.path:
                sys.path.insert(0, str(PL.FIELDLOG))
            import verify_session as VS
            r = VS.verify(d)
            v = r["verdict"]
            msg = f"verify_session: {v}" + (f" ({'; '.join(r['reasons'][:3])})" if r.get("reasons") else "")
            if v == "FIELD_OK":
                rep.ok("C14", sc, msg)
            elif v == "FIELD_WITH_WARNINGS" or (v == "SIMULATION" and a.expect == "simulation"):
                rep.warn("C14", sc, msg + (" - explain every warning in operator_notes.csv" if v != "SIMULATION" else ""))
            else:
                rep.fail("C14", sc, msg)
        except Exception as e:
            rep.warn("C14", sc, f"FieldLog verify_session could not run: {e}")
    # C15 loader
    if not a.no_loader:
        try:
            import torch  # noqa: F401
            from capxfer.field import loader as L
            raws = L.load_session(d, allow_unverified=True, schedule=sched)
            desc = ", ".join(f"{x['meta']['assembly'].name}: {x['counts'].shape[0]} trials" for x in raws)
            mism = sum(1 for x in raws for ok in x["meta"]["labels"].get("schedule_match", []) if not ok)
            rep.ok("C15", sc, f"capxfer loader read {desc}" + (f"; {mism} schedule mismatch(es)" if mism else ""))
        except ImportError:
            rep.warn("C15", sc, "torch not installed here: loader check skipped (it runs again on the analysis side)")
        except Exception as e:
            rep.fail("C15", sc, f"capxfer loader failed: {type(e).__name__}: {e}")
    return {"dir": d.name, "assemblies": dict(asm_seen), "trials": len(trial_files), "label": label,
            "done": [(num, ok) for _, num, ok in executed], "sched": sched,
            "seg": seg and f"{seg['sched_session']}/{seg.get('unit', seg.get('segment'))}"}


# ------------------------------------------------------------------------------------------------ manual CSVs
def check_link_rows(md, rep, scheds):
    """M4 (WS-P2): episode_log.csv against the sealed link units (MASK builds, DEMO episode phases); choice_runs.csv
    (DEMO docks: module of the episode, its sealed start mark, each module at most once per episode)."""
    units = {}
    for sc in scheds:
        for u in sc.get("link_units") or []:
            units[(u["episode"], u["phase"], u["module"])] = u
    p = md / "episode_log.csv"
    if p.is_file() and units:
        hdr, rows = PL.read_csv(p)
        errs = []
        for i, r in enumerate(rows, start=2):
            row = dict(zip(hdr, r))
            u = units.get((row.get("episode"), row.get("phase"), row.get("module")))
            if u is None:
                errs.append(f"line {i}: ({row.get('episode')}, {row.get('phase')}, {row.get('module')}) is not a build / "
                            f"episode phase of the schedule")
                continue
            for c in ("first_trial", "last_trial"):
                try:
                    if int(row.get(c, "0")) not in u["trials"]:
                        errs.append(f"line {i}: {c} {row.get(c)} is not a trial of {u['episode']} {u['phase']} {u['module']}")
                except ValueError:
                    pass
            if row.get("fault_carrier_seen", "").strip() != u["fault_carrier"]:
                errs.append(f"line {i}: physical fault carrier seen {row.get('fault_carrier_seen')!r}, schedule "
                            f"{u['fault_carrier']!r}")
            seen = sorted(x for x in row.get("zip_tie_wheels_seen", "").split(";") if x.strip())
            if seen != sorted(u["zip_wheels"]):
                errs.append(f"line {i}: zip-tied wheels seen {seen}, schedule {sorted(u['zip_wheels'])}")
        if errs:
            rep.fail("M4", "manual/episode_log.csv", f"{len(errs)} link problem(s): {errs[:4]}")
        else:
            rep.ok("M4", "manual/episode_log.csv", f"{len(rows)} row(s) agree with the sealed schedule")
    eps = {e["episode"]: e for sc in scheds for e in ((sc.get("choice") or {}).get("episodes") or [])}
    p = md / "choice_runs.csv"
    if p.is_file() and eps:
        hdr, rows = PL.read_csv(p)
        errs, seen_dock = [], set()
        for i, r in enumerate(rows, start=2):
            row = dict(zip(hdr, r))
            e = eps.get(row.get("episode"))
            if e is None:
                errs.append(f"line {i}: episode {row.get('episode')} not in the DEMO schedule")
                continue
            try:
                j = int(row.get("dock_in_episode", "0"))
            except ValueError:
                j = 0
            m = row.get("module")
            if not (1 <= j <= len(e["modules"])) or m not in e["modules"]:
                errs.append(f"line {i}: {e['episode']} dock {j} by {m}: not a dock slot / module of the episode")
            elif row.get("start_pose_id") != str(e["start_pose_id"][m]):
                errs.append(f"line {i}: {m} started from mark {row.get('start_pose_id')}, sheet says {e['start_pose_id'][m]}")
            if (row.get("invalid_reason") or "none") == "none" and row.get("repeat_of", "") == "":
                key = (e["episode"], m)
                if key in seen_dock:
                    errs.append(f"line {i}: {m} docked twice in {e['episode']} (the dock list is deduplicated)")
                seen_dock.add(key)
        if errs:
            rep.fail("M4", "manual/choice_runs.csv", f"{len(errs)} problem(s): {errs[:4]}")
        else:
            rep.ok("M4", "manual/choice_runs.csv", f"{len(rows)} row(s) agree with the DEMO sheet")


def check_manual(md, rep, session_dirs, scheds=()):
    md = Path(md)
    known = set(PL.SCHEMAS) | {"measurements"}
    names = {d.name for d in session_dirs}
    for p in sorted(md.glob("*.csv")):
        name = p.stem
        if name not in known:
            rep.warn("M1", f"manual/{p.name}", "unknown manual file (not in CSV_SPEC.md)")
            continue
        spec = PL.MEASUREMENT_COLUMNS if name == "measurements" else PL.SCHEMAS[name]
        hdr, rows = PL.read_csv(p)
        if hdr != [c[0] for c in spec]:
            rep.fail("M1", f"manual/{p.name}", f"header differs from the template (copy templates/{p.name}): {hdr[:6]}...")
            continue
        errs, refs = [], []
        for i, r in enumerate(rows, start=2):
            if not any(x.strip() for x in r):
                continue
            if len(r) != len(spec):
                errs.append(f"line {i}: {len(r)} cells, expected {len(spec)}")
                continue
            for c, v in zip(spec, r):
                if name == "measurements" and c[0] == "value":
                    st = r[[s[0] for s in spec].index("status")]
                    if st == "MEASURED" and not v.strip():
                        errs.append(f"line {i}: status MEASURED but value empty")
                    elif v.strip():
                        why = PL.check_cell(c, v)
                        if why:
                            errs.append(f"line {i} {c[0]}: {why}")
                    continue
                why = PL.check_cell(c, v)
                if why:
                    errs.append(f"line {i} {c[0]}: {why}")
            row = dict(zip([c[0] for c in spec], r))
            if "fieldlog_dir" in row and names and row["fieldlog_dir"] and row["fieldlog_dir"] not in names:
                refs.append(row["fieldlog_dir"])
        if errs:
            rep.fail("M2", f"manual/{p.name}", f"{len(errs)} problem(s): {errs[:4]}")
        else:
            rep.ok("M2", f"manual/{p.name}", f"{len(rows)} row(s) valid")
        if refs:
            rep.warn("M3", f"manual/{p.name}", f"fieldlog_dir not among the checked sessions: {sorted(set(refs))[:3]}")
        if name == "measurements":
            todo = [r[0] for r in rows if len(r) == len(spec) and r[[s[0] for s in spec].index("status")] == "TODO"]
            if todo:
                rep.warn("M2", f"manual/{p.name}", f"{len(todo)} item(s) still TODO: {todo[:6]}")
    if session_dirs:
        si = md / "session_index.csv"
        if si.is_file():
            _, rows = PL.read_csv(si)
            listed = {r[0] for r in rows if r}
            miss = sorted(names - listed)
            if miss:
                rep.warn("M3", "manual/session_index.csv", f"{len(miss)} session folder(s) without a row: {miss[:3]}")
        else:
            rep.warn("M3", "manual", "session_index.csv missing")
    check_link_rows(md, rep, [s for s in scheds if s])


# ------------------------------------------------------------------------------------------------ main
def find_sessions(paths):
    out, manual = [], []
    for p in paths:
        p = Path(p)
        if (p / "session.yaml").is_file():
            out.append(p)
        else:
            out += sorted(q.parent for q in p.glob("sessions/*/session.yaml"))
            out += sorted(q.parent for q in p.glob("*/session.yaml") if q.parent.name.startswith("gnd_"))
            if (p / "manual").is_dir():
                manual.append(p / "manual")
    return sorted(set(out)), manual


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--sheets", action="append", default=[], help="run-sheet folder from make_run_sheets.py (repeatable)")
    ap.add_argument("--schedule", help="schedule.json (if no --sheets)")
    ap.add_argument("--manual", help="folder with the manual CSVs")
    ap.add_argument("--expect", choices=["field", "simulation"], default="field")
    ap.add_argument("--static-ok", action="store_true", help="trials without probe markers are OK (AprilTag static set)")
    ap.add_argument("--out", help="write the text report here (never inside a session folder)")
    ap.add_argument("--csv", help="also write the report as CSV")
    ap.add_argument("--no-loader", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    sessions, manual = find_sessions(a.paths)
    for o in (a.out, a.csv):
        if o and any(PL.inside(o, s) for s in sessions):
            sys.exit(f"refusing to write {o} inside a session folder")
    a.schedule_obj = json.loads(Path(a.schedule).read_text()) if a.schedule else None
    seg_by_hash, scheds = load_sheets(a.sheets)
    params = A.load_params(PARAMS)
    rep = Report()
    summ = []
    for d in sessions:
        try:
            summ.append(check_session(d, rep, a, params, seg_by_hash, scheds))
        except Exception as e:                       # a crash of the validator is itself a FAIL for that session
            rep.fail("C0", d.name, f"validator crashed: {type(e).__name__}: {e}")
    check_episodes(summ, rep)
    all_scheds = list(scheds.values()) + ([a.schedule_obj] if a.schedule_obj is not None else [])
    for md in ([Path(a.manual)] if a.manual else []) + manual:
        check_manual(md, rep, sessions, all_scheds)
    if not sessions and not (a.manual or manual):
        sys.exit("nothing to validate (no session.yaml found)")
    lines = [f"capxfer CSV validation  expect={a.expect}  sessions={len(sessions)}", ""]
    for s in summ:
        if s:
            lines.append(f"  {s['dir']}: {rep.verdict(s['dir'])}  label={s['label']}  trial files={s['trials']}  "
                         f"assemblies={s['assemblies']}  segment={s['seg']}")
    lines.append("")
    for lv in ("FAIL", "WARN"):
        for it in rep.items:
            if it[0] == lv:
                lines.append(f"  {it[0]:4s} {it[1]:4s} {it[2]}: {it[3]}")
    n = Counter(i[0] for i in rep.items)
    lines += ["", f"checks: {n.get('PASS', 0)} pass, {n.get('WARN', 0)} warn, {n.get('FAIL', 0)} fail",
              f"VERDICT: {rep.verdict()}", "This is a data-quality screen. It never modifies the session folders."]
    txt = "\n".join(lines) + "\n"
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["level", "check", "scope", "message"])
            w.writerows(rep.items)
    return {"PASS": 0, "WARN": 1, "FAIL": 2}[rep.verdict()]


if __name__ == "__main__":
    sys.exit(main())

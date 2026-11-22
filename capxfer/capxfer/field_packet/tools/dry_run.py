#!/usr/bin/env python3
"""Dry run of the packet tools on SYNTHETIC SIMULATION sessions (no robot, no field data). AI-side / CI check.

  .venv/bin/python capxfer/field_packet/tools/dry_run.py [--work ~/.cache/capxfer/P/dryrun] [--trials 2]

1. make_run_sheets.py main + extra hooktest with throw-away seeds
2. capxfer.field.synth_session writes SIMULATION FieldLog sessions (every assembly type, current wired) from that schedule
3. validate_csv_session.py on them (--expect simulation): must give no FAIL; the capxfer loader must read them
4. a corrupted copy (one edited cell, one text cell) must FAIL (C2 checksum + C5 dtype)
5. manual CSV templates + a synthetic filled set validate; a broken set must FAIL
6. check_hook_test.py --setup floor on a simulated hook-test session, truth_check.py motion, measurements_to_params.py,
   docking_to_analysis.py on synthetic rows
7. data volume: bytes per trial per assembly type, raw and zipped
8. PROTOCOL_V2 MASK: make_run_sheets.py extra masking (realv2_design.expand_schedule); one (module, wheel) condition
   simulated in each of its parts (parent assemblies + the module alone) -> validator (C7 module check, E1 link, M4
   episode_log) + loader link labels; a physical fault logged for a MASK build must FAIL M4
9. PROTOCOL_V2 DEMO: capxfer.field.choose_module dry run (DEMO run sheets, synthetic pair-probe sessions, synthetic
   bundle and lambda) -> the certificate sends the module without the zip ties
Everything is written under --work; nothing in the repository or in crane_ml changes.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import packet_lib as PL  # noqa: E402

PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=str(PL.REPO), CUDA_VISIBLE_DEVICES="")


def run(args, expect=None):
    r = subprocess.run([PY, *map(str, args)], capture_output=True, text=True, env=ENV, cwd=str(PL.REPO))
    if expect is not None and r.returncode not in (expect if isinstance(expect, tuple) else (expect,)):
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(map(str, args))} -> exit {r.returncode}, expected {expect}")
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=str(Path.home() / ".cache" / "capxfer" / "P" / "dryrun"))
    ap.add_argument("--trials", type=int, default=2)
    a = ap.parse_args(argv)
    W = Path(a.work)
    shutil.rmtree(W, ignore_errors=True)
    (W / "sealed").mkdir(parents=True)
    (W / "sealed" / "seed_main.txt").write_text("5eed" * 16 + "\n")
    (W / "sealed" / "seed_hook.txt").write_text("ab12" * 16 + "\n")
    T = HERE
    res = {}
    # 1 run sheets
    r = run([T / "make_run_sheets.py", "main", W / "sheets_main", "--seed-file", W / "sealed" / "seed_main.txt"], 0)
    print(r.stdout.strip())
    r = run([T / "make_run_sheets.py", "extra", "hooktest", W / "sheets_hook", "--module", "M1",
             "--seed-file", W / "sealed" / "seed_hook.txt"], 0)
    print(r.stdout.strip())
    # 2 synthetic sessions
    from capxfer.field import synth_session as SS
    sched = json.loads((W / "sheets_main" / "schedule.json").read_text())
    sess = W / "return" / "sessions"
    sess.mkdir(parents=True)
    dirs = {}
    for k, asm in enumerate(("single", "pair", "L3", "square4")):
        d, _ = SS.make_session(sess, asm, n_trials=a.trials, seed=11 + k, schedule=sched, current_wired=True,
                               codebook_dir=W / "_cb", device="cpu")
        dirs[asm] = Path(d)
        print(f"synthetic {asm}: {d}")
    for sub in ("_codebook", "_runsheets", "_no_roster"):
        shutil.rmtree(sess / sub, ignore_errors=True)
    # 3 validate
    r = run([T / "validate_csv_session.py", *dirs.values(), "--sheets", W / "sheets_main", "--schedule",
             W / "sheets_main" / "schedule.json", "--expect", "simulation", "--out", W / "validate_report.txt",
             "--csv", W / "validate_report.csv"], (0, 1))
    txt = r.stdout
    print(txt.split("\n\n")[1] if "\n\n" in txt else txt)
    fails = [ln for ln in txt.splitlines() if ln.strip().startswith("FAIL")]
    loader_ok = sum(1 for ln in open(W / "validate_report.csv") if ",C15," in ln and ln.startswith("PASS"))
    assert not fails, fails
    assert loader_ok == 4, f"loader passed on {loader_ok}/4 sessions"
    res["validate"] = "no FAIL; loader read all 4 types"
    # 4 corruption
    bad = W / "corrupt" / dirs["pair"].name
    shutil.copytree(dirs["pair"], bad)
    for p in bad.iterdir():
        p.chmod(0o644)
    f = sorted(bad.glob("gnd_capxprobe_*.csv"))[0]
    lines = f.read_text().splitlines()
    cols = lines[0].split(",")
    j = cols.index("cmd_vx")
    for i in range(1, len(lines)):
        c = lines[i].split(",")
        if c[3] == "7" and c[j] not in ("", "0"):
            c[j] = "nan"
            lines[i] = ",".join(c)
            break
    f.write_text("\n".join(lines) + "\n")
    r = run([T / "validate_csv_session.py", bad, "--schedule", W / "sheets_main" / "schedule.json", "--expect", "simulation",
             "--no-loader"], 2)
    assert "C2" in r.stdout and "C5" in r.stdout, r.stdout
    res["corruption"] = "edited cell -> FAIL C2 (checksum) and C5 (text 'nan')"
    # 5 manual
    man = W / "return" / "manual"
    man.mkdir()
    for p in (PL.PACKET / "templates").glob("*.csv"):
        shutil.copy(p, man / p.name)
    shutil.copy(PL.PACKET / "MEASUREMENTS_SHEET.csv", man / "measurements.csv")
    d0 = dirs["single"].name
    with open(man / "session_index.csv", "a", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        for asm, d in dirs.items():
            w.writerow([d.name, "main", "S01", 1, asm, "M1=m1", "M1=SYNTH", "A", "vinyl", "2026-11-21", 901, 902, 0.0, 0.0, 0.0,
                        "0" * 64, "0" * 64, "synthetic.mp4", "", "", "SYNTHETIC dry run"])
    with open(man / "block_log.csv", "a", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerow([d0, "S01", 1, 1, "single", "", "10:00", "10:05", 1, 3, "", "", "", "", "",
                                                     "", "", "", "", "", "", 12.4, 12.3, 0, 2, 0, "SYNTHETIC"])
    r = run([T / "validate_csv_session.py", W / "return", "--sheets", W / "sheets_main", "--expect", "simulation",
             "--no-loader", "--no-verify"], (0, 1))
    assert not [ln for ln in r.stdout.splitlines() if ln.strip().startswith("FAIL")], r.stdout
    man2 = W / "manual_bad"
    shutil.copytree(man, man2)
    with open(man2 / "block_log.csv", "a", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerow([d0, "S01", 2, 2, "single", "F6", "10:10", "10:20", 0, 0, "", "", "", "", "",
                                                     "", "", "M1", 7.5, "fl", 0.15, 12.2, 12.1, 0, 10, 0, "too heavy"])
    r = run([T / "validate_csv_session.py", dirs["single"], "--manual", man2, "--expect", "simulation", "--no-loader",
             "--no-verify"], 2)
    assert "payload_mass_kg" in r.stdout, r.stdout
    res["manual"] = "templates + synthetic rows validate; 7.5 kg payload / 0.15 m offset -> FAIL M2"
    # 6 hook test (simulated on the floor), truth check, converters
    hs = json.loads((W / "sheets_hook" / "schedule.json").read_text())
    dh, _ = SS.make_session(W / "hook", "single", n_trials=6, seed=3, schedule=hs, current_wired=True, codebook_dir=W / "_cb",
                            device="cpu")
    r = run([T / "check_hook_test.py", dh, "--sheets", W / "sheets_hook", "--setup", "floor", "--csv", W / "hook_results.csv"])
    print(r.stdout.strip().splitlines()[0])
    res["hooktest_floor_sim"] = r.stdout.strip().splitlines()[-1]
    r = run([T / "truth_check.py", "motion", *dirs.values(), "--csv", W / "truth_motion.csv"], 0)
    res["truth_motion"] = r.stdout.strip().splitlines()[-1]
    mm = W / "measurements_synth.csv"
    hdr, rows = PL.read_csv(PL.PACKET / "MEASUREMENTS_SHEET.csv")
    fake = {"MASS_M1": 5.3, "ROLL5_M1": 1.50, "ROLL5_M2": 1.51, "CPR10_M1": 65330, "FLOOR_A_MU": 38.0, "SHEET1_MU": 14.0,
            "ROLLER_FL_M1": 1, "ROLLER_FL_M2": 1, "BACKDRIVE_M1": 12.0}
    with open(mm, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(hdr)
        for r_ in rows:
            if r_[0] in fake:
                r_[3], r_[9] = str(fake[r_[0]]), "MEASURED"
            w.writerow(r_)
    r = run([T / "measurements_to_params.py", mm, "--out", W / "derived" / "params_from_measurements.yaml"], 0)
    res["measurements"] = r.stdout.strip().splitlines()[-2]
    dr = W / "docking_runs_synth.csv"
    with open(dr, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(PL.header("docking_runs"))
        for i, (code, oc, inv, rep) in enumerate([("A", "success", "none", ""), ("B", "collision", "none", ""),
                                                  ("C", "success", "executor_crash", ""), ("D", "infeasible", "none", ""),
                                                  ("C", "success", "none", "3")], start=1):
            w.writerow(["gnd_synth", 1, min(i, 4), i, code, 2, "10:00:00", "done", oc, int(oc == "collision"), 1, 1, inv, rep, "", ""])
    key = W / "sealed" / "dock_arm_key_synth.csv"
    key.write_text("arm_code,arm\nA,proposed\nB,omni\nC,oracle\nD,analytic_bayes\n")
    r = run([T / "docking_to_analysis.py", dr, "--arm-key", key, "--out", W / "derived" / "docking.csv"], 0)
    res["docking"] = r.stdout.strip()
    # 7 data volume
    vol = {}
    for asm, d in dirs.items():
        fs = sorted(d.glob("gnd_capxprobe_*.csv"))
        raw = sum(p.stat().st_size for p in fs) / len(fs)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in fs:
                z.write(p, p.name)
        vol[asm] = (raw / 1e6, buf.tell() / len(fs) / 1e6)
    res["volume_MB_per_trial_raw_zip_current_wired"] = {k: (round(v[0], 2), round(v[1], 2)) for k, v in vol.items()}
    # 8 WS-P2 / PROTOCOL_V2 MASK: one (module, wheel) condition in every part (parent assemblies + the module alone)
    (W / "sealed" / "seed_v2.txt").write_text("3a5c" * 16 + "\n")
    r = run([T / "make_run_sheets.py", "extra", "masking", W / "sheets_mask", "--seed-file", W / "sealed" / "seed_v2.txt"], 0)
    print(r.stdout.strip())
    ms = json.loads((W / "sheets_mask" / "schedule.json").read_text())
    ep = next(e for e in ms["episodes"] if len(e["parts"]) == 4)          # a condition of a module in 3 assemblies
    mdirs = []
    for k, (part, nums) in enumerate(ep["parts"].items()):
        d, _ = SS.make_session(W / "mask_sessions", n_trials=len(nums), seed=40 + k, schedule=ms, trial_numbers=nums,
                               current_wired=True, codebook_dir=W / "_cb", device="cpu")
        mdirs.append((part, Path(d), nums))
    for sub in ("_codebook", "_runsheets", "_no_roster"):
        shutil.rmtree(W / "mask_sessions" / sub, ignore_errors=True)
    mman = W / "mask_manual"
    mman.mkdir()
    shutil.copy(PL.PACKET / "templates" / "episode_log.csv", mman / "episode_log.csv")
    with open(mman / "episode_log.csv", "a", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        for part, d, nums in mdirs:
            lone = part.startswith("single_")
            w.writerow([d.name, "MASK1", part, "split" if lone else "assembly", part.split("_")[1] if lone else "all",
                        "SYNTH", 1, nums[0], nums[-1], "none", "", "", part.split("_")[1] if lone else "assembly", 1,
                        "SYNTHETIC"])
    r = run([T / "validate_csv_session.py", *[d for _, d, _ in mdirs], "--schedule", W / "sheets_mask" / "schedule.json",
             "--manual", mman, "--expect", "simulation", "--out", W / "validate_mask.txt", "--csv", W / "validate_mask.csv"],
            (0, 1))
    assert not [ln for ln in r.stdout.splitlines() if ln.strip().startswith("FAIL")], r.stdout
    vm = (W / "validate_mask.csv").read_text()
    assert "PASS,E1,episodes,1 of 1 started condition(s) linked" in vm and "PASS,M4," in vm, vm[-2000:]
    assert sum(1 for ln in vm.splitlines() if ln.startswith("PASS,C15,")) == len(mdirs), vm
    from capxfer.field import loader as LD
    raws = [x for _, d, _ in mdirs for x in LD.load_session(d, allow_unverified=True, schedule=ms)]
    links = sorted({(x["meta"]["labels"]["episode"][0], x["meta"]["labels"]["phase"][0], x["meta"]["labels"]["round"][0],
                     x["meta"]["labels"]["fault_carrier"][0]) for x in raws})
    assert {l_[0] for l_ in links} == {ep["episode"]} and {l_[1] for l_ in links} == {"assembly", "split"}, links
    bad = W / "mask_manual_bad"
    shutil.copytree(mman, bad)
    part, d0, nums = mdirs[0]
    with open(bad / "episode_log.csv", "a", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerow([d0.name, "MASK1", part, "assembly", "all", "SYNTH", 1, nums[0], nums[-1],
                                                       "M1", "M1:fl", 1, "assembly", 1, "zip ties seen in MASK"])
    r = run([T / "validate_csv_session.py", d0, "--schedule", W / "sheets_mask" / "schedule.json", "--manual", bad,
             "--expect", "simulation", "--no-loader", "--no-verify"], 2)
    assert "M4" in r.stdout, r.stdout
    res["masking"] = (f"condition {ep['episode']} ({ep['design']}) in {list(ep['parts'])}: {len(mdirs)} SIMULATION sessions "
                      f"validate with no FAIL, E1 linked, M4 PASS; loader links {links}; a physical fault logged in "
                      f"episode_log for a MASK build -> FAIL M4")
    # 9 WS-P2 module choice on the laptop (synthetic bundle, stub models)
    from capxfer.field import choose_module as CM
    cres = CM.dry_run(W / "choice")
    res["choice"] = f"choose_module dry run: {len(cres)} DEMO episodes; the certificate sent the capable (coast-free) module; oracle from the lone-probe truth"
    (W / "dry_run_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"DRY RUN OK  (work dir {W})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

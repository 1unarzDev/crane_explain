"""Generate the sealed run sheets (CSV) from the study seed. The seed file stays OFF the kit; only its commitment is
published. Usage: python capxfer/field_kit_v2/make_run_sheets.py <seed_file> <out_dir>"""
import csv, hashlib, json, os, sys
from capxfer.analysis import realv2_design as D
from capxfer import assembly as A
from capxfer.probes import Schedule, WHEEL_SPEED_CAP, W_NOM

LAYOUT = {k: ";".join(f"{m}@{i},{j},{y}" for m, i, j, y in v) for k, v in A.PHYSICAL_TYPES.items()}
for m in D.MODULES:
    LAYOUT[f"single_{m}"] = f"{m}@0,0,0"

def main(seed_file, out):
    seed = bytes.fromhex(open(seed_file).read().strip())
    s = D.expand_schedule(seed)
    os.makedirs(out, exist_ok=True)
    P = A.load_params("params/module_measured.yaml")
    def w(name, rows, cols):
        with open(os.path.join(out, name), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader(); [wr.writerow({c: r.get(c, "") for c in cols}) for r in rows]
    rows = [dict(seq=t["seq"], build=t["build"], layout=LAYOUT[t["build"]],
                 fault="ok" if t["module"] is None else f"{t['module']}:{t['wheel']}:{t['firmware']}") for t in s["mask_trials"]]
    w("RUN_MASK.csv", rows, ["seq", "build", "layout", "fault"])
    rows = []
    for e in s["demo"]:
        rows.append(dict(episode=e["episode"], layout=LAYOUT["L3"],
                         faults=";".join(f"{m}:{e['faulted_wheels'][m]}:coast" for m in e["faulted_modules"]),
                         dock_order=" ".join(e["dock_order"]),
                         start_offset_m=" ".join(f"{m}={e['offset_m'][m]:+.2f}" for m in e["dock_order"])))
    w("RUN_DEMO.csv", rows, ["episode", "layout", "faults", "dock_order", "start_offset_m"])
    rows = [dict(seq=x["seq"], layout="M1@0,0,0", fault="ok" if x["wheel"] is None else f"M1:{x['wheel']}:{x['firmware']}")
            for x in s["simval_day2"]]
    w("RUN_SIMVAL_DAY2.csv", rows, ["seq", "layout", "fault"])
    w("RUN_APRILTAG_POSES.csv", [dict(order=i, pose=p) for i, p in enumerate(s["apriltag_pose_order"])], ["order", "pose"])
    # yaw-rate scaling per assembly (farthest wheel <= 0.45 m/s) for the probe table
    rows = []
    for k in ("single", "pair", "L3", "square4"):
        a = A.physical(k, P); wz = W_NOM * min(1.0, WHEEL_SPEED_CAP / max(a.char_length() * W_NOM, 1e-9))
        rows.append(dict(assembly=k, layout=LAYOUT[k], p5_yaw_rate_radps=f"{wz:.3f}", trial_seconds=f"{Schedule(a.char_length()).T:.1f}"))
    w("PROBE_YAW_BY_ASSEMBLY.csv", rows, ["assembly", "layout", "p5_yaw_rate_radps", "trial_seconds"])
    meta = dict(commitment=s["commitment"], schedule_sha256=D.schedule_sha256(s), version=s["version"])
    json.dump(meta, open(os.path.join(out, "SEAL.json"), "w"), indent=1)
    print(json.dumps(meta), len(s["mask_trials"]), "MASK trials,", len(s["demo"]), "DEMO episodes")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

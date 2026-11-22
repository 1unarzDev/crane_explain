"""Write rc-job/1 bodies for capxfer cluster tasks (stdlib only). Submit the file with `research submit -p modular`.

  python3 cluster/jobs.py --ref SHA --task smoke --seeds 0 --tags node-b --campaign capxfer-smoke --out /tmp/j.json
  python3 cluster/jobs.py --ref SHA --task i2_sweep_cell --args '{"n_asm":336,"expect_n":[20,24]}' \
      --seeds 8,9,14,... --per-job 8 --est 60 --campaign capxfer-i2-n2024 --purpose "..." --out /tmp/j.json

Each job is a `fastsim` job whose spec runs cluster/adapter.py (one task call per seed). --host/--tags pin it (--host node-a,
--tags node-b); without it any live worker with fastsim that serves the project may take it.
"""
import argparse
import hashlib
import json
import os

WT = os.path.expanduser("~/worktrees/capxfer-cluster")
INNOVATION_PY = ".venv/bin/python"   # used only where it exists


def seeds_of(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="capxfer-cluster-pkg commit (printed by cluster/rebundle.sh)")
    ap.add_argument("--task", required=True)
    ap.add_argument("--args", default="{}", help="task kwargs as JSON")
    ap.add_argument("--seeds", required=True, help="e.g. 0,1,5-9")
    ap.add_argument("--per-job", type=int, default=8)
    ap.add_argument("--est", type=float, default=120, help="estimated seconds per seed")
    ap.add_argument("--host", default=None, help="needs.host (works for node-a; node-b's worker has no host name: use --tags node-b)")
    ap.add_argument("--tags", default="", help="needs.tags, comma-separated (all required), e.g. node-b or windows")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--purpose", default="capxfer simulation study task on the research cluster: modular mecanum "
                                          "assembly simulation, CPU only, outputs uploaded as artifacts.")
    ap.add_argument("--priority", type=int, default=150)
    ap.add_argument("--submitter", default="capxfer")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    args = json.loads(a.args)
    seeds = seeds_of(a.seeds)
    cfg = {"task": a.task, "args": args, "threads": a.threads, "python": [INNOVATION_PY], "bootstrap": True}
    jobs = []
    for c in range(0, len(seeds), a.per_job):
        ss = seeds[c:c + a.per_job]
        tag = hashlib.sha1(json.dumps([a.ref, a.task, args, ss, a.host, a.tags]).encode()).hexdigest()[:8]
        jobs.append({
            "schema": "rc-job/1",
            "job_id": f"{a.campaign}-{ss[0]}-{ss[-1]}-{tag}",
            "kind": "fastsim",
            "purpose": a.purpose,
            "est_seconds": a.est,
            "submitter": a.submitter,
            "campaign": a.campaign,
            "priority": a.priority,
            "needs": {**({"host": a.host} if a.host else {}), **({"tags": a.tags.split(",")} if a.tags else {})},
            "repos": [{"name": "capxfer", "url": WT, "ref": a.ref}],
            "spec": {"repo": "capxfer", "workdir": "cluster", "config": "domain.yaml", "variant": a.task,
                     "overrides": {"adapter": {"options": {"cfg": cfg, "timeout_s": int(max(1800, 3 * a.est + 1800))}}}},
            "seeds": ss,
            "outputs": {"class": "summary"},
            "expect": {"owner": a.submitter, "metrics": ["cx.task_ok"],
                       "stop": "Stop when every listed seed has a valid task output; a failed seed is fixed locally, not retried in bulk."},
        })
    with open(a.out, "w") as fh:
        json.dump({"jobs": jobs}, fh, indent=1)
    print(f"{len(jobs)} job(s) -> {a.out}: " + " ".join(j["job_id"] for j in jobs))


if __name__ == "__main__":
    main()

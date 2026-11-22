"""Download a cluster job's artifacts (stdlib only).

  python3 cluster/fetch.py -p modular JOB_ID... [--campaign C] --out DIR [--match SUBSTR]

Lists GET /v1/artifacts?job=<project>/<job> and downloads each file to DIR/<job>/<seed>/<name>. Artifact reads need a
global viewer token: $RC_VIEWER_TOKEN, else ~/.config/research-cluster/platform-viewer.token (the project token can
list jobs, not read artifacts). Raw artifacts expire after 14 days: fetch soon after `research wait`.
"""
import argparse
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

URL = os.environ.get("RC_URL", "https://research.lunarz.dev")
CFG = Path.home() / ".config" / "research-cluster"


def token(name):
    return (os.environ.get("RC_VIEWER_TOKEN") if name == "platform-viewer" else None) or (CFG / f"{name}.token").read_text().strip()


def get(path, tok):
    req = urllib.request.Request(URL + path, headers={"Authorization": "Bearer " + tok, "User-Agent": "capxfer-fetch/1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--project", default="modular")
    ap.add_argument("jobs", nargs="*")
    ap.add_argument("--campaign")
    ap.add_argument("--out", required=True)
    ap.add_argument("--match", default="")
    a = ap.parse_args()
    ptok, vtok = token(a.project), token("platform-viewer")
    jobs = [j if "/" in j else f"{a.project}/{j}" for j in a.jobs]
    if a.campaign:
        q = urllib.parse.urlencode({"campaign": a.campaign, "limit": 1000})
        jobs += [j["job_id"] for j in json.loads(get(f"/v1/projects/{a.project}/jobs?{q}", ptok))["jobs"]]
    n = 0
    for job in dict.fromkeys(jobs):
        arts = json.loads(get("/v1/artifacts?" + urllib.parse.urlencode({"job": job}), vtok))["artifacts"]
        for x in arts:
            if a.match not in x["name"]:
                continue
            seed = re.search(r"\.s(\d+)$", x.get("run_id") or "")
            dest = Path(a.out) / job.split("/", 1)[1] / (f"s{seed.group(1)}" if seed else "job") / x["name"]
            if dest.exists() and dest.stat().st_size == x["size"]:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(get("/v1/artifact/" + urllib.parse.quote(x["key"], safe="/"), vtok))
            n += 1
    print(f"fetched {n} file(s) into {a.out}")


if __name__ == "__main__":
    main()

"""fastsim DomainAdapter for capxfer cluster jobs: ONE capxfer task per job seed (the fastsim core's loop is a stub).

The job spec sets `overrides: {adapter: {options: {cfg: {...}}}}`, deep-merged over domain.yaml:
  cfg.task      a name in cluster/tasks.py (smoke, i2_sweep_cell, ...)
  cfg.args      that task's keyword arguments (the job seed is passed as `seed`)
  cfg.python    optional list of interpreters to try first (the first with torch+numpy+yaml wins)
  cfg.bootstrap allow building a uv venv from cluster/requirements-cluster.txt when none has torch (default true)
  cfg.threads   torch/BLAS threads (default 4)
This module runs inside the worker's own Python (stdlib + fastsim only). Files the task writes land in the run
directory and the worker uploads each one as an artifact; the task's numbers become the run's metrics.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fastsim.controller import Controller, DomainAdapter

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "runner.py")
_RESULTS = {}  # seed -> result, from the controller to finalize (seeds run sequentially in one adapter)
NO_WINDOW = 0x08000000 | 0x00004000 if os.name == "nt" else 0  # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS


class _Once(Controller):
    def __init__(self, ctx, cfg, base, timeout):
        self.ctx, self.cfg, self.base, self.timeout = ctx, cfg, base, timeout

    def tick(self, obs):
        seed = self.ctx.seed
        out = tempfile.mkdtemp(prefix="capxfer-s%d-" % seed)
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, RUNNER], input=json.dumps({"cfg": self.cfg, "seed": seed, "out": out}),
                               capture_output=True, text=True, timeout=self.timeout, creationflags=NO_WINDOW)
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            res = json.loads(lines[-1]) if lines else {
                "error": "runner rc=%s: %s" % (p.returncode, (p.stderr or p.stdout)[-1500:])}
            Path(out, "runner.log").write_text((p.stdout or "")[-200000:] + "\n--- stderr ---\n" + (p.stderr or "")[-200000:])
        except Exception as e:  # noqa: BLE001  (timeout etc.)
            res = {"error": "%s: %s" % (type(e).__name__, e)}
        res.setdefault("metrics", {})
        res["wall_s"] = time.perf_counter() - t0
        res["out"] = out
        _RESULTS[seed] = res
        self.ctx.emit("task", task=self.cfg.get("task"), ok=not res.get("error"))
        return "FAILED" if res.get("error") else "SUCCEEDED"


class CapxferAdapter(DomainAdapter):
    def __init__(self, base=None, cfg=None, timeout_s=7200, strict=True, **options):
        super().__init__(base, cfg=cfg, timeout_s=timeout_s, strict=strict, **options)
        self.cfg, self.timeout, self.strict = dict(cfg or {}), float(timeout_s), bool(strict)

    def identity(self):
        return {"plugin": "capxfer-cluster/1", "cfg": self.cfg}

    def limits(self):
        from fastsim.plant import Limits
        return Limits.from_dict({"vmax": [1, 1, 1], "vmin": [-1, -1, -1], "accel": [1, 1, 1], "decel": [-1, -1, -1],
                                 "goal_tol": [0.5, 0.5], "precise_tol": [0.2, 0.1], "rotate_above": 0.8,
                                 "rotate_wz": 0.5, "progress_radius": 0.3, "progress_s": 10.0, "pass_radius": 1.0,
                                 "footprint": [[0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1]]})

    def controller(self, ctx):
        return _Once(ctx, self.cfg, self.base, self.timeout)

    def time_limit(self, ctx):
        return 5.0

    def finalize(self, run_dir, result):
        res = _RESULTS.pop(result["seed"], {"metrics": {}, "error": "no result", "out": None, "wall_s": 0.0})
        out = res.pop("out", None)
        if out and os.path.isdir(out):
            for f in sorted(os.listdir(out)):
                src = os.path.join(out, f)
                if os.path.isfile(src):
                    shutil.move(src, os.path.join(str(run_dir), f))
            shutil.rmtree(out, ignore_errors=True)
        rec = {"cfg": self.cfg, "seed": result["seed"], **res}
        Path(run_dir, "task.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")

    def metrics(self, run_dir):
        e = json.loads(Path(run_dir, "task.json").read_text())
        if e.get("error") and self.strict:  # a failed seed -> a failed job with the error in last_error
            raise RuntimeError("capxfer task failed: " + str(e["error"])[:1500])
        out = {k: float(v) for k, v in (e.get("metrics") or {}).items()
               if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")}
        out["task_ok"] = 0.0 if e.get("error") else 1.0
        out["task_wall_s"] = float(e["wall_s"])
        return {"cx": out}

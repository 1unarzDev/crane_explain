"""Stdlib-only entry, run by adapter.py in the worker's Python: pick (or build) a torch interpreter, run one task.

stdin  {"cfg": {...}, "seed": int, "out": dir}     stdout  last line: {"metrics": {...}, "env": {...}} or {"error": "..."}

Interpreter: the first of cfg.python, this interpreter, then a cached venv that imports torch, numpy and yaml.
With none and cfg.bootstrap (default true) it builds the cached venv from requirements-cluster.txt with uv
(CPU torch wheels), else with `python -m venv` + pip. The venv lives in the worker user's cache
(%LOCALAPPDATA%\\capxfer-cluster or ~/.cache/capxfer-cluster), keyed by the requirements hash, and is built
atomically (temp dir, then rename) under a lock dir, so a preempted build is redone, never half-used.
The task process: GPU hidden (CUDA_VISIBLE_DEVICES=""), <= cfg.threads threads, nice 15 / below-normal priority.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # the capxfer project root (capxfer/, params/, cluster/)
REQ = os.path.join(HERE, "requirements-cluster.txt")
WIN = os.name == "nt"
FLAGS = (0x08000000 | 0x00004000) if WIN else 0  # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PROBE = "import torch, numpy, yaml, sys; print(sys.version.split()[0], torch.__version__)"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, creationflags=FLAGS, **kw)


def has_torch(py):
    if not py or not os.path.exists(py):
        return False
    try:
        return run([py, "-c", PROBE], timeout=300).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def cache_dir():
    base = os.environ.get("LOCALAPPDATA") if WIN else os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.join(base or os.path.expanduser("~"), "capxfer-cluster")


def venv_python(d):
    return os.path.join(d, "Scripts", "python.exe") if WIN else os.path.join(d, "bin", "python")


def find_uv():
    for c in (shutil.which("uv"), os.path.expanduser("~/.local/bin/uv.exe" if WIN else "~/.local/bin/uv"),
              os.path.expanduser("~/.cargo/bin/uv")):
        if c and os.path.exists(c):
            return c
    return None


def build_venv(target):
    tmp = target + ".building-%d" % os.getpid()
    shutil.rmtree(tmp, ignore_errors=True)
    uv = find_uv()
    steps = []
    if uv:
        steps.append([uv, "venv", "--python", "3.12", tmp])
        py = venv_python(tmp)
        steps.append([uv, "pip", "install", "--python", py, "torch", "--index-url", TORCH_CPU_INDEX])
        steps.append([uv, "pip", "install", "--python", py, "-r", REQ])
    else:
        steps.append([sys.executable.replace("pythonw", "python"), "-m", "venv", tmp])
        py = venv_python(tmp)
        steps.append([py, "-m", "pip", "install", "-q", "torch", "--index-url", TORCH_CPU_INDEX])
        steps.append([py, "-m", "pip", "install", "-q", "-r", REQ])
    for s in steps:
        log("bootstrap:", " ".join(s))
        p = run(s, timeout=3600)
        if p.returncode != 0:
            raise RuntimeError("bootstrap step failed (%s): %s" % (s[1], (p.stderr or p.stdout)[-800:]))
    if not has_torch(py):
        raise RuntimeError("bootstrap venv does not import torch")
    os.replace(tmp, target)
    return "uv" if uv else "venv+pip"


def ensure_python(cfg, info):
    cands = list(cfg.get("python") or [])
    cands = [os.path.expanduser(os.path.expandvars(c)) for c in cands] + [sys.executable.replace("pythonw", "python")]
    for c in cands:
        if has_torch(c):
            info["python_source"] = "candidate"
            return c
    with open(REQ, "rb") as fh:
        key = hashlib.sha256(fh.read()).hexdigest()[:12]
    target = os.path.join(cache_dir(), "venv-" + key)
    py = venv_python(target)
    if has_torch(py):
        info["python_source"] = "cached_venv"
        return py
    if not cfg.get("bootstrap", True):
        raise RuntimeError("no interpreter with torch and bootstrap is off")
    os.makedirs(cache_dir(), exist_ok=True)
    lock = target + ".lock"
    t0 = time.time()
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            if time.time() - os.path.getmtime(lock) > 3 * 3600:  # stale (a preempted build)
                shutil.rmtree(lock, ignore_errors=True)
                continue
            if has_torch(py):
                info["python_source"] = "cached_venv"
                return py
            if time.time() - t0 > 3600:
                raise RuntimeError("timed out waiting for another job's venv build")
            time.sleep(20)
    try:
        if not has_torch(py):
            shutil.rmtree(target, ignore_errors=True)
            info["python_source"] = "bootstrap_" + build_venv(target)
            info["bootstrap_s"] = round(time.time() - t0, 1)
        else:
            info["python_source"] = "cached_venv"
    finally:
        shutil.rmtree(lock, ignore_errors=True)
    return py


def main():
    req = json.loads(sys.stdin.read())
    cfg, seed, out = req["cfg"], int(req["seed"]), req["out"]
    info = {}
    try:
        py = ensure_python(cfg, info)
        threads = str(int(cfg.get("threads", 4)))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                   OPENBLAS_NUM_THREADS=threads, CAPXFER_THREADS=threads, PYTHONPATH=ROOT, PYTHONDONTWRITEBYTECODE="1")
        pre = None
        if not WIN:
            def pre():
                try:
                    cur = os.getpriority(os.PRIO_PROCESS, 0)
                    if cur < 15:
                        os.nice(15 - cur)
                except OSError:
                    pass
        job = json.dumps({"task": cfg.get("task", "smoke"), "args": cfg.get("args") or {}, "seed": seed, "out": out})
        p = subprocess.run([py, "-m", "cluster.tasks"], input=job, capture_output=True, text=True, cwd=ROOT, env=env,
                           creationflags=FLAGS, preexec_fn=pre)
        sys.stderr.write(p.stderr[-100000:])
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        res = json.loads(lines[-1]) if lines else {"error": "task rc=%s: %s" % (p.returncode, p.stderr[-1200:])}
        res.setdefault("env", {}).update(info)
    except Exception as e:  # noqa: BLE001
        res = {"error": "%s: %s | %s" % (type(e).__name__, e, traceback.format_exc()[-800:]), "env": info}
    sys.stdout.write("\n" + json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

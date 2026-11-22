"""Field-side module choice for the DEMO (SPINE D11, D12, D15; CHOICE.md §4; PROTOCOL_V2 R4; card 10). Runs on the ROBOT
LAPTOP, by people, right after the episode's L3 probe and BEFORE the split; computes and logs every arm's module from the
just-logged CSV and prints the deduplicated DOCK LIST (every module chosen by >= 1 arm, in the sealed order).

  python -m capxfer.field.choose_module SESSION_DIR --episode K --sheets RUN_SHEET_DIR --bundle BUNDLE_DIR \
         --lambda lambda_v2.json [--log RUN_SHEET_DIR/choices/choices.csv] [--announce] [--device cpu]
  python -m capxfer.field.choose_module SESSION_DIR --episode K --sheets RUN_SHEET_DIR --log ... --finalize
         # step 2, after the lone probes: oracle from their truth + the deduplicated DOCK LIST
  python -m capxfer.field.choose_module --dry-run [--work DIR]        # synthetic end-to-end check (no robot, no ROS)
  python -m capxfer.field.choose_module bundle OUT_DIR --structured M.pt --blackbox M.pt --wheelsets W.npz \
         --library LIB.npz --bb-offset T [--tau 0.8] [--rung E1] [--pids P1,P5,P2]   # AI side, at the freeze

What it does (read-only on the session folder, which is still OPEN: no session.yaml / SHA256SUMS yet):
  1. schedule.json of RUN_SHEET_DIR (hash-checked) -> episode K's L3-probe trial, modules, sealed coins and orders.
  2. Finds the trial file holding that trial's markers and reads it with loader.load_trial_files (PROVISIONAL).
  3. featurize (bundle rung, P1+P5+P2 of the full-library run) -> the frozen bundle -> per module the score
     s = SPLIT method (a) COMP lower bound on the module's post-split strafe z: structured posterior -> I1 sparse wheel
     sets -> mode vectors consistent with them on the module -> min of the frozen single-module band (library rebuilt
     at the sys-ID box). Also the black-box current-body bound (carry-over; logged only).
  4. Pre-split arms (PROTOCOL_V2 R4): certificate = the largest s among s > lambda_hat (lambda_v2.json, alpha 0.10; tie
     -> sealed cert_tie_coin; none certified -> abstain); carry-over = go iff the black-box current-body bound >= tau,
     then the module nearest the goal (largest y, M4 in L3; tie -> sealed coin), else abstain; random = the sealed
     module. Logged (one row per arm + a diagnostic row) to --log, never inside the session folder.
  5. --finalize, after the lone probes: oracle = largest measured lone z (AprilTag truth of the lone probes) if >= tau
     (tie -> sealed coin), else abstain; then the DOCK LIST = every module chosen by >= 1 arm, deduplicated, in the
     sealed order, printed with start marks and the --set-faults trial numbers. Arm identities are never shown
     (PROTOCOL_V2 §8.3); --announce prints the certificate's module (for the video: only AFTER the episode's docks).

Bundle (capxfer.choice_bundle/v1, a folder hashed into the PROTOCOL_V2 freeze record): bundle.yaml {tau, alpha1, K, rung,
pids, structured, blackbox, wheelsets, bands, bb_offset, synthetic, files: {name: sha256}}, wheelsets.npz {s_ok, s_f},
bands.npz {mid, key, lb}. lambda_v2.json (after MASK): {"lambda_hat": float or null (= +inf, nothing certified),
"lambda_hat_alpha05": float or null, ...}; full schema capxfer.lambda_v2/v1 in capxfer.field.bundle (WS-I3F: a non-FINAL
file is refused on non-SIMULATION sessions, its bundle_sha256 must match, its carry_over_offset replaces bb_offset). `synthetic: true` bundles (stub models: 'stub:echo' reads the logged fault
echoes, 'stub:healthy') are for --dry-run only and are refused on any non-SIMULATION session.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import itertools
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

from . import schedule as SCH

BUNDLE_SCHEMA = "capxfer.choice_bundle/v1"
STRAFE = (2, 3)                                    # TARGET_KEYS P2+, P2-
N_MODES = 5


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ decision core
def cq(s, a):
    """Conformal quantile (as capxfer.ideas.i1.stage1.cq): ceil((n+1)(1-a))-th smallest score, inf if too few."""
    s = np.sort(np.asarray(s, float))
    k = math.ceil((s.size + 1) * (1 - a))
    return math.inf if k > s.size else float(s[k - 1])


def wheel_sets(P, ws):
    """Sparse label-conditional sets (I1): OK at alpha1/(2n), fault classes at alpha1/(2K). P (n,5) -> bool (n,5)."""
    n = P.shape[0]
    a, K = float(ws["alpha1"]), int(ws["K"])
    t = np.array([cq(ws["s_ok"], a / (2 * n))] + [cq(ws["s_f"], a / (2 * K))] * (N_MODES - 1))
    return (1 - P) <= t[None, :]


def mode_key(m):
    return int(sum(int(v) * N_MODES ** j for j, v in enumerate(m)))


def projected_mvs(S, w, K):
    """Mode vectors on wheels w consistent with the set S (<= K faults over the assembly, minus forced faults elsewhere)."""
    outside = np.setdiff1d(np.arange(S.shape[0]), w)
    kk = K - int((~S[outside, 0]).sum())
    if kk < 0:
        return []
    return [mode_key(mv) for mv in itertools.product(*[np.flatnonzero(S[j]) for j in w]) if sum(1 for v in mv if v) <= kk]


def certificate_bounds(P, wheel_module, mids, ws, bands):
    """Per module of the assembly: z_lb = min over consistent mode vectors of the frozen band (-inf if none / unknown)."""
    S = wheel_sets(P, ws)
    out = {}
    for k, mid in enumerate(mids):
        w = np.flatnonzero(np.asarray(wheel_module) == k)
        mvs = projected_mvs(S, w, int(ws["K"]))
        lbs = [bands.get((mid, u), -math.inf) for u in mvs]
        out[mid] = min(lbs) if lbs else -math.inf
    return out


def decide(lb, tau, fallback):
    """(module, abstain) from {module: z_lb}: argmax among z_lb >= tau; none certified -> (fallback(lb), True)."""
    ok = {m: v for m, v in lb.items() if v >= tau}
    if ok:
        return max(sorted(ok), key=lambda m: ok[m]), False
    return fallback, True


# ------------------------------------------------------------------------------------------------ bundle
def load_bundle(d):
    d = Path(d)
    b = yaml.safe_load((d / "bundle.yaml").read_text())
    if b.get("schema") != BUNDLE_SCHEMA:
        raise SystemExit(f"{d}: not a {BUNDLE_SCHEMA} bundle")
    for name, h in (b.get("files") or {}).items():
        if sha256_file(d / name) != h:
            raise SystemExit(f"{d}/{name}: sha256 differs from bundle.yaml (altered bundle; use the frozen one)")
    W = np.load(d / b["wheelsets"])
    ws = {"s_ok": W["s_ok"], "s_f": W["s_f"], "alpha1": b["alpha1"], "K": b["K"]}
    B = np.load(d / b["bands"])
    bands = {(str(m), int(k)): float(v) for m, k, v in zip(B["mid"], B["key"], B["lb"])}
    b["_dir"], b["_ws"], b["_bands"] = d, ws, bands
    b["_sha"] = sha256_file(d / "bundle.yaml")
    return b


def write_bundle(out, structured, blackbox, wheelsets, bands, bb_offset, tau=0.8, alpha1=0.05, K=2, rung="E1",
                 pids=("P1", "P5", "P2"), synthetic=False, note=""):
    """Assemble a bundle folder. structured/blackbox: model files (copied in) or 'stub:*'. wheelsets: npz {s_ok, s_f};
    bands: npz {mid, key, lb} or a SPLIT library.npz {modes, keys, Z_<mid>} (bands computed here, alpha2 = 0.05)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    ent = {}
    for name, src in (("structured", structured), ("blackbox", blackbox)):
        if str(src).startswith("stub:"):
            ent[name] = str(src)
        else:
            dst = out / f"{name}.pt"
            shutil.copyfile(src, dst)
            ent[name] = dst.name
            files[dst.name] = sha256_file(dst)
    W = np.load(wheelsets)
    np.savez(out / "wheelsets.npz", s_ok=W["s_ok"], s_f=W["s_f"])
    L = np.load(bands)
    if "lb" in L:
        np.savez(out / "bands.npz", mid=L["mid"], key=L["key"], lb=L["lb"])
    else:
        mid, key, lb = bands_from_library(L)
        np.savez(out / "bands.npz", mid=mid, key=key, lb=lb)
    for f in ("wheelsets.npz", "bands.npz"):
        files[f] = sha256_file(out / f)
    b = {"schema": BUNDLE_SCHEMA, "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
         "note": note, "tau": float(tau), "alpha1": float(alpha1), "K": int(K), "rung": rung, "pids": list(pids),
         "structured": ent["structured"], "blackbox": ent["blackbox"], "wheelsets": "wheelsets.npz",
         "bands": "bands.npz", "bb_offset": float(bb_offset), "synthetic": bool(synthetic), "files": files}
    (out / "bundle.yaml").write_text(yaml.safe_dump(b, sort_keys=False))
    return out


def bands_from_library(L, alpha2=0.05):
    """SPLIT single-module library (capxfer/ideas/split/sims.py library) -> one-sided band lower bound of z per
    (module, mode-vector key), as evaluate.Library: envelope over the WEAK-gain corners, (k1)-th smallest over plants."""
    keys = L["keys"]
    mids = sorted(k[2:] for k in L.files if k.startswith("Z_"))
    out_m, out_k, out_v = [], [], []
    for mid in mids:
        Z = L[f"Z_{mid}"].astype(float)
        M = Z.shape[1]
        k1 = math.floor((M + 1) * alpha2)
        for u in np.unique(keys):
            Zu = Z[np.flatnonzero(keys == u)]
            zlo = np.minimum(Zu[..., STRAFE[0]], Zu[..., STRAFE[1]]).min(0)
            out_m.append(mid)
            out_k.append(int(u))
            out_v.append(float(np.sort(zlo)[k1 - 1]) if k1 >= 1 else -math.inf)
    return np.array(out_m), np.array(out_k), np.array(out_v)


# ------------------------------------------------------------------------------------------------ models
class Models:
    def __init__(self, b, device="cpu"):
        self.b, self.device = b, device
        self._s = self._load(b["structured"])
        self._bb = self._load(b["blackbox"])

    def _load(self, ref):
        if str(ref).startswith("stub:"):
            if not self.b.get("synthetic"):
                raise SystemExit("stub models are allowed only in a synthetic (dry-run) bundle")
            return str(ref)
        from ..learn.base import Predictor
        return Predictor.load(self.b["_dir"] / ref, device=self.device)

    def predict(self, raw):
        """-> (P (B,n,5) per-wheel mode probabilities, zs (B,S) black-box samples of z on the current body)."""
        B, n = raw["duty"].shape[0], raw["duty"].shape[2]
        feat = None
        if not (isinstance(self._s, str) and isinstance(self._bb, str)):
            from ..featurize import featurize
            feat = featurize(raw, rung=self.b["rung"], pids=tuple(self.b["pids"]), with_targets=False)
        if self._s == "stub:echo":                      # dry run: the logged fault echoes (evaluator labels!)
            me = raw["meta"]["labels"]["mode_echo"].numpy()
            P = np.full((B, n, N_MODES), 0.02)
            for b_ in range(B):
                P[b_, np.arange(n), me[b_]] = 0.92
        else:
            P = self._s.predict(feat)["mode_prob"].detach().cpu().numpy().astype(float)
        if self._bb == "stub:healthy":
            zs = np.random.default_rng(0).normal(0.95, 0.02, (B, 64))
        else:
            ys = self._bb.predict(feat)["y_samples"].detach().cpu().numpy().astype(float)
            zs = np.minimum(ys[..., STRAFE[0]], ys[..., STRAFE[1]])
        return np.nan_to_num(P, nan=0.0), zs


# ------------------------------------------------------------------------------------------------ field run
def find_trial_files(session_dir, nums):
    """Trial files of the session holding a start..end marker pair of any of the schedule trial numbers."""
    from . import loader as L
    found = []
    for p in sorted(Path(session_dir).glob("gnd_capxprobe_*.csv")):
        segs = L._segments(L.read_trial_file(p))
        if any(int(s[0]) in nums for s in segs):
            found.append(p)
    return found


def load_lambda(path):
    d = json.loads(Path(path).read_text())
    f = lambda v: math.inf if v is None else float(v)                                   # noqa: E731
    return f(d.get("lambda_hat")), f(d.get("lambda_hat_alpha05", d.get("lambda_hat"))), sha256_file(path)


def load_lambda_record(path):
    """WS-I3F: lambda_v2.json through the one schema (capxfer.field.bundle.read_lambda; legacy files read as LEGACY).
    -> (lambda_hat, lambda_hat_alpha05, sha256, normalised record)."""
    from .bundle import LambdaFormatError, read_lambda
    try:
        r = read_lambda(path)
    except LambdaFormatError as e:
        raise SystemExit(f"bad lambda file: {e}") from None
    return r["lambda_hat"], r["lambda_hat_alpha05"], r["sha256"], r


def realv2_choice(s, lam, tie):
    """Certificate arm (PROTOCOL_V2 R4, CHOICE §1): among modules with s > lambda_hat, the largest s (exact tie -> the
    sealed coin); no module certified -> abstain (None). Returns (module or None, {module: certified})."""
    cert = {m: bool(s[m] > lam) for m in sorted(s)}
    ok = [m for m in sorted(s) if cert[m]]
    if not ok:
        return None, cert
    top = max(s[m] for m in ok)
    best = [m for m in ok if s[m] == top]
    return (best[0] if len(best) == 1 else (tie if tie in best else best[0])), cert


def dock_list(picks, order):
    """Deduplicated modules chosen by at least one arm (None = abstained), in the sealed dock order."""
    chosen = {m for m in picks.values() if m}
    return [m for m in order if m in chosen]


def _episode(sheets, episode):
    sched = SCH.load(Path(sheets) / "schedule.json")       # refuses an altered schedule
    ch = sched.get("choice")
    if not ch:
        raise SystemExit(f"{sheets}: not a DEMO schedule")
    ep = next((e for e in ch["episodes"] if int(e["choice_block"]) == int(episode)), None)
    if ep is None:
        raise SystemExit(f"no episode {episode} in the DEMO schedule")
    return sched, ch, ep


def _read(session_dirs, nums, sched):
    from . import loader as L
    files = [f for d in session_dirs for f in find_trial_files(d, set(nums))]
    if not files:
        raise SystemExit(f"no trial file in {[str(d) for d in session_dirs]} holds trial(s) {sorted(nums)} "
                         f"(was the FieldLog row ended with 'e'? did the probe finish?)")
    return files, L.load_trial_files(files, schedule=sched, trials=set(nums))


def _log_rows(log, session_dirs, rows):
    if not log:
        return
    log = Path(log)
    for d in session_dirs:
        if any(Path(d).resolve() == q for q in log.resolve().parents):
            raise SystemExit("refusing to write the choice log inside the session folder")
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.is_file()
    with open(log, "a", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if new:
            w.writerow(LOG_HEADER)
        w.writerows(rows)


LOG_HEADER = ["t_utc", "step", "episode", "trial", "arm", "module", "certified_or_go", "rule", "scores",
              "trial_files_sha256", "bundle_sha256", "lambda_sha256"]


def choose(session_dir, episode, sheets, bundle_dir, lambda_path, log=None, device="cpu", announce=False,
           allow_synthetic_on=("SIMULATION",)):
    """Step 1 (after the L3 probe, BEFORE the split): the pre-commitment arms (certificate, carry-over, random)."""
    sched, ch, ep = _episode(sheets, episode)
    b = load_bundle(bundle_dir)
    lam, lam05, lam_sha, lam_rec = load_lambda_record(lambda_path)          # WS-I3F: one schema (capxfer.field.bundle)
    num = int(ep["assembly_trial"])
    label = "UNKNOWN"
    sy = Path(session_dir) / "session.yaml"
    if sy.is_file():
        label = str(yaml.safe_load(sy.read_text()).get("label"))
    if b.get("synthetic") and label not in allow_synthetic_on:
        raise SystemExit(f"refusing a SYNTHETIC bundle on a {label} session: use the frozen bundle from the freeze record")
    if lam_rec["status"] != "FINAL" and label not in allow_synthetic_on:                       # WS-I3F
        raise SystemExit(f"refusing a {lam_rec['status']} lambda file on a {label} session: use the FINAL lambda_v2.json "
                         f"computed from the MASK CAL units (hash in the freeze record)")
    if lam_rec["bundle_sha256"] and lam_rec["bundle_sha256"] != b["_sha"]:                       # WS-I3F
        raise SystemExit("lambda_v2.json was calibrated on scores of another bundle (bundle_sha256 differs)")
    files, raws = _read([session_dir], [num], sched)
    if len(raws) != 1 or raws[0]["meta"]["labels"]["schedule_trial"] != [num]:
        raise SystemExit(f"could not read exactly the assembly probe {num} ({len(raws)} group(s))")
    raw = raws[0]
    asm = raw["meta"]["assembly"]
    mids = [m for m, *_ in asm.modules]
    P, zs = Models(b, device).predict(raw)
    s = certificate_bounds(P.mean(0), asm.wheel_module, mids, b["_ws"], b["_bands"])
    coins = ep["coins"]
    cert_m, cert = realv2_choice(s, lam, coins["cert_tie_coin"])
    cert05_m, cert05 = realv2_choice(s, lam05, coins["cert_tie_coin"])
    tau = float(ch.get("tau", b["tau"]))
    best = max(sorted(s), key=lambda m: s[m])
    bb_off = float(b["bb_offset"]) if lam_rec["carry_over_offset"] is None else lam_rec["carry_over_offset"]  # WS-I3F
    lb_cur = float(np.quantile(zs.reshape(-1), 0.05) - bb_off)
    near = list(ep["bb_nearest_goal"])
    bb_m = (near[0] if len(near) == 1 else coins["carry_over_tie_coin"]) if lb_cur >= tau else None
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    c = [stamp, "pre_split", ep["episode"], num]
    rows = [c + ["certificate", cert_m or "", int(cert_m is not None), "largest s among s > lambda_hat(0.10); none -> abstain"],
            c + ["certificate_alpha05", cert05_m or "", int(cert05_m is not None), "same at lambda_hat(0.05)"],
            c + ["sim_only_certificate", best if s[best] >= tau else "", int(s[best] >= tau),
                 f"largest COMP lb if >= tau={tau}, else abstain (no real recalibration)"],
            c + ["carry_over", bb_m or "", int(bb_m is not None),
                 f"go iff current-body lb >= tau, then nearest goal (largest y: {'/'.join(near)})"],
            c + ["random", coins["random_coin"], "", "sealed uniform module"]]
    rows = [r + ["", "", "", ""] for r in rows]
    rows.append(c + ["-", "", "", "", ";".join(f"s_{k}={v:.4f}" for k, v in s.items()) +
                     f";lambda_hat={lam};lambda_hat_alpha05={lam05};carry_over_lb={lb_cur:.4f};tau={tau}",
                     ";".join(f"{p.name}={sha256_file(p)}" for p in files), b["_sha"], lam_sha])
    _log_rows(log, [session_dir], rows)
    msg = (f"{ep['episode']}: pre-split choice logged. Now SPLIT and probe each module alone: "
           f"{', '.join(f'{m} (trial {ep['lone_trials'][m]})' for m in ep['dock_order'])}; then run choose_module "
           f"--finalize for the dock list.")
    if announce:
        msg += f"\n  certificate's module: {cert_m or 'ABSTAIN (none certified)'}"
    print(msg)
    return {"scores": s, "certified": cert, "certificate": cert_m, "carry_over": bb_m, "random": coins["random_coin"],
            "carry_over_lb": lb_cur, "rows": rows}


def lone_strafe(raws):
    """Measured strafe z = min(rho_P2+, rho_P2-) of each lone probe (evaluator-side truth: AprilTag)."""
    from ..featurize import targets
    out = {}
    for raw in raws:
        y, _ = targets(raw, raw["meta"]["assembly"], float(raw["meta"]["truth_latency_s"]))
        z = np.minimum(y[:, STRAFE[0]].numpy(), y[:, STRAFE[1]].numpy())
        out[raw["meta"]["assembly"].modules[0][0]] = float(np.nanmean(z)) if np.isfinite(z).any() else float("nan")
    return out


def finalize(session_dirs, episode, sheets, log):
    """Step 2 (after the lone probes, before any dock): the oracle (largest measured lone z if >= tau, ties -> sealed
    coin, else abstain) from the lone probes' truth, then the deduplicated DOCK LIST from the logged pre-split arms."""
    sched, ch, ep = _episode(sheets, episode)
    if not log or not Path(log).is_file():
        raise SystemExit("--finalize needs the --log written by the pre-split step")
    pre = [r for r in csv.DictReader(open(log)) if r["episode"] == ep["episode"] and r["step"] == "pre_split"]
    picks = {r["arm"]: (r["module"] or None) for r in pre if r["arm"] in ("certificate", "carry_over", "random")}
    if set(picks) != {"certificate", "carry_over", "random"}:
        raise SystemExit(f"no complete pre-split choice for {ep['episode']} in {log}: run the pre-split step first")
    nums = list(ep["lone_trials"].values())
    files, raws = _read([Path(d) for d in session_dirs], nums, sched)
    z = lone_strafe(raws)
    if set(z) != set(ep["modules"]):
        raise SystemExit(f"lone probes found for {sorted(z)}, need {ep['modules']}")
    tau = float(ch["tau"])
    fin = {m: v for m, v in z.items() if np.isfinite(v)}
    top = max(fin.values()) if fin else -math.inf
    best = sorted(m for m in fin if fin[m] == top)
    oracle = (best[0] if len(best) == 1 else ep["coins"]["oracle_tie_coin"]) if top >= tau else None
    picks["oracle"] = oracle
    docks = dock_list(picks, ep["dock_order"])
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    rows = [[stamp, "finalize", ep["episode"], " ".join(map(str, nums)), "oracle", oracle or "", int(oracle is not None),
             f"largest measured lone z if >= tau={tau}; tie -> sealed coin; else abstain",
             ";".join(f"z_{k}={v:.4f}" for k, v in z.items()), ";".join(f"{p.name}={sha256_file(p)}" for p in files), "", ""],
            [stamp, "finalize", ep["episode"], "", "dock_list", "+".join(docks), "", "deduplicated, sealed order", "", "", "",
             ""]]
    _log_rows(log, [Path(d) for d in session_dirs], rows)
    marks = ep["start_pose_id"]
    print(f"{ep['episode']}: DOCK LIST ({len(docks)}): " +
          (", ".join(f"#{j} {m} from mark {marks[m]}" for j, m in enumerate(docks, 1)) or "none (every arm abstained)") +
          "\n  before each dock: probe_runner SCHEDULE --set-faults <trial>, after it --clear-faults <trial>: " +
          ", ".join(f"{m} -> {ep['lone_trials'][m]}" for m in docks))
    return {"docks": docks, "z_alone": z, "oracle": oracle, "picks": picks}


# ------------------------------------------------------------------------------------------------ dry run
def synthetic_bundle(out):
    """Synthetic (dry-run only) bundle: stub models, wheel-set scores that give tight sets on a confident posterior, and a
    toy band table: healthy 0.95, one passive / weak fault 0.85, one roller lock 0.60, two faults with a lock 0.30, else 0.5."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.savez(out / "_ws.npz", s_ok=rng.uniform(0.0, 0.1, 2000), s_f=rng.uniform(0.0, 0.1, 2000))
    mid, key, lb = [], [], []
    for m in ("M1", "M2", "M3", "M4"):
        for mv in itertools.product(range(N_MODES), repeat=4):
            nf = sum(1 for v in mv if v)
            if nf > 2:
                continue
            v = 0.95 if nf == 0 else (0.60 if (nf == 1 and 4 in mv) else (0.85 if nf == 1 else (0.30 if 4 in mv else 0.5)))
            mid.append(m)
            key.append(mode_key(mv))
            lb.append(v)
    np.savez(out / "_bands.npz", mid=np.array(mid), key=np.array(key), lb=np.array(lb))
    return write_bundle(out / "bundle", "stub:echo", "stub:healthy", out / "_ws.npz", out / "_bands.npz", bb_offset=0.02,
                        tau=0.8, synthetic=True, note="SYNTHETIC dry-run bundle: never for field use")


def dry_run(work):
    """DEMO schedule (throw-away 256-bit seed) -> SIMULATION FieldLog sessions of three episodes' L3 probes (FieldLog's
    own writer) -> choose(). The certificate must send the capable module (the one without a coast wheel)."""
    import subprocess
    from . import synth_session as SS
    work = Path(work)
    shutil.rmtree(work, ignore_errors=True)
    (work / "sealed").mkdir(parents=True)
    (work / "sealed" / "seed_v2.txt").write_text("c4" * 32 + "\n")
    repo = Path(__file__).resolve().parents[2]
    tool = repo / "capxfer" / "field_packet" / "tools" / "make_run_sheets.py"
    r = subprocess.run([sys.executable, str(tool), "extra", "choice", str(work / "sheets"), "--seed-file",
                        str(work / "sealed" / "seed_v2.txt")],
                       capture_output=True, text=True, cwd=str(repo), env={"PYTHONPATH": str(repo), "PATH": "/usr/bin:/bin"})
    if r.returncode != 0:
        raise SystemExit(r.stderr[-2000:])
    sched = SCH.load(work / "sheets" / "schedule.json")
    bundle = synthetic_bundle(work / "bundle")
    lam = work / "lambda_v2_SYNTHETIC.json"
    lam.write_text(json.dumps({"lambda_hat": 0.5, "lambda_hat_alpha05": 0.7, "note": "SYNTHETIC dry run"}))
    res = []
    for ep in sched["choice"]["episodes"][:3]:
        d, _ = SS.make_session(work / "sessions", n_trials=1, seed=7 + ep["choice_block"], schedule=sched,
                               trial_numbers=[ep["assembly_trial"]], codebook_dir=work / "_cb", device="cpu")
        log = work / "choices" / "choices.csv"
        out = choose(d, ep["choice_block"], work / "sheets", bundle, lam, log=log, announce=True)
        assert out["certificate"] == ep["capable_module"], (ep, out)
        lone = []
        for i, m in enumerate(ep["dock_order"]):                     # one synthetic session per lone layout
            dl, _ = SS.make_session(work / "sessions", n_trials=1, seed=70 + 3 * ep["choice_block"] + i, schedule=sched,
                                    trial_numbers=[ep["lone_trials"][m]], codebook_dir=work / "_cb", device="cpu")
            lone.append(dl)
        fin = finalize(lone, ep["choice_block"], work / "sheets", log)
        assert out["certificate"] in fin["docks"] and len(fin["docks"]) == len(set(fin["docks"])) <= 3
        res.append({"episode": ep["episode"], "faulted": ep["faulted_modules"], "certificate": out["certificate"],
                    "z_alone": fin["z_alone"], "oracle": fin["oracle"], "docks": fin["docks"], "scores": out["scores"]})
    (work / "dry_run_choice.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"CHOOSE_MODULE DRY RUN OK ({len(res)} episodes; work dir {work})")
    return res


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "bundle":
        ap = argparse.ArgumentParser(prog="choose_module bundle")
        ap.add_argument("out")
        for k in ("structured", "blackbox", "wheelsets", "library"):
            ap.add_argument(f"--{k}", required=True)
        ap.add_argument("--bb-offset", type=float, required=True)
        ap.add_argument("--tau", type=float, default=0.8)
        ap.add_argument("--alpha1", type=float, default=0.05)
        ap.add_argument("--K", type=int, default=2)
        ap.add_argument("--rung", default="E1")
        ap.add_argument("--pids", default="P1,P5,P2")
        ap.add_argument("--note", default="")
        a = ap.parse_args(argv[1:])
        d = write_bundle(a.out, a.structured, a.blackbox, a.wheelsets, a.library, a.bb_offset, a.tau, a.alpha1, a.K,
                         a.rung, a.pids.split(","), note=a.note)
        print(f"bundle {d}: bundle.yaml sha256 {sha256_file(d / 'bundle.yaml')} (hash it into PROTOCOL_V2)")
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="*")
    ap.add_argument("--episode", type=int)
    ap.add_argument("--sheets")
    ap.add_argument("--bundle")
    ap.add_argument("--lambda", dest="lam")
    ap.add_argument("--announce", action="store_true")
    ap.add_argument("--finalize", action="store_true", help="step 2, after the lone probes: oracle + the DOCK LIST")
    ap.add_argument("--log")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--work", default=str(Path.home() / ".cache" / "capxfer" / "P2" / "choose_dryrun"))
    a = ap.parse_args(argv)
    if a.dry_run:
        dry_run(a.work)
        return 0
    if a.finalize:
        if not (a.session and a.episode and a.sheets and a.log):
            ap.error("--finalize needs SESSION_DIR(s), --episode, --sheets and --log")
        finalize(a.session, a.episode, a.sheets, a.log)
        return 0
    if not (len(a.session) == 1 and a.episode and a.sheets and a.bundle and a.lam):
        ap.error("SESSION_DIR, --episode, --sheets, --bundle and --lambda are required (or --dry-run / --finalize)")
    choose(a.session[0], a.episode, a.sheets, a.bundle, a.lam, a.log, a.device, a.announce)
    return 0


if __name__ == "__main__":
    sys.exit(main())

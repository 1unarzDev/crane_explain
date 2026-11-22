"""CHOICE: which module do we send? Module-choice (task-allocation) experiment, D9 (SIMULATION ONLY).

A faulted assembly (2-6 modules; SPLIT test episodes: calB, heldout, physical pair/L3/square4; in-model single/double
faults incl. passive) must send ONE module on a sideways dock after an all-singles split. Before the split each policy
picks a module or abstains:
  cert      SPLIT (a) compositional certificate (structured_ensemble -> I1 sparse sets -> projected -> library
            bands): argmax certified lower bound on z = min(rho_P2+, rho_P2-); abstain if max lb < tau.
  bb        black-box carry-over: the current-body lower bound is given to every module; go iff it is >= tau, then a
            fixed rule (module nearest the goal = largest y in the assembly frame, the bay opens in +y; ties random).
  healthy   assume healthy: never abstains, nearest-goal rule.
  random    a uniformly random module, never abstains.
  probe_all split first, probe every module alone with P1+P5+P2 (12.7 s each), W-ensemble one-sided CQR lower bound
            (SPLIT arm f), argmax certified; none certified -> reassemble.
  oracle    true z: best module if z >= tau, else no split.
Outcome of a dispatch = dock success of that module (closed-loop sideways dock into a narrow bay, dock_runs.py),
modelled as p_dock(z) (isotonic fit to the dock simulations); validated against direct dock sims on a subsample.

Mission time model (with fallback; constants below are ASSUMPTIONS except T_PROBE):
  pre-split policy: dispatch -> T_SPLIT + dock attempt; on failure (or on abstention for cert / bb) escalate to
  probe-all over the modules not yet tried (T_PROBE each) and send the best probe-certified one; none -> T_REASM.
  probe_all: T_SPLIT + T_PROBE * n_mod, best certified, on failure the next certified; none -> T_REASM.
  oracle: best capable, then next capable; none capable -> no split (0 s).
  At most 2 dock attempts per mission. Failed attempt = T_DOCK_FAIL + T_RETURN.

  PYTHONPATH=. nice -n 15 .venv/bin/python -m capxfer.ideas.choice.evaluate [alpha05|run]
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

from capxfer.ideas.split import common as C, evaluate as EV

OUT = pathlib.Path.home() / ".cache/capxfer/CHOICE"
TAUS = (0.5, 0.8, 0.9)   # 0.5 / 0.8 = SPLIT; 0.9 = matched to the narrow-bay dock (p_dock(z) ~ 0.5 at z 0.8-0.9)
ALPHAS = (0.10, 0.05)
T_SPLIT = 10.0          # ASSUMPTION: autonomous unlatch + separation
T_PROBE = 12.7          # measured schedule length of P1+P5+P2 on a single module (SPLIT)
T_REASM = 240.0         # ASSUMPTION: manual reassembly 3-5 min (PROTOCOL_DRAFT TEAM-CONFIRM), midpoint 4 min
T_RETURN = 30.0         # ASSUMPTION: recover a failed module out of the bay and clear the start area
PARALLEL = False        # sensitivity: all modules probe at once (wall-clock T_PROBE)
POLICIES = ("cert", "bb", "healthy", "random", "probe_all", "oracle")
REPS = 10


# ----------------------------------------------------------------------------------------------- inputs
def alpha_rows(alpha):
    """per-module lower bounds (cert, bb, healthy) at total level alpha; 0.10 = SPLIT's cached rows."""
    if abs(alpha - 0.10) < 1e-9:
        S = np.load(C.OUT / "rows_singles_P1P5P2.npz")
        return {k: S[f"lb_{m}"].astype(float) for k, m in (("cert", "comp_se"), ("bb", "bb_carry"),
                                                           ("post", "post_se"))}
    path = OUT / f"rows_alpha{alpha:.3f}.npz"
    if path.exists():
        S = np.load(path)
        return {k: S[k] for k in S.files}
    EV.ALPHA1 = EV.ALPHA2 = alpha / 2
    EV.ALPHA = alpha
    _, _, R = EV.build("P1P5P2")
    out = dict(cert=R["lb"]["comp_se"], bb=np.asarray(R["lb"]["bb_carry"], float), post=R["lb"]["post_se"])
    np.savez_compressed(path, **out)
    return out


def pava(x, y, w=None):
    """isotonic (non-decreasing) regression; returns knots (x sorted, fitted y)."""
    o = np.argsort(x)
    x, y = np.asarray(x, float)[o], np.asarray(y, float)[o]
    w = np.ones_like(y) if w is None else np.asarray(w, float)[o]
    v, ww, n = list(y), list(w), [1] * len(y)
    i = 0
    vals, wts, cnt = [], [], []
    for yi, wi in zip(v, ww):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            a = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]; cnt[-2] += cnt[-1]; vals[-2] = a
            vals.pop(); wts.pop(); cnt.pop()
    fit = np.repeat(vals, cnt)
    return x, fit


class DockModel:
    """p_dock(z): isotonic fit of closed-loop narrow-bay sideways-dock success on the module's true z."""

    def __init__(self, rows):
        z = np.array([r["z"] for r in rows])
        s = np.array([float(r["success"]) for r in rows])
        self.x, self.f = pava(z, s)
        self.n = len(rows)
        ts = np.array([r["time"] for r in rows])
        self.t_ok = float(ts[s == 1].mean()) if (s == 1).any() else 15.0
        self.t_fail = float(ts[s == 0].mean()) if (s == 0).any() else 20.0

    def p(self, z):
        z = np.asarray(z, float)
        return np.interp(z, self.x, self.f, left=self.f[0], right=self.f[-1])


def load_dock():
    rows = [json.loads(l) for l in (OUT / "dock_rows.jsonl").open()]
    return rows


# ----------------------------------------------------------------------------------------------- episodes
def fault_type(e):
    m = e["mode"]
    f = np.flatnonzero(m > 0)
    if len(f) == 1:
        return {1: "single weak", 2: "single passive", 3: "single passive", 4: "single roller"}[int(m[f[0]])]
    mods = {int(j) // 4 for j in f}
    return "double, same module" if len(mods) == 1 else "double, two modules"


class Episodes:
    def __init__(self):
        self.E = C.episodes("P1P5P2")
        Gs = C.split_groups(self.E)
        T = np.load(C.OUT / "truth_singles.npz")
        self.ep, self.mod = T["ep"], T["mod"]
        self.z = EV.zmin(T["rho"].astype(float))
        test = np.setdiff1d(np.arange(len(self.E)), Gs["calA"])
        self.faulted = np.array([k for k in test if (self.E[k]["mode"] > 0).any()])
        self.rows_of = {}
        for r, k in enumerate(self.ep):
            self.rows_of.setdefault(int(k), []).append(r)
        self.ypos = np.array([self.E[k]["modules"][m][2] for k, m in zip(self.ep, self.mod)], float)
        self.calrows_pa = None

    def probe_all_lb(self, alpha):
        A = np.load(C.OUT / "probe_after.npz")
        cal = A["rows"][:int(A["ncal"])]
        t = EV.cq(A["ensemble|P1P5P2|q05z"][:int(A["ncal"])] - self.z[cal], alpha)
        P = np.load(OUT / "probe_all.npz")
        lb = np.full(len(self.z), np.nan)
        lb[P["rows"]] = P["q05z"] - t
        return lb, np.sort(P["eps"])


# ----------------------------------------------------------------------------------------------- policies
def pick(policy, rows, lbs, z, ypos, tau, rng):
    """-> (module row or None, ordered fallback list used by probe_all/oracle)."""
    if policy == "oracle":
        o = rows[np.argsort(-z[rows])]
        cap = [r for r in o if z[r] >= tau]
        return (cap[0] if cap else None), cap
    if policy == "random":
        return int(rng.choice(rows)), []
    if policy in ("bb", "healthy"):
        if policy == "bb" and lbs["bb"][rows[0]] < tau:
            return None, []
        y = ypos[rows] + rng.uniform(0, 1e-6, len(rows))
        return int(rows[np.argmax(y)]), []
    key = {"cert": "cert", "probe_all": "pa", "post": "post"}[policy]
    lb = lbs[key][rows] + rng.uniform(0, 1e-9, len(rows))
    o = np.argsort(-lb)
    cert = [int(rows[i]) for i in o if lb[i] >= tau]
    return (cert[0] if cert else None), cert


def probe_time(n):
    return T_PROBE if (PARALLEL and n > 0) else T_PROBE * n


def mission(policy, rows, lbs, z, ypos, tau, rng, succ, dm):
    """one mission with fallback; succ: dict row -> bool (dock outcome draw). Returns (success, time, n_attempts)."""
    first, order = pick(policy, rows, lbs, z, ypos, tau, rng)
    t_ok, t_fail = dm.t_ok, dm.t_fail + T_RETURN
    if policy == "oracle":
        t = 0.0
        if first is None:
            return False, 0.0, 0
        t += T_SPLIT
        for k, r in enumerate(order[:2]):
            if succ[r]:
                return True, t + t_ok, k + 1
            t += t_fail
        return False, t + T_REASM, min(2, len(order))
    if policy == "probe_all":
        t = T_SPLIT + probe_time(len(rows))
        if first is None:
            return False, t + T_REASM, 0
        for k, r in enumerate(order[:2]):
            if succ[r]:
                return True, t + t_ok, k + 1
            t += t_fail
        return False, t + T_REASM, min(2, len(order))
    # pre-split policies
    t, tried = T_SPLIT, []
    if first is not None:
        tried.append(first)
        if succ[first]:
            return True, t + t_ok, 1
        t += t_fail
    # escalate: probe the remaining modules, send the best probe-certified one
    rest = np.array([r for r in rows if r not in tried])
    t += probe_time(len(rest))
    lb = lbs["pa"][rest]
    ok = rest[lb >= tau]
    if len(ok) == 0:
        return False, t + T_REASM, len(tried)
    r = int(ok[np.argmax(lbs["pa"][ok])])
    if succ[r]:
        return True, t + t_ok, len(tried) + 1
    return False, t + t_fail + T_REASM, len(tried) + 1


def single_shot(policy, eps, EP, lbs, tau, dm, rng):
    rec = []
    for k in eps:
        rows = np.array(EP.rows_of[int(k)])
        r, _ = pick(policy, rows, lbs, EP.z, EP.ypos, tau, rng)
        if r is None:
            rec.append((0.0, 0, 1, int((EP.z[rows] >= tau).any()), 0))
        else:
            rec.append((float(dm.p(EP.z[r])), int(EP.z[r] < tau), 0, 0, 1))
    return np.array(rec)          # cols: p_success, unsafe, abstain, missed (abstain while a capable module exists), dispatched


def run_missions(policy, eps, EP, lbs, tau, dm, seed=0):
    out = []
    pz = dm.p(EP.z)
    U = np.random.default_rng(seed).uniform(size=(REPS, len(EP.z)))              # common random numbers
    for rep in range(REPS):
        rng = np.random.default_rng([seed, rep])
        for k in eps:
            rows = np.array(EP.rows_of[int(k)])
            succ = {int(r): bool(U[rep, r] < pz[r]) for r in rows}
            s, t, n = mission(policy, rows, lbs, EP.z, EP.ypos, tau, rng, succ, dm)
            out.append((s, t, n))
    return np.array(out, float)


def boot_ci(vals, groups, B=300, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    m = []
    for _ in range(B):
        gs = rng.choice(ug, len(ug))
        ii = np.concatenate([idx[g] for g in gs])
        m.append(vals[ii].mean())
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def summarize_ss(R, groups=None):
    d = dict(n=int(len(R)), success=float(R[:, 0].mean()), unsafe=float(R[:, 1].mean()),
             abstain=float(R[:, 2].mean()), missed=float(R[:, 3].mean()),
             unsafe_given_dispatch=float(R[:, 1].sum() / max(R[:, 4].sum(), 1)))
    if groups is not None:
        d["success_ci"] = boot_ci(R[:, 0], groups)
        d["unsafe_ci"] = boot_ci(R[:, 1].astype(float), groups, seed=1)
    return d


def summarize_mission(M):
    s, t = M[:, 0], M[:, 1]
    return dict(final_success=float(s.mean()), mean_time=float(t.mean()),
                time_per_success=float(t.sum() / max(s.sum(), 1)),
                time_given_success=float(t[s == 1].mean()) if (s == 1).any() else float("nan"),
                attempts=float(M[:, 2].mean()))


def main():
    global T_REASM, PARALLEL
    EP = Episodes()
    dock = load_dock()
    calib = [r for r in dock if r["set"] in ("ep", "zbin")]
    dm = DockModel(calib)
    out = dict(constants=dict(T_SPLIT=T_SPLIT, T_PROBE=T_PROBE, T_REASM=T_REASM, T_RETURN=T_RETURN,
                              T_DOCK_OK=dm.t_ok, T_DOCK_FAIL=dm.t_fail, REPS=REPS),
               dock_model=dict(n=dm.n, knots_z=[float(v) for v in np.quantile(dm.x, np.linspace(0, 1, 21))],
                               p_at=dict((f"{v:.2f}", float(dm.p(v))) for v in np.arange(0.0, 1.05, 0.05))),
               tables={})
    # proxy validation: success by z bin and agreement with z >= tau
    zz = np.array([r["z"] for r in calib]); ss = np.array([r["success"] for r in calib], float)
    bins = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 2]
    out["dock_by_z"] = [dict(lo=lo, hi=hi, n=int(((zz >= lo) & (zz < hi)).sum()),
                             success=float(ss[(zz >= lo) & (zz < hi)].mean()) if ((zz >= lo) & (zz < hi)).any() else None)
                        for lo, hi in zip(bins[:-1], bins[1:])]
    out["proxy_agreement"] = {str(t): float(((zz >= t) == (ss == 1)).mean()) for t in (0.5, 0.6, 0.7, 0.8)}
    demo = [r for r in dock if r["set"] == "demo"]
    out["dock_demo_kinds"] = {k: dict(n=len([r for r in demo if r["kind"] == k]),
                                      success=float(np.mean([r["success"] for r in demo if r["kind"] == k])),
                                      z_median=float(np.median([r["z"] for r in demo if r["kind"] == k])))
                              for k in sorted({r["kind"] for r in demo})}
    for alpha in ALPHAS:
        lbs = alpha_rows(alpha)
        lbs["pa"], sub = EP.probe_all_lb(alpha)
        groups_sub = np.array([EP.E[k]["gi"] for k in sub])
        for tau in TAUS:
            for pol in POLICIES:
                rng = np.random.default_rng(0)
                R = single_shot(pol, sub, EP, lbs, tau, dm, rng)
                key = f"{alpha}|{tau}|{pol}"
                out["tables"][key] = dict(single_shot=summarize_ss(R, groups_sub))
                M = run_missions(pol, sub, EP, lbs, tau, dm)
                out["tables"][key]["mission"] = summarize_mission(M)
                # breakdowns (single shot + mission)
                for nm in range(2, 7):
                    sel = np.array([EP.E[k]["nmod"] == nm for k in sub])
                    Msel = M.reshape(REPS, len(sub), 3)[:, sel].reshape(-1, 3)
                    out["tables"][key][f"nmod{nm}"] = dict(single_shot=summarize_ss(R[sel]),
                                                            mission=summarize_mission(Msel))
                fts = np.array([fault_type(EP.E[k]) for k in sub])
                for ft in np.unique(fts):
                    sel = fts == ft
                    Msel = M.reshape(REPS, len(sub), 3)[:, sel].reshape(-1, 3)
                    out["tables"][key][f"ft|{ft}"] = dict(single_shot=summarize_ss(R[sel]),
                                                           mission=summarize_mission(Msel))
                if pol != "probe_all":         # full faulted test set (natural size mix), single shot only
                    Rf = single_shot(pol, EP.faulted, EP, lbs, tau, dm, np.random.default_rng(0))
                    out["tables"][key]["all_faulted"] = summarize_ss(Rf)
                print(key, json.dumps(out["tables"][key]["single_shot"]), json.dumps(out["tables"][key]["mission"]),
                      flush=True)
    # direct-dock validation on the episodes whose every module was dock-simulated
    direct = {r["row"]: r["success"] for r in dock if r["set"] == "ep"}
    full = [k for k in np.unique([EP.ep[r] for r in direct]) if all(r in direct for r in EP.rows_of[int(k)])]
    lbs = alpha_rows(0.10)
    lbs["pa"], sub = EP.probe_all_lb(0.10)
    val = {}
    for tau in TAUS:
        for pol in POLICIES:
            eps = [k for k in full if pol != "probe_all" or k in set(sub.tolist())]
            sim_s, prox = [], []
            rng = np.random.default_rng(0)
            for k in eps:
                rows = np.array(EP.rows_of[int(k)])
                if pol == "probe_all" and np.isnan(lbs["pa"][rows]).any():
                    continue
                r, _ = pick(pol, rows, lbs, EP.z, EP.ypos, tau, rng)
                sim_s.append(0.0 if r is None else float(direct[int(r)]))
                prox.append(0.0 if r is None else float(dm.p(EP.z[r])))
            val[f"{tau}|{pol}"] = dict(n=len(sim_s), direct_sim=float(np.mean(sim_s)) if sim_s else None,
                                       proxy=float(np.mean(prox)) if prox else None)
    out["direct_validation"] = val
    # sensitivity of time-per-success to T_REASM / parallel probing (alpha 0.10, tau 0.8 and 0.5)
    sens = {}
    lbs = alpha_rows(0.10)
    lbs["pa"], sub = EP.probe_all_lb(0.10)
    for name, (tr, tp) in {"reasm60": (60.0, 12.7), "reasm600": (600.0, 12.7), "parallel_probe": (240.0, None)}.items():
        T_REASM = tr
        PARALLEL = tp is None
        for tau in TAUS:
            for pol in POLICIES:
                M = run_missions(pol, sub, EP, lbs, tau, dm)
                sens[f"{name}|{tau}|{pol}"] = summarize_mission(M)
        PARALLEL = False
    T_REASM = 240.0
    out["time_sensitivity"] = sens
    (OUT / "results.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "results.json")




# ----------------------------------------------------------------------------------------------- D11 demo arms
DEMO_ARMS = ("cert", "bb", "random", "oracle", "healthy")


def demo_main(n_runs=(10, 20, 30)):
    """Predicted per-arm outcomes for the physical demo setups (choice.demo trials) and the power of a
    one-sided Fisher test certificate > black-box carry-over at n runs per arm."""
    from scipy.stats import fisher_exact
    D = np.load(OUT / "demo_preds.npz")
    dock = load_dock()
    dm = DockModel([r for r in dock if r["set"] in ("ep", "zbin")])
    demo = [r for r in dock if r["set"] == "demo"]
    kind_p = {k: float(np.mean([r["success"] for r in demo if r["kind"] == k])) for k in {r["kind"] for r in demo}}
    res = {}
    for name in D["setups"]:
        z, fmod, ypos = D[f"{name}|z"], D[f"{name}|fmod"], D[f"{name}|ypos"]
        mode = D[f"{name}|mode"]
        B, nm = z.shape
        for alpha in ALPHAS:
            lbc, lbb = D[f"{name}|lb_cert_{alpha}"], D[f"{name}|lb_bb_{alpha}"]
            for tau in TAUS:
                for arm in DEMO_ARMS:
                    rng = np.random.default_rng(0)
                    rec = []
                    for b in range(B):
                        if arm == "cert":
                            j = int(np.argmax(lbc[b] + rng.uniform(0, 1e-9, nm)))
                            j = j if lbc[b, j] >= tau else None
                        elif arm == "oracle":
                            j = int(np.argmax(z[b]))
                            j = j if z[b, j] >= tau else None
                        elif arm == "random":
                            j = int(rng.integers(nm))
                        else:
                            j = None if (arm == "bb" and lbb[b] < tau) else \
                                int(np.argmax(ypos + rng.uniform(0, 1e-6, nm)))
                        if j is None:
                            rec.append((0.0, 0.0, 0, 1, 0))
                        else:
                            m = mode[b, 4 * j:4 * j + 4]
                            kind = demo_kind(m)
                            rec.append((float(dm.p(z[b, j])), kind_p.get(kind, np.nan), int(z[b, j] < tau), 0,
                                        int(fmod[b, j])))
                    R = np.array(rec, float)
                    res[f"{name}|{alpha}|{tau}|{arm}"] = dict(
                        success=float(R[:, 0].mean()), success_kindrate=float(np.nanmean(R[:, 1])),
                        unsafe=float(R[:, 2].mean()), abstain=float(R[:, 3].mean()),
                        sent_faulted=float(R[:, 4].mean()))
                # power: cert vs bb, cert vs random
                for other in ("bb", "random"):
                    p1 = res[f"{name}|{alpha}|{tau}|cert"]["success"]
                    p0 = res[f"{name}|{alpha}|{tau}|{other}"]["success"]
                    rng = np.random.default_rng(1)
                    for n in n_runs:
                        hits = 0
                        for _ in range(1000):
                            a, b0 = rng.binomial(n, p1), rng.binomial(n, p0)
                            hits += fisher_exact([[a, n - a], [b0, n - b0]], alternative="greater")[1] < 0.05
                        res[f"{name}|{alpha}|{tau}|power_cert_vs_{other}|n{n}"] = hits / 1000
        res[f"{name}|z_faulted_median"] = float(np.median(z[fmod])) if fmod.any() else None
        res[f"{name}|z_healthy_median"] = float(np.median(z[~fmod])) if (~fmod).any() else None
    res["kind_p"] = kind_p
    (OUT / "demo_results.json").write_text(json.dumps(res, indent=1))
    print("wrote demo_results.json")
    return res


def demo_kind(m):
    m = np.asarray(m)
    f = np.flatnonzero(m > 0)
    if len(f) == 0:
        return "F0"
    code = {2: "F2", 3: "F2", 4: "F4", 1: "F1"}[int(m[f[0]])]
    return code + ("x2d" if len(f) == 2 else "")


if __name__ == "__main__":
    if sys.argv[1:2] == ["alpha05"]:
        alpha_rows(0.05)
    elif sys.argv[1:2] == ["demo"]:
        demo_main()
    else:
        main()

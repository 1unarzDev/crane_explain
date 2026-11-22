"""Design of the PROTOCOL_V2 real-robot bonus study (docs/PROTOCOL_V2.md): trial counts, hours, sealed schedule
expansion, and design-stage power. No data are read here. Numbers printed by `main()` are design numbers, not results.

    PYTHONPATH=. .venv/bin/python -m capxfer.analysis.realv2_design            # hours + power tables

Sealed randomization: a team member draws a 256-bit seed offline; the freeze record holds only
`commitment(seed)` = SHA-256(seed || b"capxfer-v2"). `expand_schedule(seed)` deterministically expands the seed into
every random choice of the study (fault modes, CAL/TEST split, block and trial orders, docking-episode assignments).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math

import numpy as np
from scipy import stats as sps

from . import stats as S

# ----------------------------------------------------------------------------------------------------- constants
MODULES = ("M1", "M2", "M3", "M4")
WHEELS = ("fl", "fr", "rl", "rr")
# capxfer.assembly.PHYSICAL_TYPES module membership (read-only copy; asserted equal in tests)
TYPE_MODULES = {"single": ("M1",), "pair": ("M1", "M3"), "L3": ("M2", "M3", "M4"), "square4": ("M1", "M2", "M3", "M4")}
MULTI = ("pair", "L3", "square4")            # size order (8, 12, 16 wheels)
N_WHEELS = {"single": 4, "pair": 8, "L3": 12, "square4": 16}

TRIAL_S = 29.7            # one full probe-library trial (field packet; WS-B/P measured schedule length)
GAP_S = 3.0               # between-trial gap -> 32.7 s per trial, ~110 trials/h (design maximum)
RATE_REALISTIC = 90.0     # trials/h, realistic pace (field packet, TEAM-CONFIRM)
REBUILD_MIN = (3.0, 5.0)  # minutes per reassembly / split (TEAM-CONFIRM)

# masking set (MASK)
MASK_REPEATS = 3          # repeats per (build, module, wheel) passive-fault condition
MASK_HEALTHY = 6          # healthy trials per build (baseline for the loss)
# docking demo (DEMO, D11 + D15 / docs/ideas/CHOICE.md): L3, one firmware COAST fault on each of two sealed-random
# modules, narrow bay (sideways entry only). Paired episodes: every module chosen by >= 1 arm is docked once.
DEMO_EPISODES = 20
DEMO_ASSEMBLY = "L3"
DEMO_MODULES = TYPE_MODULES["L3"]           # (M2, M3, M4)
DEMO_FAULT = "coast"                        # firmware, one wheel on each of 2 modules
DEMO_BAY = "narrow"                         # width = module x-length + 0.12 m (0.36 m for the measured 0.24 m module), CHOICE sec. 1
DOCK_OFFSETS_M = (-0.10, -0.05, 0.0, 0.05, 0.10)
# Predicted arm rates (sim, alpha 0.10, tau 0.8 RELATIVE, L3 coast on 2 modules; 256 runs): success / unsafe dispatch /
# abstain. PROTOCOL_V2 v5 (WS-FREEZE, pre-freeze): MEASURED robot (params 8644befd, confirmed controller, models/M1, SPLIT_M1
# library), docs/ideas/MEASURED.md "DEMO predictions, measured robot", ~/.cache/capxfer/MS/demo_predictions_measured.json
# (sha256 f2ee6efd...). Superseded estimated-robot rates (CHOICE sec. 4): cert (0.97, 0.00, 0.00), carry_over (0.44, 0.48,
# 0.11), random (0.45, 0.61, 0.0), oracle (0.97, 0.0, 0.0).
CHOICE_RATES = dict(cert=(0.98, 0.00, 0.02), carry_over=(0.24, 0.41, 0.39), random=(0.40, 0.66, 0.0),
                    oracle=(1.00, 0.0, 0.0))
DOCK_MIN = (2.0, 3.0)                       # minutes per dock incl. placement/reset (TEAM-CONFIRM)
# D16 (WS-KIT2, 2026-11-21, owner-approved via the integrator; PROTOCOL_V2 change log v3): the team runs its own code and
# cannot run choose_module in the field, so EVERY episode docks ALL THREE L3 modules after the split (sealed dock order);
# every arm's choice is computed OFFLINE afterwards from the pre-split probe CSVs with the frozen bundle (oracle from
# the lone probes' truth), blind to the dock outcomes. Arm outcomes are the same dock outcomes as before, so the R4
# power is unchanged; only the robot time grows (3 docks per episode instead of ~1.9).
DEMO_DOCK_ALL = True
# per-episode minutes (TEAM-CONFIRM) excluding the docks: rejoin L3, set firmware faults, assembly probe, split,
# 3 lone-module probes (truth for the oracle and for R3-type reporting), clear faults
DEMO_EPISODE_FIXED_MIN = dict(rejoin=REBUILD_MIN, set_faults=(0.3, 0.5), assembly_probe=(0.55, 0.55),
                              split=REBUILD_MIN, module_probes=(1.65, 1.65), clear_faults=(0.2, 0.3))
# simulator-validation / repeatability / AprilTag add-on (SIMVAL)
SIMVAL_DAY2 = dict(module="M1", healthy=5, faulted=5)     # day-2 repeat of a day-1 MASK singles condition
APRILTAG_POSES = 5
# fixed pre-freeze steps (field packet card hours; TEAM-CONFIRM)
PRE_FREEZE_H = dict(t0=(2.0, 3.0), hook_tests=(0.5, 1.0), shakedown=(0.2, 0.3), sysid=(1.1, 1.4))

TAG = b"capxfer-v2"


def mw_conditions():
    """The 16 (module, wheel) passive-fault conditions."""
    return [(m, w) for m in MODULES for w in WHEELS]


def mask_builds():
    """Build list of the masking set: name -> list of (module, wheel) conditions it carries."""
    out = {}
    for t in ("square4", "L3", "pair"):
        out[t] = [(m, w) for (m, w) in mw_conditions() if m in TYPE_MODULES[t]]
    for m in MODULES:
        out[f"single_{m}"] = [(m, w) for w in WHEELS]
    return out


def mask_counts():
    b = mask_builds()
    rows = {k: dict(conditions=len(v), faulted=len(v) * MASK_REPEATS, healthy=MASK_HEALTHY,
                    trials=len(v) * MASK_REPEATS + MASK_HEALTHY) for k, v in b.items()}
    tot = sum(r["trials"] for r in rows.values())
    return dict(per_build=rows, trials=tot, builds=len(rows),
                assembly_conditions=sum(len(v) for k, v in b.items() if not k.startswith("single")))


def _hours(trials, rebuilds):
    lo = trials * (TRIAL_S + GAP_S) / 3600 + rebuilds * REBUILD_MIN[0] / 60
    hi = trials / RATE_REALISTIC + rebuilds * REBUILD_MIN[1] / 60
    return (round(lo, 2), round(hi, 2))


def hours_table():
    """Robot-hours per step (lo = design pace + 3 min rebuilds; hi = realistic pace + 5 min rebuilds)."""
    mc = mask_counts()
    simval_trials = SIMVAL_DAY2["healthy"] + SIMVAL_DAY2["faulted"]
    nd = docks_per_episode()
    ep_lo = sum(v[0] for v in DEMO_EPISODE_FIXED_MIN.values()) + nd * DOCK_MIN[0]
    ep_hi = sum(v[1] for v in DEMO_EPISODE_FIXED_MIN.values()) + nd * DOCK_MIN[1]
    rows = dict(PRE_FREEZE_H)
    rows["simval_day2_and_apriltag"] = tuple(round(a + b, 2) for a, b in zip(_hours(simval_trials, 1),
                                                                              (APRILTAG_POSES * 1.5 / 60 + 0.1,
                                                                               APRILTAG_POSES * 3.0 / 60 + 0.2)))
    rows["mask"] = _hours(mc["trials"], mc["builds"])
    rows["demo"] = (round(DEMO_EPISODES * ep_lo / 60, 2), round(DEMO_EPISODES * ep_hi / 60, 2))
    total = (round(sum(v[0] for v in rows.values()), 2), round(sum(v[1] for v in rows.values()), 2))
    return dict(rows=rows, total=total, mask_trials=mc["trials"], demo_episode_min=(round(ep_lo, 2), round(ep_hi, 2)),
                demo_expected_docks=round(nd, 2), demo_dock_all=DEMO_DOCK_ALL)


def docks_per_episode():
    """Docks per DEMO episode: all three L3 modules under D16 (DEMO_DOCK_ALL), else the deduplicated expectation."""
    return float(len(DEMO_MODULES)) if DEMO_DOCK_ALL else expected_docks()["mean"]


# ----------------------------------------------------------------------------------------------- sealed schedule
def commitment(seed: bytes) -> str:
    """SHA-256(seed || b'capxfer-v2'), the only seed-derived value in the freeze record."""
    return hashlib.sha256(bytes(seed) + TAG).hexdigest()


def _rng(seed: bytes, stream: str):
    h = hashlib.sha256(bytes(seed) + TAG + b"|" + stream.encode()).digest()
    return np.random.default_rng(int.from_bytes(h, "big"))


def expand_schedule(seed: bytes) -> dict:
    """Deterministic expansion of the sealed seed into every random choice of the study. Pure function of `seed`."""
    if len(seed) < 32:
        raise ValueError("seed must be >= 256 bits")
    # 1. passive mode per (module, wheel): coast or brake (P4: equivalent above tau*, both injected)
    r = _rng(seed, "mode")
    mode = {f"{m}:{w}": ("coast", "brake")[int(r.integers(2))] for m, w in mw_conditions()}
    # 2. CAL/TEST split for R3 at the (module, wheel) level, stratified: 2 of 4 wheels of each module are CAL
    r = _rng(seed, "calsplit")
    split = {}
    for m in MODULES:
        cal = set(r.choice(len(WHEELS), size=2, replace=False).tolist())
        for i, w in enumerate(WHEELS):
            split[f"{m}:{w}"] = "CAL" if i in cal else "TEST"
    # 3. MASK order: square4 first, then split into singles (random module order), then L3/pair in random order
    r = _rng(seed, "mask_order")
    singles = [f"single_{m}" for m in r.permutation(list(MODULES)).tolist()]
    rest = r.permutation(["L3", "pair"]).tolist()
    order = ["square4"] + singles + rest
    builds = mask_builds()
    trials = []
    for b in order:
        items = [dict(build=b, module=None, wheel=None, firmware="ok") for _ in range(MASK_HEALTHY)]
        for m, w in builds[b]:
            items += [dict(build=b, module=m, wheel=w, firmware=mode[f"{m}:{w}"], rep=k) for k in range(MASK_REPEATS)]
        rb = _rng(seed, f"mask_trials|{b}")
        perm = rb.permutation(len(items))
        trials += [items[i] for i in perm]
    for i, t in enumerate(trials):
        t["seq"] = i
        t["r3_split"] = split[f"{t['module']}:{t['wheel']}"] if t["module"] else "NONE"
    # 4. DEMO episodes (D15): the faulted module pair balanced 7/7/6 over the 3 choices, one coast wheel per faulted
    #    module, random-arm module, tie coins, dock order of the deduplicated set, start offsets
    r = _rng(seed, "demo")
    pairs = list(itertools.combinations(DEMO_MODULES, 2))
    reps = [DEMO_EPISODES // 3 + (1 if k < DEMO_EPISODES % 3 else 0) for k in range(3)]
    fp = r.permutation(np.repeat(np.arange(3), reps)).tolist()
    eps = []
    for e in range(DEMO_EPISODES):
        fm = list(pairs[fp[e]])
        eps.append(dict(episode=e, assembly=DEMO_ASSEMBLY, bay=DEMO_BAY, fault=DEMO_FAULT, faulted_modules=fm,
                        faulted_wheels={m: WHEELS[int(r.integers(4))] for m in fm},
                        random_module=DEMO_MODULES[int(r.integers(3))],
                        carry_over_tie_coin=DEMO_MODULES[int(r.integers(3))],
                        cert_tie_coin=DEMO_MODULES[int(r.integers(3))],
                        oracle_tie_coin=DEMO_MODULES[int(r.integers(3))],
                        dock_order=r.permutation(list(DEMO_MODULES)).tolist(),
                        offset_m={m: float(r.choice(DOCK_OFFSETS_M)) for m in DEMO_MODULES}))
    # 5. SIMVAL day 2: M1 healthy + one sealed M1 wheel (its MASK mode), shuffled; AprilTag pose order
    r = _rng(seed, "simval")
    w2 = WHEELS[int(r.integers(4))]
    d2 = ([dict(module="M1", wheel=None, firmware="ok")] * SIMVAL_DAY2["healthy"]
          + [dict(module="M1", wheel=w2, firmware=mode[f"M1:{w2}"])] * SIMVAL_DAY2["faulted"])
    d2 = [dict(x, seq=i) for i, x in enumerate([d2[j] for j in r.permutation(len(d2))])]
    tag_order = r.permutation(APRILTAG_POSES).tolist()
    out = dict(version="capxfer-v2-schedule/1", commitment=commitment(seed), mode=mode, r3_split=split,
               mask_order=order, mask_trials=trials, demo=eps, simval_day2=d2, apriltag_pose_order=tag_order)
    return out


def schedule_sha256(sched: dict) -> str:
    return hashlib.sha256(json.dumps(sched, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------- design power
def power_r1_trend(sigma_L, n_rep=2000, seed=0, loss=None):
    """MC power of the R1 stratified trend test at one-sided 0.05. Condition-level losses = predicted medians
    (sim: pair 0.049, L3 0.025, square4 0.019) + per-condition spread + estimation noise N(0, sigma_L)."""
    from .realv2 import stratified_trend_test
    loss = loss or {"pair": 0.049, "L3": 0.025, "square4": 0.019}
    rng = np.random.default_rng(seed)
    rows = [(f"{m}:{w}", t) for t in MULTI for (m, w) in mw_conditions() if m in TYPE_MODULES[t]]
    hits = 0
    for _ in range(n_rep):
        L = [max(0.0, loss[t] * rng.lognormal(0, 0.3)) + rng.normal(0, sigma_L) for _, t in rows]
        p = stratified_trend_test([b for b, _ in rows], [t for _, t in rows], L, MULTI)["p"]
        hits += p < 0.05
    return hits / n_rep


def power_r1_equiv(n, sigma_d, bias=0.0, delta=0.05, n_rep=1000, seed=0, n_boot=500):
    """MC power of the R1 equivalence criterion for one type (90% bootstrap CI of mean d inside +-delta)."""
    from .realv2 import equivalence_bootstrap
    rng = np.random.default_rng(seed)
    ok = 0
    for i in range(n_rep):
        d = bias + rng.normal(0, sigma_d, n)
        ok += equivalence_bootstrap(d, delta, n_boot=n_boot, seed=i)["pass"]
    return ok / n_rep


def power_r2(n, p_true, pi0=0.25, alpha=0.05):
    """Exact power of the one-sided binomial test H0: pi <= pi0."""
    k = np.arange(n + 1)
    crit = next(int(x) for x in k if sps.binom.sf(x - 1, n, pi0) < alpha)
    return float(sps.binom.sf(crit - 1, n, p_true)), crit


def power_r3(n_null, fg_true, alpha=0.10, conf=0.95, deff=1.0):
    """(P(not refuted: CP one-sided lower <= alpha), P(strong: CP one-sided upper <= 2 alpha)) at true false-go."""
    ne = max(int(round(n_null / deff)), 1)
    k = np.arange(ne + 1)
    pmf = sps.binom.pmf(k, ne, fg_true)
    nr = np.array([S.cp_lower(int(x), ne, conf) <= alpha for x in k])
    st = np.array([S.cp_upper(int(x), ne, conf) <= 2 * alpha for x in k])
    return float(pmf @ nr), float(pmf @ st)


def _choice_model(rates=None, p_capable=None, p_faulted=0.05):
    """Module-level model matched to CHOICE's predicted arm rates. The capable module docks with p_capable
    (default = the oracle's predicted success), a coast-faulted module with p_faulted (CHOICE dock table: z 0.5-0.7 ->
    0.05-0.15; single coast 0/20 direct). P(arm sends the capable module | dispatch) is solved from its success rate."""
    rates = rates or CHOICE_RATES
    pc = rates["oracle"][0] if p_capable is None else p_capable
    q = {}
    for a in ("cert", "carry_over", "random"):
        succ, _, abst = rates[a]
        disp = max(1e-9, 1 - abst)
        q[a] = float(np.clip((succ / disp - p_faulted) / (pc - p_faulted), 0, 1))
    return pc, p_faulted, q


def simulate_demo(n_ep, rng, rates=None, p_faulted=0.05):
    """One simulated DEMO (paired episodes). Returns per-episode arm outcomes and `docks` = the number of distinct
    modules chosen by >= 1 arm (the deduplicated D15 dock count). Every module's dock outcome is drawn, so the arm
    outcomes are identical under D16 (all three modules docked; `docked` = 3)."""
    rates = rates or CHOICE_RATES
    pc, pf, q = _choice_model(rates, p_faulted=p_faulted)
    out = dict(cert=[], carry_over=[], random=[], oracle=[], docks=[])
    for _ in range(n_ep):
        cap = int(rng.integers(3))                                   # index of the capable module
        y = [float(rng.random() < (pc if m == cap else pf)) for m in range(3)]   # one dock per module (dedup)
        others = [m for m in range(3) if m != cap]
        ch = {}
        for a in ("cert", "carry_over", "random"):
            if rng.random() < rates[a][2]:
                ch[a] = None
            else:
                ch[a] = cap if rng.random() < q[a] else int(rng.choice(others))
        ch["oracle"] = cap
        for a, m in ch.items():
            out[a].append(0.0 if m is None else y[m])
        out["docks"].append(len({m for m in ch.values() if m is not None}))
    out = {k: np.array(v) for k, v in out.items()}
    out["docked"] = np.full(n_ep, 3 if DEMO_DOCK_ALL else -1)
    return out


def expected_docks(n_rep=20000, seed=0, rates=None):
    d = simulate_demo(n_rep, np.random.default_rng(seed), rates)["docks"]
    return dict(mean=float(d.mean()), p3=float((d == 3).mean()), max=int(d.max()))


def power_r4(n_ep=DEMO_EPISODES, n_rep=3000, seed=0, alpha=0.05, rates=None, p_faulted=0.05):
    """MC power of the R4 rule: exact one-sided McNemar cert > carry-over and cert > random on paired episodes,
    Holm over the 2; SUPPORTED iff both adjusted p < alpha. Returns (P(both), P(cert>carry), P(cert>random))."""
    rng = np.random.default_rng(seed)
    both = c1 = c2 = 0
    for _ in range(n_rep):
        o = simulate_demo(n_ep, rng, rates, p_faulted)
        p1 = S.mcnemar_greater(o["cert"].astype(bool), o["carry_over"].astype(bool))[0]
        p2 = S.mcnemar_greater(o["cert"].astype(bool), o["random"].astype(bool))[0]
        h = S.holm([p1, p2])
        both += bool((h < alpha).all())
        c1 += h[0] < alpha
        c2 += h[1] < alpha
    return both / n_rep, c1 / n_rep, c2 / n_rep


def main():
    ht = hours_table()
    print("## Robot-hours (lo = 110 trials/h + 3 min rebuilds; hi = 90 trials/h + 5 min rebuilds)")
    for k, v in ht["rows"].items():
        print(f"{k} | {v[0]:.2f} | {v[1]:.2f}")
    print(f"TOTAL | {ht['total'][0]:.2f} | {ht['total'][1]:.2f}   (MASK trials {ht['mask_trials']}, DEMO episode "
          f"{ht['demo_episode_min'][0]}-{ht['demo_episode_min'][1]} min)")
    print("\n## R1 trend power (stratified exact test, one-sided 0.05) by condition-level noise SD of L")
    for s in (0.005, 0.01, 0.015, 0.02, 0.03):
        print(s, f"{power_r1_trend(s, n_rep=1000):.3f}")
    print("\n## R1 equivalence (delta 0.05) power by type n and SD of d (bias 0 / 0.02)")
    for n in (8, 12, 16):
        print(n, *[f"{power_r1_equiv(n, s, b, n_rep=300):.2f}" for s in (0.02, 0.04, 0.06) for b in (0.0, 0.02)])
    print("\n## R2 exact binomial power, H0 pi <= 0.25")
    for n in (12, 16):
        print(n, *[f"{power_r2(n, p)[0]:.3f}" for p in (0.4, 0.5, 0.6, 0.85)], "crit", power_r2(n, 0.5)[1])
    print("\n## R3 (P(not refuted), P(strong)) at n_null_test, deff")
    for n in (30, 46, 60):
        for d in (1.0, 2.0):
            print(n, d, *[tuple(round(x, 3) for x in power_r3(n, f, deff=d)) for f in (0.05, 0.10, 0.20, 0.30)])
    ed = expected_docks()
    print(f"\n## R4 DEMO (L3, coast on 2 modules): docks per episode {docks_per_episode():.0f} (D16: all three); "
          f"distinct arm choices per episode {ed['mean']:.2f} (P(3) = {ed['p3']:.2f})")
    print("scenario | episodes | P(both, Holm) | P(cert>carry) | P(cert>random)")
    pess = dict(cert=(0.85, 0.05, 0.05), carry_over=(0.55, 0.40, 0.11), random=(0.45, 0.61, 0.0), oracle=(0.95, 0, 0))
    worse = dict(cert=(0.75, 0.05, 0.15), carry_over=(0.55, 0.40, 0.11), random=(0.50, 0.55, 0.0), oracle=(0.95, 0, 0))
    for lbl, rt in (("CHOICE predicted", None), ("pessimistic cert 0.85 / bb 0.55", pess),
                    ("cert 0.75 (abstain 0.15) / bb 0.55 / random 0.50", worse)):
        for n in (12, 16, 20):
            print(lbl, n, *[f"{x:.3f}" for x in power_r4(n, n_rep=2000, rates=rt)], sep=" | ")


if __name__ == "__main__":
    main()

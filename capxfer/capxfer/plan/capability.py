"""Capability -> planner adapter (contract C6: `from_intervals(lo, hi, asm) -> CapSet`).

Input: per-direction capability ratios over the 14 probe targets (probes.TARGET_KEYS: P1+-, P2+-, P3+-, P4+-, P5+-,
P6x+-, P6y+-). A ratio rho = <xi_real, xi_cmd>_W / ||xi_cmd||^2_W is the fraction of a commanded probe twist that the
assembly realises (W = diag(1, 1, Lc^2)); `drift` is the W-norm of the orthogonal residual relative to ||xi_cmd||_W.

Output: a CapSet, the achievable-twist description a lattice planner consumes:
  * `usable[d]` per probe direction d (gate on the CONSERVATIVE lower bound lo and on the drift bound),
  * `expected(xi_cmd)` the predicted realised twist (planner prediction; uses the interval midpoint),
  * `drift(xi_cmd)` the relative orthogonal-error bound used to penalise / reject primitives,
  * `allowed(xi_cmd)` whether every component of xi_cmd lies in usable directions,
  * `vmax` per linear axis (0.45 m/s if the P6 high-speed probe is usable, else the 0.30 m/s probe speed).

The twist model is component-wise and piecewise-linear in direction: the linear part of a command is written as a
conic combination of the two adjacent probe directions among the 8 planar ones (+-x, +-y, +-(1,1), +-(1,-1)), each
scaled by its own ratio; the yaw part is scaled by the P5 ratio of its sign. Ratios are assumed speed-independent at or
below the probe speeds (the docking planner commands <= 0.2 m/s, below the 0.30 m/s probes). ASSUMPTION, stated.

Constructors: `from_intervals` (proposed: conformal bounds), `from_point` (oracle / point estimates), `omni` (assumes a
fully healthy mecanum base: rho = 1, drift = 0). `oracle_capset` measures the true ratios by running the probe
library in capxfer.sim on the true health and domain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..probes import TARGET_KEYS

KEYS = tuple(TARGET_KEYS)
KIDX = {k: i for i, k in enumerate(KEYS)}
# planar probe directions (unit, body frame) in angular order, with their target keys
PLANAR = (
    ("P1+", (1.0, 0.0)), ("P3+", (math.sqrt(.5), math.sqrt(.5))), ("P2+", (0.0, 1.0)), ("P4-", (-math.sqrt(.5), math.sqrt(.5))),
    ("P1-", (-1.0, 0.0)), ("P3-", (-math.sqrt(.5), -math.sqrt(.5))), ("P2-", (0.0, -1.0)), ("P4+", (math.sqrt(.5), -math.sqrt(.5))),
)
YAW = {+1: "P5+", -1: "P5-"}
V_PROBE = 0.30
V_FAST = 0.45

# gating defaults (pre-specifiable constants)
def _study_threshold(fallback=0.5):
    """capability_threshold from configs/study_constants.yaml (integrator-owned single source of truth, shared with
    capxfer.explain.contracts.study_threshold; CAPXFER_STUDY_CONSTANTS overrides the path)."""
    import os
    from pathlib import Path
    import yaml
    path = Path(os.environ.get("CAPXFER_STUDY_CONSTANTS",
                               Path(__file__).resolve().parents[2] / "configs" / "study_constants.yaml"))
    if path.exists():
        with open(path) as fh:
            return float(yaml.safe_load(fh)["capability_threshold"])
    return fallback


RHO_MIN = _study_threshold()   # a direction is usable only if its conservative ratio is at least this
DRIFT_MAX = 0.30       # ... and its drift bound is at most this (planar directions P1-P4, P6) ...
DRIFT_MAX_YAW = 1.0    # ... or this for in-place yaw (P5). Pilot-informed (docs/coordination/F.md, pilot v2): the v1
                       # gates 0.10 / 0.25 caused planner abstentions that cost successes under the closed-loop
                       # executor; drift still raises primitive cost. Pre-specify before use.


@dataclass
class CapSet:
    name: str
    lo: np.ndarray                      # (14,) conservative ratios
    hi: np.ndarray                      # (14,)
    drift_b: np.ndarray                 # (14,) relative drift bound
    Lc: float
    rho_min: float = RHO_MIN
    drift_max: float = DRIFT_MAX
    drift_max_yaw: float = DRIFT_MAX_YAW
    meta: dict = field(default_factory=dict)

    # -------------------------------------------------------------- per-direction
    @property
    def mid(self):
        return 0.5 * (self.lo + self.hi)

    def usable_key(self, key):
        i = KIDX[key]
        dmax = self.drift_max_yaw if key.startswith("P5") else self.drift_max
        return bool(self.lo[i] >= self.rho_min and self.drift_b[i] <= dmax)

    @property
    def usable(self):
        return {k: self.usable_key(k) for k in KEYS}

    @property
    def vmax(self):
        """Max commanded linear speed per axis direction (m/s)."""
        out = {}
        for ax, fast in (("x", "P6x"), ("y", "P6y")):
            for s in "+-":
                slow_ok = self.usable_key(("P1" if ax == "x" else "P2") + s)
                out[ax + s] = V_FAST if (slow_ok and self.usable_key(fast + s)) else (V_PROBE if slow_ok else 0.0)
        return out

    # -------------------------------------------------------------- general twists
    def _decompose(self, v):
        """Linear part v (2,) -> [(key, coef, unit dir)] over the two adjacent planar probe directions."""
        n = math.hypot(v[0], v[1])
        if n < 1e-12:
            return []
        a = math.atan2(v[1], v[0]) % (2 * math.pi)
        k = int(a // (math.pi / 4)) % 8
        (k1, d1), (k2, d2) = PLANAR[k], PLANAR[(k + 1) % 8]
        M = np.array([d1, d2]).T
        c = np.linalg.solve(M, np.asarray(v, float))
        out = []
        for key, d, ci in ((k1, d1, c[0]), (k2, d2, c[1])):
            if ci > 1e-9 * n:
                out.append((key, float(ci), np.array(d)))
        return out

    def _parts(self, xi, tol=0.03):
        """Significant components [(key, weight)] of xi; components below `tol` of the W-norm are ignored."""
        xi = np.asarray(xi, float)
        nw = math.sqrt(xi[0] ** 2 + xi[1] ** 2 + (self.Lc * xi[2]) ** 2)
        if nw < 1e-12:
            return []
        parts = [(key, c / nw) for key, c, _ in self._decompose(xi[:2])]
        if abs(xi[2]) > 1e-12:
            parts.append((YAW[int(np.sign(xi[2]))], self.Lc * abs(xi[2]) / nw))
        return [(k, w) for k, w in parts if w > tol]

    def allowed(self, xi, tol=0.03):
        return all(self.usable_key(k) for k, _ in self._parts(xi, tol))

    def drift(self, xi):
        p = self._parts(xi, 0.0)
        return max([self.drift_b[KIDX[k]] for k, w in p if w > 0.03] + [0.0])

    def expected(self, xi, which="mid"):
        """Predicted realised twist for command xi (component-wise ratio model)."""
        xi = np.asarray(xi, float)
        r = getattr(self, which) if which in ("lo", "hi") else self.mid
        out = np.zeros(3)
        for key, c, d in self._decompose(xi[:2]):
            out[:2] += r[KIDX[key]] * c * d
        if abs(xi[2]) > 1e-12:
            out[2] = r[KIDX[YAW[int(np.sign(xi[2]))]]] * xi[2]
        return out

    def command_for(self, xi_exp, which="mid"):
        """Inverse of `expected`: the command whose predicted realised twist is xi_exp (exact within a direction cone,
        since positive per-direction scaling keeps the cone). Unusable / zero-ratio components map to 0."""
        xi_exp = np.asarray(xi_exp, float)
        r = getattr(self, which) if which in ("lo", "hi") else self.mid
        out = np.zeros(3)
        for key, c, d in self._decompose(xi_exp[:2]):
            rk = r[KIDX[key]]
            if rk > 1e-6:
                out[:2] += c / rk * d
        if abs(xi_exp[2]) > 1e-12:
            rk = r[KIDX[YAW[int(np.sign(xi_exp[2]))]]]
            out[2] = xi_exp[2] / rk if rk > 1e-6 else 0.0
        return out

    def summary(self):
        u = self.usable
        return dict(name=self.name, usable=[k for k in KEYS if u[k]], unusable=[k for k in KEYS if not u[k]],
                    strafe=bool(u["P2+"] and u["P2-"]), lo=np.round(self.lo, 3).tolist(),
                    drift=np.round(self.drift_b, 3).tolist())


def _arr(x):
    x = np.asarray(x, float).reshape(-1)
    if x.shape != (14,):
        raise ValueError(f"expected 14 capability values ordered as probes.TARGET_KEYS, got {x.shape}")
    return x


def from_intervals(lo, hi, asm, drift=None, name="proposed", **kw):
    """Conservative CapSet from per-direction conformal intervals [lo, hi] (each (14,), TARGET_KEYS order).

    drift: optional (14,) drift upper bound. When absent (the C4 interval contract has no drift output) the drift bound
    is the conservative proxy 1 - lo clipped to [0, 1]: lost capability is assumed to possibly reappear as orthogonal
    error. ASSUMPTION, stated."""
    lo, hi = _arr(lo), _arr(hi)
    if np.any(hi < lo - 1e-9):
        raise ValueError("hi < lo")
    d = _arr(drift) if drift is not None else np.clip(1.0 - lo, 0.0, 1.0)
    return CapSet(name, lo, hi, d, float(asm.char_length()), **kw)


def from_point(rho, asm, drift=None, name="point", **kw):
    rho = _arr(rho)
    d = _arr(drift) if drift is not None else np.zeros(14)
    return CapSet(name, rho.copy(), rho.copy(), d, float(asm.char_length()), **kw)


def omni(asm, **kw):
    """Assumes a fully healthy mecanum base (every direction at ratio 1, no drift)."""
    return from_point(np.ones(14), asm, np.zeros(14), name="omni", **kw)


# ---------------------------------------------------------------------------------------------- truth from the sim
def measure_ratios(R, asm, schedule=None):
    """Ratios and drift (B,14) from a sim result R (ground-truth body twist over the probe hold windows)."""
    sched = schedule or R["meta"]["schedule"]
    Lc = asm.char_length()
    W = np.array([1.0, 1.0, Lc ** 2])
    t = R["t"].cpu().numpy()
    tw = R["twist"].cpu().numpy()
    B = tw.shape[0]
    rho = np.full((B, 14), np.nan)
    dr = np.full((B, 14), np.nan)
    for pid, leg, ta, tb, cmd in sched.windows():
        key = f"{pid}{'+' if leg > 0 else '-'}"
        m = (t >= ta) & (t < tb)
        xr = tw[:, m].mean(1)
        den = float((cmd * cmd * W).sum())
        r = (xr * cmd * W).sum(1) / den
        rho[:, KIDX[key]] = r
        dr[:, KIDX[key]] = np.sqrt((((xr - r[:, None] * cmd) ** 2) * W).sum(1) / den)
    return rho, dr


def true_ratios(asm, params, dom, mode, g, device="cpu", seed=0):
    """Run the full probe library in capxfer.sim with the true health and domain; returns (rho, drift) (B,14)."""
    from .. import sim
    from ..probes import Schedule
    dom = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in dom.items()}
    sched = Schedule(asm.char_length())
    R = sim.simulate(asm, params, dom, mode, g, schedule=sched, device=device, seed=seed)
    return measure_ratios(R, asm, sched)


def oracle_capset(rho, drift, asm, **kw):
    return from_point(rho, asm, drift, name="oracle", **kw)

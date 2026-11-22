"""Planner interface and the shared SE(2) lattice engine (weighted hybrid A*).

Every planner = a MotionModel generator (what it believes the robot can do) + this engine. Planners differ only
in what they know and how they turn it into primitives, so comparisons isolate the information each one uses.

Planner protocol (evaluate.py drives it):
  setup(world, task, known_graph, rng)       new episode; known_graph = diagnosed graph at t = 0
  update(known_graph, t, hypotheses=None)     knowledge changed (detected fault, detach, sensor loss)
  plan(pose, t) -> Plan                       plan from the current pose
  payload_placement() -> dict | None          where this planner puts the payload (carry); None = refuses
  reconfiguration() -> list of ops            operators the planner wants executed before moving (SEQ etc.)
Plan.status: ok | infeasible (declared, with reason) | fail (search exhausted)
"""
import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .._shim import rot, wrap  # VENDOR: was `from ..common import rot, wrap`
from .._shim import POS_TOL, HEAD_TOL  # VENDOR: was `from ..tasks import POS_TOL, HEAD_TOL`


@dataclass
class Prim:
    xi: np.ndarray            # body twist (vx, vy, w)
    T: float                  # duration (s)
    kind: str = "roll"        # roll | slide
    cost_mult: float = 1.0


@dataclass
class MotionModel:
    prims: list
    pts: np.ndarray           # body-frame collision sample points (boundary of the believed footprint)
    margin: float = 0.05
    bins: int = 16
    v_heur: float = 0.5
    w_heur: float = 1.5
    r_in: float = 0.0         # clearance radius around the origin used by the 2D heuristic
    allowed: object = None    # callable(xi) -> bool for analytic shots (None: no analytic shot)
    ray: object = None        # callable(unit xi) -> max scale
    name: str = ""


@dataclass
class Plan:
    status: str
    segments: list = field(default_factory=list)   # [(xi, T, kind)]
    reason: str = ""
    expansions: int = 0
    compute_s: float = 0.0
    goal_index: int = -1
    poses: list = field(default_factory=list)


def integrate(pose, xi, T):
    """Exact pose after constant body twist xi for time T."""
    x, y, th = pose
    vx, vy, w = xi
    if abs(w) < 1e-9:
        dx, dy = vx * T, vy * T
    else:
        a = w * T
        dx = (vx * math.sin(a) + vy * (math.cos(a) - 1)) / w
        dy = (vx * (1 - math.cos(a)) + vy * math.sin(a)) / w
    c, s = math.cos(th), math.sin(th)
    return (x + c * dx - s * dy, y + s * dx + c * dy, th + w * T)


def se2_log(rel):
    """Body twist that maps identity to the relative pose rel = (dx, dy, dth) in unit time."""
    dx, dy, a = rel
    if abs(a) < 1e-9:
        return np.array([dx, dy, 0.0])
    w = a
    A = np.array([[math.sin(a), math.cos(a) - 1], [1 - math.cos(a), math.sin(a)]]) / w
    v = np.linalg.solve(A, [dx, dy])
    return np.array([v[0], v[1], w])


def sweep_offsets(prim, pts, R, step=0.04):
    """Body-frame-at-start offsets of the sample points along the primitive."""
    vx, vy, w = prim.xi
    L = math.hypot(vx, vy) * prim.T + abs(w) * prim.T * R
    n = max(1, int(math.ceil(L / step)))
    out = []
    for k in range(1, n + 1):
        x, y, th = integrate((0.0, 0.0, 0.0), prim.xi, prim.T * k / n)
        out.append(pts @ rot(th).T + [x, y])
    return np.vstack(out)


class Lattice:
    """Weighted hybrid A* over (x, y, heading bin) with continuous positions and binned closed set."""

    def __init__(self, world, model, cell=0.1, eps=2.0, budget=1200, wall_s=2.5):
        self.w, self.m = world, model
        self.cell, self.eps, self.budget, self.wall_s = cell, eps, budget, wall_s
        self.R = float(np.hypot(model.pts[:, 0], model.pts[:, 1]).max()) if len(model.pts) else 0.3
        self.grid = world.inflated(model.margin)
        self.dth = 2 * math.pi / model.bins
        self.base = []
        for p in model.prims:
            offs = sweep_offsets(p, model.pts, self.R)
            end = integrate((0.0, 0.0, 0.0), p.xi, p.T)
            db = int(round(p.xi[2] * p.T / self.dth))
            self.base.append((offs, end, db))
        self._cache = {}

    def _at(self, h, th0):
        key = (h, round(th0, 9))
        if key not in self._cache:
            th = th0 + h * self.dth
            R = rot(th)
            offs = [b[0] @ R.T for b in self.base]
            lens = np.array([len(o) for o in offs])
            starts = np.concatenate([[0], np.cumsum(lens)[:-1]]) if len(lens) else np.zeros(0, int)
            ends = [R @ np.array(b[1][:2]) for b in self.base]
            self._cache[key] = (np.vstack(offs) if offs else np.zeros((0, 2)), starts, ends)
        return self._cache[key]

    def collides(self, pose_pts):
        return self.w.occupied_pts(pose_pts, self.grid)

    def pose_free(self, pose):
        x, y, th = pose
        P = self.m.pts @ rot(th).T + [x, y]
        return not self.collides(P).any()

    def search(self, start, goals, pos_tol=POS_TOL, head_tol=HEAD_TOL):
        t0 = time.time()
        if not self.m.prims:
            return Plan("infeasible", reason="no motion primitives", compute_s=time.time() - t0)
        x0, y0, th0 = start
        H = self.m.bins
        D = self.w.dijkstra_from([(g[0], g[1]) for g in goals], max(0.0, 0.8 * self.m.r_in))
        res = self.w.res

        def h2(x, y, th):
            i, j = int(x / res), int(y / res)
            if not (0 <= i < self.w.nx and 0 <= j < self.w.ny):
                return math.inf
            d = D[i, j]
            if not math.isfinite(d):
                return math.inf
            dh = min(abs(wrap(th - g[2])) for g in goals)
            return max(d / self.m.v_heur, dh / self.m.w_heur)

        def at_goal(x, y, th):
            for k, g in enumerate(goals):
                if math.hypot(x - g[0], y - g[1]) <= pos_tol and abs(wrap(th - g[2])) <= head_tol:
                    return k
            return -1

        if at_goal(x0, y0, th0) >= 0:
            return Plan("ok", [], expansions=0, compute_s=time.time() - t0, goal_index=at_goal(x0, y0, th0))
        hs = h2(x0, y0, th0)
        if not math.isfinite(hs):
            return Plan("infeasible", reason="no 2D corridor of the believed clearance to the goal", compute_s=time.time() - t0)
        start_key = (round(x0 / self.cell), round(y0 / self.cell), 0)
        gbest = {start_key: 0.0}
        nodes = [(x0, y0, 0, -1, -1)]          # (x, y, h, parent node id, primitive index)
        heap = [(self.eps * hs, 0.0, 0)]
        exp = 0
        while heap:
            f, g, nid = heapq.heappop(heap)
            x, y, h, _, _ = nodes[nid]
            key = (round(x / self.cell), round(y / self.cell), h % H)
            if g > gbest.get(key, math.inf) + 1e-12:
                continue
            exp += 1
            if exp > self.budget or time.time() - t0 > self.wall_s:
                break
            th = th0 + h * self.dth
            offs, starts, ends = self._at(h % H, th0)
            occ = self.collides(offs + [x, y]).astype(np.int8)
            hit = np.maximum.reduceat(occ, starts) if len(starts) else np.zeros(0)
            for pi, prim in enumerate(self.m.prims):
                if hit[pi]:
                    continue
                e = ends[pi]
                nx, ny = x + e[0], y + e[1]
                nh = h + self.base[pi][2]
                nth = th0 + nh * self.dth
                ng = g + prim.T * prim.cost_mult
                k = at_goal(nx, ny, nth)
                if k >= 0:
                    nodes.append((nx, ny, nh, nid, pi))
                    return self._reconstruct(nodes, len(nodes) - 1, exp, t0, k)
                nkey = (round(nx / self.cell), round(ny / self.cell), nh % H)
                if ng >= gbest.get(nkey, math.inf):
                    continue
                hh = h2(nx, ny, nth)
                if not math.isfinite(hh):
                    continue
                gbest[nkey] = ng
                nodes.append((nx, ny, nh, nid, pi))
                heapq.heappush(heap, (ng + self.eps * hh, ng, len(nodes) - 1))
            # analytic shot near the goal
            if self.m.allowed is not None:
                for k, gpose in enumerate(goals):
                    if math.hypot(gpose[0] - x, gpose[1] - y) > 1.2:
                        continue
                    c, s = math.cos(th), math.sin(th)
                    dx, dy = gpose[0] - x, gpose[1] - y
                    rel = (c * dx + s * dy, -s * dx + c * dy, wrap(gpose[2] - th))
                    xi = se2_log(rel)
                    n = np.linalg.norm(xi)
                    if n < 1e-9 or not self.m.allowed(xi / n):
                        continue
                    sc = self.m.ray(xi / n)
                    if sc <= 1e-6:
                        continue
                    T = n / sc
                    pr = Prim(xi / T, T, "roll")
                    offs2 = sweep_offsets(pr, self.m.pts, self.R) @ rot(th).T + [x, y]
                    if self.collides(offs2).any():
                        continue
                    pl = self._reconstruct(nodes, nid, exp, t0, k)
                    pl.segments.append((pr.xi, pr.T, "roll"))
                    return pl
        return Plan("fail", reason="search budget exhausted" if heap else "search space exhausted", expansions=exp,
                    compute_s=time.time() - t0)

    def _reconstruct(self, nodes, nid, exp, t0, k):
        segs = []
        while nodes[nid][3] >= 0:
            p = self.m.prims[nodes[nid][4]]
            segs.append((np.array(p.xi), p.T, p.kind))
            nid = nodes[nid][3]
        segs.reverse()
        return Plan("ok", segs, expansions=exp, compute_s=time.time() - t0, goal_index=k)


def plan_poses(start, segments):
    out = [tuple(start)]
    p = tuple(start)
    for xi, T, _ in segments:
        p = integrate(p, xi, T)
        out.append(p)
    return out


class Planner:
    name = "base"

    def setup(self, world, task, known_graph, rng=None):
        self.world, self.task, self.known = world, task, known_graph
        self.rng = rng
        self.t = 0.0

    def update(self, known_graph, t, hypotheses=None):
        self.known = known_graph
        self.t = t

    def payload_placement(self):
        return None

    def reconfiguration(self):
        return []

    def plan(self, pose, t):
        raise NotImplementedError

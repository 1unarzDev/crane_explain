"""2D occupancy world: procedural maps (rooms + doorways, corridors with turns, clutter, single doorway, mixed).

Grid resolution 0.05 m; cell (i, j) covers x in [i r, (i+1) r), y in [j r, (j+1) r). Outside the map is
occupied. Everything is deterministic given (family, seed)."""
import heapq
import math

import numpy as np

RES = 0.05
FAMILIES = ("rooms", "corridor", "clutter", "doorway", "mixed")


class World:
    def __init__(self, occ, res=RES, name="", meta=None):
        self.occ = occ.astype(bool)          # shape (nx, ny)
        self.res = res
        self.name = name
        self.meta = dict(meta or {})
        self.nx, self.ny = occ.shape
        self._infl = {}
        self._dist = None

    @property
    def size(self):
        return self.nx * self.res, self.ny * self.res

    def copy(self):
        w = World(self.occ.copy(), self.res, self.name, self.meta)
        if getattr(self, "truth_world", None) is not None:
            w.truth_world = self.truth_world      # keep the pinned physical world (truth.step reads it)
        return w

    def cells(self, pts):
        ij = np.floor(np.asarray(pts) / self.res).astype(np.int64)
        return ij

    def occupied_pts(self, pts, grid=None):
        """Vectorised: True where a world point is in an occupied (or out-of-map) cell."""
        grid = self.occ if grid is None else grid
        ij = self.cells(pts)
        out = (ij[:, 0] < 0) | (ij[:, 1] < 0) | (ij[:, 0] >= self.nx) | (ij[:, 1] >= self.ny)
        res = out.copy()
        ok = ~out
        res[ok] = grid[ij[ok, 0], ij[ok, 1]]
        return res

    def distance(self):
        """Euclidean-ish distance (m) from each cell centre to the nearest occupied cell (two-pass chamfer 3-4)."""
        if self._dist is not None:
            return self._dist
        INF = 10 ** 9
        d = np.where(self.occ, 0, INF).astype(np.int64)
        nx, ny = d.shape
        p = np.full((nx + 2, ny + 2), 0, np.int64)  # border = occupied
        p[1:-1, 1:-1] = d
        for i in range(1, nx + 1):
            row = p[i]
            prev = p[i - 1]
            row[1:-1] = np.minimum.reduce([row[1:-1], prev[1:-1] + 3, prev[:-2] + 4, prev[2:] + 4])
            for j in range(1, ny + 1):
                if row[j - 1] + 3 < row[j]:
                    row[j] = row[j - 1] + 3
        for i in range(nx, 0, -1):
            row = p[i]
            nxt = p[i + 1]
            row[1:-1] = np.minimum.reduce([row[1:-1], nxt[1:-1] + 3, nxt[:-2] + 4, nxt[2:] + 4])
            for j in range(ny, 0, -1):
                if row[j + 1] + 3 < row[j]:
                    row[j] = row[j + 1] + 3
        self._dist = p[1:-1, 1:-1] * (self.res / 3.0)
        return self._dist

    def inflated(self, margin):
        key = round(margin, 4)
        if key not in self._infl:
            if margin <= 0:
                self._infl[key] = self.occ
            else:
                self._infl[key] = self.distance() < margin + 1e-9
        return self._infl[key]

    def free_xy(self, x, y, clearance=0.0):
        i, j = int(x / self.res), int(y / self.res)
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            return False
        return self.distance()[i, j] > clearance

    def dijkstra_from(self, goals_xy, clearance):
        """8-connected cost-to-go (m) from a set of goal points on cells with distance > clearance."""
        free = self.distance() > clearance
        INF = math.inf
        D = np.full(self.occ.shape, INF)
        h = []
        for gx, gy in goals_xy:
            i, j = int(gx / self.res), int(gy / self.res)
            if 0 <= i < self.nx and 0 <= j < self.ny:
                D[i, j] = 0.0
                heapq.heappush(h, (0.0, i, j))
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
        nx, ny = self.nx, self.ny
        while h:
            d, i, j = heapq.heappop(h)
            if d > D[i, j]:
                continue
            for di, dj, c in steps:
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and free[a, b]:
                    nd = d + c * self.res
                    if nd < D[a, b]:
                        D[a, b] = nd
                        heapq.heappush(h, (nd, a, b))
        return D


def _blank(W, H, res=RES):
    occ = np.zeros((int(round(W / res)), int(round(H / res))), bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return occ


def _box(occ, x0, x1, y0, y1, val=True, res=RES):
    i0, i1 = max(0, int(math.floor(x0 / res))), min(occ.shape[0], int(math.ceil(x1 / res)))
    j0, j1 = max(0, int(math.floor(y0 / res))), min(occ.shape[1], int(math.ceil(y1 / res)))
    occ[i0:i1, j0:j1] = val


def _wall_x(occ, x, y0, y1, doors, t=0.1):
    """Vertical wall at x from y0..y1 with door intervals [(yc, w)]."""
    _box(occ, x - t / 2, x + t / 2, y0, y1)
    for yc, w in doors:
        _box(occ, x - t / 2 - 0.01, x + t / 2 + 0.01, yc - w / 2, yc + w / 2, False)


def _clutter(rng, occ, k, W, H, keep_out=(), smin=0.25, smax=0.7):
    for _ in range(k):
        for _try in range(50):
            sx, sy = rng.uniform(smin, smax, 2)
            x, y = rng.uniform(0.6, W - 0.6 - sx), rng.uniform(0.6, H - 0.6 - sy)
            if any(abs(x + sx / 2 - kx) < kr + sx and abs(y + sy / 2 - ky) < kr + sy for kx, ky, kr in keep_out):
                continue
            _box(occ, x, x + sx, y, y + sy)
            break


def make_world(family, seed):
    rng = np.random.default_rng(seed)
    W, H = 10.0, 8.0
    occ = _blank(W, H)
    meta = dict(family=family, seed=seed, doors=[])
    if family in ("rooms", "mixed"):
        for x in (W / 3, 2 * W / 3):
            w = float(rng.uniform(0.7, 1.5))
            yc = float(rng.uniform(0.3 + w / 2 + 0.6, H - 0.3 - w / 2 - 0.6))
            _wall_x(occ, x, 0, H, [(yc, w)])
            meta["doors"].append(dict(x=x, y=yc, w=w))
        meta["regions"] = [(0.3, W / 3 - 0.3), (2 * W / 3 + 0.3, W - 0.3)]
        if family == "mixed":
            keep = [(d["x"], d["y"], 1.0) for d in meta["doors"]]
            _clutter(rng, occ, 8, W, H, keep)
    elif family == "corridor":
        occ[:, :] = True
        w1, w2, w3 = rng.uniform(1.0, 1.8, 3)
        ya = float(rng.uniform(1.0, 2.2))
        yb = float(rng.uniform(H - 2.2, H - 1.0))
        xa = float(rng.uniform(3.5, 6.5))
        _box(occ, 0.3, xa + w2 / 2, ya - w1 / 2, ya + w1 / 2, False)          # bottom leg
        _box(occ, xa - w2 / 2, xa + w2 / 2, ya - w1 / 2, yb + w3 / 2, False)  # vertical leg (turns)
        _box(occ, xa - w2 / 2, W - 0.3, yb - w3 / 2, yb + w3 / 2, False)      # top leg
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
        meta.update(widths=[float(w1), float(w2), float(w3)], regions=[(0.3, xa - w2 / 2), (xa + w2 / 2, W - 0.3)],
                    legs=dict(ya=ya, yb=yb, xa=xa))
    elif family == "clutter":
        _clutter(rng, occ, 16, W, H)
        meta["regions"] = [(0.3, 3.5), (6.5, W - 0.3)]
    elif family == "doorway":
        w = float(rng.uniform(0.6, 1.4))
        yc = float(rng.uniform(1.5, H - 1.5))
        _wall_x(occ, W / 2, 0, H, [(yc, w)])
        meta["doors"].append(dict(x=W / 2, y=yc, w=w))
        _clutter(rng, occ, 4, W, H, [(W / 2, yc, 1.2)])
        meta["regions"] = [(0.3, W / 2 - 0.4), (W / 2 + 0.4, W - 0.3)]
    else:
        raise ValueError(family)
    return World(occ, RES, f"{family}_{seed}", meta)


def wall_stations(world, rng, k=8, standoff=0.12):
    """Docking stations: points `standoff` m in front of a wall, with the inward wall normal."""
    out = []
    occ = world.occ
    for _ in range(4000):
        i, j = int(rng.integers(2, world.nx - 2)), int(rng.integers(2, world.ny - 2))
        if occ[i, j]:
            continue
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if occ[a, b]:
                # wall face between (i,j) and (a,b); inward normal = -(di,dj)
                x = (i + 0.5 + 0.5 * di) * world.res - di * standoff
                y = (j + 0.5 + 0.5 * dj) * world.res - dj * standoff
                # require a straight wall segment of 0.5 m and free space 1 m in front
                seg_ok = True
                for s in np.linspace(-0.25, 0.25, 11):
                    wx = (a + 0.5) * world.res + (s if dj else 0)
                    wy = (b + 0.5) * world.res + (s if di else 0)
                    fx, fy = wx - di * 0.06, wy - dj * 0.06
                    if not world.occupied_pts(np.array([[wx, wy]]))[0] or world.occupied_pts(np.array([[fx, fy]]))[0]:
                        seg_ok = False
                        break
                if not seg_ok or not world.free_xy(x - di * 0.8, y - dj * 0.8, 0.5):
                    continue
                out.append(dict(x=float(x), y=float(y), normal=(-di, -dj)))
                break
        if len(out) >= k:
            break
    return out

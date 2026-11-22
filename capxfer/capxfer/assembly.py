"""Assembly geometry for square mecanum modules that dock on any face.

Frames: x forward, y left, z up (ROS FLU). A module sits on an integer grid cell (i, j) of the assembly with yaw k*90 deg.
Each module has four wheels at (+-lx, +-ly). The contact-roller axis of a corner is given as seen from above with the
module front up: "/" -> axis (1, -1)/sqrt2, "\\" -> axis (1, 1)/sqrt2 (x up, y left in that view).

A mecanum wheel transmits traction only along its contact-roller axis r; the perpendicular direction f is free (the
roller rolls), resisted only by roller-bearing drag. A roller-locked wheel behaves like a conventional wheel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math

import numpy as np
import yaml

CORNERS = ("fl", "fr", "rl", "rr")
_SIGN = {"fl": (1, 1), "fr": (1, -1), "rl": (-1, 1), "rr": (-1, -1)}
_AXIS = {"/": (1.0, -1.0), "\\": (1.0, 1.0)}


def load_params(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


@dataclass
class Assembly:
    name: str
    modules: list            # [(module_id, i, j, yaw_k)]
    wheel_pos: np.ndarray    # (n, 2) assembly frame, origin at geometric centre of module centres
    w_hat: np.ndarray        # (n, 2) rolling direction
    r_hat: np.ndarray        # (n, 2) contact-roller axis (traction direction)
    wheel_module: np.ndarray  # (n,) index into modules
    wheel_corner: list       # (n,) corner names
    module_mass: np.ndarray  # (m,)
    module_pos: np.ndarray   # (m, 2)
    lidar: np.ndarray        # (m,) bool
    side: float
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.wheel_pos)

    @property
    def mass(self):
        return float(self.module_mass.sum())

    def cg(self):
        return (self.module_mass[:, None] * self.module_pos).sum(0) / self.mass

    def izz(self, scale=1.0):
        s = self.side
        plate = self.module_mass * (2 * s * s) / 12.0 * scale
        d = self.module_pos - self.cg()
        return float((plate + self.module_mass * (d ** 2).sum(1)).sum())

    def char_length(self):
        """Lc: yaw-to-linear scale used to weight twist errors (max wheel radius from the CG)."""
        return float(np.max(np.linalg.norm(self.wheel_pos - self.cg(), axis=1)))

    def ik(self, xi, radius):
        """Ideal rolling IK: wheel speeds (rad/s) for body twist xi=(vx, vy, wz) about the assembly origin."""
        xi = np.asarray(xi, float)
        vc = np.stack([xi[..., 0:1] - xi[..., 2:3] * self.wheel_pos[:, 1],
                       xi[..., 1:2] + xi[..., 2:3] * self.wheel_pos[:, 0]], -1)
        return (vc * self.r_hat).sum(-1) / (radius * (self.w_hat * self.r_hat).sum(-1))

    def has_lidar(self):
        return bool(self.lidar.any())


def build(name, modules, params, module_ids_lidar=None):
    """modules: list of (module_id, i, j, yaw_k)."""
    mp = params["module"]
    side = float(mp["side_x_m"])
    lx, ly = float(mp["half_wheelbase_m"]), float(mp["half_track_m"])
    pattern = params["wheel"]["roller_pattern"]
    lidar_ids = set(module_ids_lidar if module_ids_lidar is not None else mp.get("lidar_modules", []))
    centres = np.array([[i * side, j * side] for _, i, j, _ in modules], float)
    origin = centres.mean(0)
    pos, wh, rh, wm, wc = [], [], [], [], []
    masses, lid = [], []
    for k, (mid, i, j, yk) in enumerate(modules):
        c = np.array([i * side, j * side]) - origin
        th = yk * math.pi / 2
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        for corner in CORNERS:
            sx, sy = _SIGN[corner]
            p_local = np.array([sx * lx, sy * ly])
            a = np.array(_AXIS[pattern[corner]]) / math.sqrt(2)
            pos.append(c + R @ p_local)
            wh.append(R @ np.array([1.0, 0.0]))
            rh.append(R @ a)
            wm.append(k)
            wc.append(f"{mid}:{corner}")
        masses.append(float(mp["mass_kg"].get(mid, np.mean(list(mp["mass_kg"].values())))))
        lid.append(mid in lidar_ids)
    return Assembly(name=name, modules=list(modules), wheel_pos=np.array(pos), w_hat=np.array(wh), r_hat=np.array(rh),
                    wheel_module=np.array(wm), wheel_corner=wc, module_mass=np.array(masses),
                    module_pos=centres - origin, lidar=np.array(lid), side=side)


# The four physical assembly types (module ids are placeholders until the team fixes them; every multi-module type
# contains a lidar module so evidence rung E2 exists).
PHYSICAL_TYPES = {
    "single": [("M1", 0, 0, 0)],
    "pair": [("M1", 0, 0, 0), ("M3", 1, 0, 0)],
    "L3": [("M2", 0, 0, 0), ("M3", 1, 0, 0), ("M4", 0, 1, 0)],
    "square4": [("M1", 0, 0, 0), ("M2", 1, 0, 0), ("M3", 0, 1, 0), ("M4", 1, 1, 0)],
}


def physical(name, params):
    return build(name, PHYSICAL_TYPES[name], params)


def _polyominoes(k):
    """Free polyomino cell sets of size k (canonical, translation-normalised; rotations/reflections kept distinct
    because the assembly is placed in a fixed body frame)."""
    shapes = {((0, 0),)}
    for _ in range(k - 1):
        new = set()
        for s in shapes:
            cells = set(s)
            for (x, y) in s:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    c = (x + dx, y + dy)
                    if c in cells:
                        continue
                    t = cells | {c}
                    mx, my = min(a for a, _ in t), min(b for _, b in t)
                    new.add(tuple(sorted((a - mx, b - my) for a, b in t)))
        shapes = new
    return sorted(shapes)


def shape_family(cells):
    """Family label used to hold out whole shape families in simulation: (size, bounding box sorted)."""
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    bb = tuple(sorted((max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)))
    return f"k{len(cells)}_bb{bb[0]}x{bb[1]}"


def random_assembly(rng, params, sizes=(1, 2, 3, 4, 5, 6), module_pool=("M1", "M2", "M3", "M4"), rotate_modules=True,
                    families=None):
    """A random docked assembly for simulation training. Module yaws are random multiples of 90 deg when
    rotate_modules (any face docks), which changes the roller pattern seen by the assembly."""
    for _ in range(1000):
        k = int(rng.choice(sizes))
        shapes = _polyominoes(k)
        cells = shapes[int(rng.integers(len(shapes)))]
        fam = shape_family(cells)
        if families is not None and fam not in families:
            continue
        mods = []
        for idx, (i, j) in enumerate(cells):
            mid = str(rng.choice(module_pool))
            yk = int(rng.integers(4)) if rotate_modules else 0
            mods.append((mid, i, j, yk))
        a = build(f"rand_{fam}", mods, params)
        a.meta["family"] = fam
        return a
    raise RuntimeError("no assembly matched the requested families")


def all_families(sizes=(1, 2, 3, 4, 5, 6)):
    return sorted({shape_family(c) for k in sizes for c in _polyominoes(k)})

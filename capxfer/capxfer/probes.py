"""The probe library (docs/TEAM_T0_HANDOFF.md §3). Identical tables drive the simulator and the field runner."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RAMP, REST = 0.3, 0.4
V_NOM, V_DIAG, W_NOM, V_MAX = 0.30, 0.21, 1.0, 0.45
WHEEL_SPEED_CAP = 0.45  # m/s at the farthest wheel during yaw primitives


@dataclass(frozen=True)
class Primitive:
    pid: str          # P1..P6x/P6y
    twist: tuple      # outbound twist (vx, vy, wz); the return leg is the negative
    hold: float


LIBRARY = (
    Primitive("P1", (V_NOM, 0.0, 0.0), 1.0),
    Primitive("P2", (0.0, V_NOM, 0.0), 1.0),
    Primitive("P3", (V_DIAG, V_DIAG, 0.0), 1.0),
    Primitive("P4", (V_DIAG, -V_DIAG, 0.0), 1.0),
    Primitive("P5", (0.0, 0.0, W_NOM), 1.0),
    Primitive("P6x", (V_MAX, 0.0, 0.0), 1.2),
    Primitive("P6y", (0.0, V_MAX, 0.0), 1.2),
)
PIDS = tuple(p.pid for p in LIBRARY)
SETTLE = 0.5


def leg_duration(p):
    return RAMP + p.hold + RAMP + REST


@dataclass
class Segment:
    pid: str
    leg: int            # +1 outbound, -1 return
    t0: float           # leg start
    hold0: float        # start of hold window
    hold1: float        # end of hold window
    t1: float           # leg end (after rest)
    twist: np.ndarray   # commanded hold twist


class Schedule:
    """Piecewise-linear twist command over a full trial. `scale_wz` limits yaw for large assemblies."""

    def __init__(self, char_length, pids=PIDS):
        self.segments = []
        t = SETTLE
        wz_scale = min(1.0, WHEEL_SPEED_CAP / max(char_length * W_NOM, 1e-9))
        for p in LIBRARY:
            if p.pid not in pids:
                continue
            tw = np.array(p.twist, float)
            tw[2] *= wz_scale
            for leg in (+1, -1):
                seg = Segment(p.pid, leg, t, t + RAMP, t + RAMP + p.hold, t + leg_duration(p), leg * tw)
                self.segments.append(seg)
                t = seg.t1
        self.T = t + 0.2

    def twist(self, t):
        """Vectorised command at times t (array) -> (len(t), 3)."""
        t = np.asarray(t, float)
        out = np.zeros(t.shape + (3,))
        for s in self.segments:
            up = np.clip((t - s.t0) / RAMP, 0, 1)
            down = np.clip((s.hold1 + RAMP - t) / RAMP, 0, 1)
            m = (t >= s.t0) & (t < s.hold1 + RAMP)
            out[m] += (np.minimum(up, down)[m, None]) * s.twist
        return out

    def windows(self):
        return [(s.pid, s.leg, s.hold0 + 0.2, s.hold1 - 0.2, s.twist) for s in self.segments]


# Capability target: one ratio per (primitive, leg) = 14 outputs.
TARGET_KEYS = tuple(f"{p.pid}{'+' if leg > 0 else '-'}" for p in LIBRARY for leg in (+1, -1))

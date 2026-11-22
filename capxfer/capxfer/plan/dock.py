"""Sideways-docking scenario, capability-aware lattice planner and sim executor for capxfer assemblies (contract C6).

Scene (adapted from Fleet capplan m6_nav.shelf_direct, commit bf4fdc53, re-dimensioned for our square 0.32 m modules):
open floor in front of a shelf face at y = y_face with one bay (slot) opening in +y. Goal: the bay centre with the START
heading (0). Two maneuver families reach it:
  * sideways: strafe +y straight into the bay (short: ~1.0-1.4 m);
  * yaw + forward: rotate +90 deg on the floor, drive forward into the bay, rotate back -90 deg inside the bay. The
    "wide" bay (width = depth = footprint diameter + 2 x 0.08 m) leaves room for that rotation; the "narrow" bay
    (width = footprint x-length + 2 x 0.06 m) admits the sideways entry only.
Success (pre-specifiable): final TRUE pose within POS_TOL = 0.05 m and HEAD_TOL = 0.087 rad (5 deg) of the goal, no
contact of the true footprint with the shelf face or bay walls at any recorded instant (100 Hz), within T_MAX = 90 s
of commanded motion.

Planner: the vendored Fleet SE(2) lattice (capxfer/vendor/capplan/planners/base.py: weighted hybrid A*, 16 headings,
analytic SE(2) shot near the goal) with motion primitives generated from a CapSet (capability.py): every primitive's
lattice twist is the PREDICTED realised twist `capset.expected(cmd)`; the command sent to the robot is recovered with
`capset.command_for`. Unusable directions produce no primitives; drift raises a primitive's cost.

Executor (`execute`): commands become a piecewise-linear twist schedule (Schedule duck type: `.T`, `.twist(t)`) run by
the unchanged `capxfer.sim.simulate`. Closed loop = stop-and-go replanning: execute a plan prefix (<= HORIZON_M of
predicted path or one rotation), come to rest, take a localisation fix (lidar scan matching against the arena walls;
modelled as Gaussian noise LOC_SIGMA), replan. Open loop = one plan, executed once. Each chunk starts the sim from rest
with the same domain and seed (so per-wheel mounting errors persist); the robot is at rest at every chunk boundary, so
chaining is exact up to the PI-integrator reset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch

from ..vendor.capplan.planners.base import Lattice, MotionModel, Prim, integrate, plan_poses
from ..vendor.capplan.world import World
from ..vendor.capplan._shim import rot, wrap
from . import capability as CAP

RES = 0.025
POS_TOL, HEAD_TOL = 0.05, math.radians(5.0)
EXEC_POS_TOL, EXEC_HEAD_TOL = 0.025, 0.04      # executor stops when its ESTIMATE is this close
T_MAX = 90.0
V_DOCK = 0.20                                  # commanded linear speed (m/s), below the 0.30 m/s probes
WHEEL_YAW_SPEED = 0.15                         # yaw rate chosen so the farthest wheel moves at <= 0.15 m/s
W_DOCK_MAX = 0.6
RAMP = 0.12                                    # command blend at every segment boundary (s), integral-preserving
REST = 0.6
HORIZON_M = 0.30
LOC_SIGMA = (0.01, 0.01, 0.01)                 # localisation fix noise (m, m, rad); ASSUMPTION (lidar scan matching)
MAX_ITERS = 40
MARGIN = 0.02                                  # planner footprint inflation (m)


# ============================================================================================== geometry
def footprint_points(asm, spacing=0.02):
    """Body-frame sample points on the boundary of the union of the module squares."""
    s = asm.side / 2
    sq = [np.array([[c[0] - s, c[1] - s], [c[0] + s, c[1] - s], [c[0] + s, c[1] + s], [c[0] - s, c[1] + s]])
          for c in asm.module_pos]
    pts = []
    for k, P in enumerate(sq):
        for i in range(4):
            a, b = P[i], P[(i + 1) % 4]
            n = max(2, int(math.ceil(np.linalg.norm(b - a) / spacing)) + 1)
            for t in np.linspace(0, 1, n)[:-1]:
                p = a + t * (b - a)
                inside = any(np.all(np.abs(p - asm.module_pos[j]) < s - 1e-6) for j in range(len(sq)) if j != k)
                if not inside:
                    pts.append(p)
    return np.array(pts)


def extents(asm):
    s = asm.side / 2
    x0, x1 = asm.module_pos[:, 0].min() - s, asm.module_pos[:, 0].max() + s
    y0, y1 = asm.module_pos[:, 1].min() - s, asm.module_pos[:, 1].max() + s
    return x0, x1, y0, y1


@dataclass
class Scene:
    world: World
    start: tuple
    goal: tuple
    floor: tuple            # free rectangle (x0, x1, y0, y1)
    bay: tuple              # free rectangle (x0, x1, y0, y1)
    variant: str
    meta: dict = field(default_factory=dict)

    def free(self, P):
        """Exact free-space test for world points P (k,2): inside the floor or the bay rectangle."""
        def inside(r):
            return (P[:, 0] >= r[0]) & (P[:, 0] <= r[1]) & (P[:, 1] >= r[2]) & (P[:, 1] <= r[3])
        return inside(self.floor) | inside(self.bay)


def make_scene(asm, seed, variant="wide"):
    """Randomised docking scene for assembly `asm` (seeded)."""
    rng = np.random.default_rng([seed, 7301])
    pts = footprint_points(asm)
    Rmax = float(np.hypot(pts[:, 0], pts[:, 1]).max())
    x0, x1, y0, y1 = extents(asm)
    Lx, Ly = x1 - x0, y1 - y0
    if variant == "wide":
        ws = depth = 2 * Rmax + 0.16
    elif variant == "narrow":
        ws, depth = Lx + 0.12, Ly + 0.12
    else:
        raise ValueError(variant)
    WX, y_face = 4.0, 2.6
    H = y_face + depth + 0.2
    occ = np.ones((int(round(WX / RES)), int(round(H / RES))), bool)
    xs = float(rng.uniform(1.6, 2.4))
    floor = (0.0, WX, 0.0, y_face)
    bay = (xs - ws / 2, xs + ws / 2, y_face - 1e-6, y_face + depth)

    def carve(r):
        i0, i1 = int(math.ceil(r[0] / RES - 1e-9)), int(math.floor(r[1] / RES + 1e-9))
        j0, j1 = int(math.ceil(r[2] / RES - 1e-9)), int(math.floor(r[3] / RES + 1e-9))
        occ[max(i0, 0):i1, max(j0, 0):j1] = False      # a cell is free only if it lies fully inside the rectangle
    carve((0.0 + RES, WX - RES, 0.0 + RES, y_face))     # keep a 1-cell border occupied (arena edge)
    carve(bay)
    w = World(occ, RES, f"dock_{variant}_{seed}", dict(slot=dict(x=xs, w=ws, depth=depth, y_face=y_face)))
    goal_y = y_face + (depth / 2 if variant == "wide" else depth - 0.06 - y1)
    goal = (xs - 0.5 * (x0 + x1), float(goal_y), 0.0)
    start = (float(xs + rng.uniform(-0.10, 0.10)), float(y_face - y1 - rng.uniform(0.7, 1.1)),
             float(rng.uniform(-0.05, 0.05)))
    return Scene(w, start, goal, floor, bay, variant,
                 dict(seed=seed, ws=ws, depth=depth, y_face=y_face, Rmax=Rmax, Lx=Lx, Ly=Ly))


# ============================================================================================== planner
def yaw_rate(asm):
    return min(W_DOCK_MAX, WHEEL_YAW_SPEED / max(asm.char_length(), 1e-6))


def motion_model(capset, asm, bins=16, step=0.3, mid=0.10, fine=0.04, drift_cost=3.0, arc_radius=0.6):
    pts = footprint_points(asm, 0.04)
    dth = 2 * math.pi / bins
    v, w = V_DOCK, yaw_rate(asm)
    prims = []

    def add(cmd, length=None, nbins=None):
        cmd = np.asarray(cmd, float)
        if not capset.allowed(cmd):
            return
        exp = capset.expected(cmd)
        mult = 1.0 + drift_cost * capset.drift(cmd)
        if nbins is not None:
            if abs(exp[2]) < 1e-6:
                return
            T = nbins * dth / abs(exp[2])
        else:
            ve = math.hypot(exp[0], exp[1])
            if ve < 1e-6:
                return
            T = length / ve
        prims.append(Prim(exp, T, "roll", mult))

    for _, d in CAP.PLANAR:
        for L in (step, mid, fine):
            add((v * d[0], v * d[1], 0.0), length=L)
    for s in (1, -1):
        for nb in (1, 2, 4):
            add((0.0, 0.0, s * w), nbins=nb)
    # forward / backward arcs (1 heading bin over ~arc_radius * dth of path)
    for sx in (1, -1):
        for sw in (1, -1):
            kx = CAP.KIDX["P1+" if sx > 0 else "P1-"]
            kw = CAP.KIDX["P5+" if sw > 0 else "P5-"]
            rx, rw = capset.mid[kx], capset.mid[kw]
            if rx <= 1e-6 or rw <= 1e-6:
                continue
            w_exp = rx * v / arc_radius
            add((sx * v, 0.0, sw * w_exp / rw), nbins=1)

    def allowed(u):
        return capset.allowed(capset.command_for(u))

    def ray(u):
        c = capset.command_for(u)
        lin, ang = math.hypot(c[0], c[1]), abs(c[2])
        s = min(v / lin if lin > 1e-9 else math.inf, w / ang if ang > 1e-9 else math.inf)
        return 0.0 if not math.isfinite(s) else s

    vh = max([math.hypot(p.xi[0], p.xi[1]) for p in prims] + [1e-3])
    wh = max([abs(p.xi[2]) for p in prims] + [1e-3])
    r_in = float(np.hypot(pts[:, 0], pts[:, 1]).min())
    return MotionModel(prims, pts, MARGIN, bins, vh, wh, r_in, allowed=allowed, ray=ray, name=capset.name)


@dataclass
class DockPlan:
    status: str                       # ok | infeasible | fail
    exp: list                         # [(xi_expected, T)]
    cmd: list                         # [(xi_command, T)]
    reason: str = ""
    expansions: int = 0
    compute_s: float = 0.0

    @property
    def duration(self):
        return float(sum(T for _, T in self.cmd))

    def route(self):
        lat = sum(abs(c[1]) * T for c, T in self.cmd)
        yaw = sum(abs(c[2]) * T for c, T in self.cmd)
        return "sideways" if lat >= 0.3 and yaw < 0.8 else ("yaw+forward" if yaw >= 0.8 else "other")


def plan_dock(capset, asm, scene, start=None, budget=12000, wall_s=120.0, tols=((0.03, 0.05), (0.08, 0.12))):
    """Plan from `start` (default scene.start) to scene.goal. Tries progressively looser search tolerances (the closed
    loop replans from the reached pose)."""
    start = tuple(scene.start if start is None else start)
    model = motion_model(capset, asm)
    lat = Lattice(scene.world, model, cell=0.05, eps=1.5, budget=budget, wall_s=wall_s)
    pl = None
    for pt, ht in tols:
        pl = lat.search(start, [tuple(scene.goal)], pos_tol=pt, head_tol=ht)
        if pl.status == "ok":
            break
        if pl.status == "infeasible":
            break
    exp = [(np.asarray(xi, float), float(T)) for xi, T, _ in pl.segments]
    # trim the residual heading (lattice headings are binned) with an in-place rotation if yaw is usable
    if pl.status == "ok":
        end = plan_poses(start, pl.segments)[-1]
        dth = wrap(scene.goal[2] - end[2])
        if abs(dth) > 0.01:
            w = yaw_rate(asm)
            cmd = np.array([0.0, 0.0, math.copysign(w, dth)])
            if capset.allowed(cmd):
                e = capset.expected(cmd)
                if abs(e[2]) > 1e-6:
                    exp.append((e, abs(dth) / abs(e[2])))
    cmd = [(capset.command_for(xi), T) for xi, T in exp]
    return DockPlan(pl.status, exp, cmd, pl.reason, pl.expansions, pl.compute_s)


# ============================================================================================== command adapter
class CommandSchedule:
    """Schedule duck type for capxfer.sim.simulate: piecewise-constant body twists [(xi, T)] blended linearly over RAMP
    at every boundary (centred on it, so the integral of each segment's command is preserved), with a lead-in of
    RAMP/2 and a final rest of REST."""

    def __init__(self, cmds, ramp=RAMP, rest=REST):
        self.cmds = [(np.asarray(c, float), float(T)) for c, T in cmds]
        self.ramp = ramp
        t = ramp / 2
        self.knots = [t]
        for _, T in self.cmds:
            t += T
            self.knots.append(t)
        self.T = t + ramp / 2 + rest
        self.segments = []

    def twist(self, t):
        t = np.asarray(t, float)
        out = np.zeros(t.shape + (3,))
        if not self.cmds:
            return out
        vals = [np.zeros(3)] + [c for c, _ in self.cmds] + [np.zeros(3)]
        # step function value
        idx = np.searchsorted(np.asarray(self.knots), t, side="right")   # 0 before the first knot
        V = np.stack(vals)
        out = V[np.clip(idx, 0, len(vals) - 1)]
        # linear blends around every knot
        h = self.ramp / 2
        for k, tk in enumerate(self.knots):
            m = (t > tk - h) & (t < tk + h)
            if m.any():
                a = ((t[m] - (tk - h)) / self.ramp)[:, None]
                out[m] = (1 - a) * vals[k] + a * vals[k + 1]
        return out

    def windows(self):
        return []


def run_chunk(asm, params, dom, mode, g, cmds, seed=0, device="cpu", dt=0.005):
    from .. import sim
    sched = CommandSchedule(cmds)
    d = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in dom.items()}
    R = sim.simulate(asm, params, d, mode, g, schedule=sched, device=device, seed=seed, dt=dt)
    return R["pose"][0].cpu().numpy().astype(float), float(sched.T)


def compose(p0, local):
    """World poses from a start pose p0 and local poses (k,3) that start at the origin."""
    c, s = math.cos(p0[2]), math.sin(p0[2])
    out = np.empty_like(local)
    out[:, 0] = p0[0] + c * local[:, 0] - s * local[:, 1]
    out[:, 1] = p0[1] + s * local[:, 0] + c * local[:, 1]
    out[:, 2] = p0[2] + local[:, 2]
    return out


def collides(scene, pts, poses, every=2):
    for p in poses[::every]:
        W = pts @ rot(p[2]).T + p[:2]
        if not scene.free(W).all():
            return True
    return False


def _prefix(plan, horizon=HORIZON_M):
    """Plan prefix for one closed-loop chunk: segments until HORIZON_M of predicted path; a rotation ends a chunk."""
    out, L = [], 0.0
    for (xe, T), (xc, _) in zip(plan.exp, plan.cmd):
        v = math.hypot(xe[0], xe[1])
        if v > 1e-9 and L + v * T > horizon:          # split a long segment (e.g. the analytic shot)
            out.append((xc, max((horizon - L) / v, 0.05)))
            break
        out.append((xc, T))
        L += v * T
        if L >= horizon - 1e-9 or abs(xe[2]) * T > 0.3:
            break
    return out


def execute(capset, asm, params, scene, dom, mode, g, loop="closed", seed=0, rng=None, device="cpu"):
    """Plan + execute in capxfer.sim. Returns a result dict (TRUE outcome; the robot sees only localisation fixes)."""
    rng = rng or np.random.default_rng(seed)
    pts = footprint_points(asm, 0.02)
    true = np.array(scene.start, float)
    sig = np.asarray(LOC_SIGMA)
    est = true + rng.normal(0, sig)
    t_used, iters, collided, status, routes, replans = 0.0, 0, False, "timeout", [], 0
    while iters < MAX_ITERS and t_used < T_MAX:
        ge = np.array(scene.goal)
        if math.hypot(*(est[:2] - ge[:2])) <= EXEC_POS_TOL and abs(wrap(est[2] - ge[2])) <= EXEC_HEAD_TOL:
            status = "done"
            break
        plan = plan_dock(capset, asm, scene, start=tuple(est))
        replans += 1
        if plan.status != "ok":
            status = "infeasible" if plan.status == "infeasible" else "plan_fail"
            break
        if not plan.cmd:
            status = "done"
            break
        routes.append(plan.route())
        cmds = plan.cmd if loop == "open" else _prefix(plan)
        local, T = run_chunk(asm, params, dom, mode, g, cmds, seed=seed, device=device)
        poses = compose(true, local)
        t_used += T
        iters += 1
        if collides(scene, pts, poses):
            collided = True
            true = poses[-1]
            status = "collision"
            break
        true = poses[-1]
        est = true + rng.normal(0, sig)
        if loop == "open":
            status = "done"
            break
    g_ = np.array(scene.goal)
    pe = float(math.hypot(*(true[:2] - g_[:2])))
    he = float(abs(wrap(true[2] - g_[2])))
    ok = (not collided) and pe <= POS_TOL and he <= HEAD_TOL and t_used <= T_MAX
    return dict(success=bool(ok), pos_err=pe, head_err=he, collision=collided, status=status, time=t_used,
                iters=iters, replans=replans, route=(routes[0] if routes else "none"),
                final=true.tolist(), capset=capset.name)


# ============================================================================================== arms
ARMS = ("proposed", "omni", "oracle", "analytic_bayes")


def fault_spec(asm, kind, wheel=0, g=0.5, rng=None):
    """(mode (1,n) long, g (1,n) float) for F0 healthy, F1 weak, F2 coast, F3 brake, F4 roller-locked,
    F4x2d: two roller-locked wheels on diagonal corners of module 0 (fl + rr)."""
    from .. import sim
    n = asm.n
    mode = torch.zeros(1, n, dtype=torch.long)
    gg = torch.ones(1, n)
    k = {"F0": None, "F1": sim.WEAK, "F2": sim.COAST, "F3": sim.BRAKE, "F4": sim.ROLLER}.get(kind, "x")
    if kind == "F4x2d":
        m0 = [i for i in range(n) if asm.wheel_module[i] == asm.wheel_module[wheel]]
        corners = {asm.wheel_corner[i].split(":")[1]: i for i in m0}
        mode[0, corners["fl"]] = sim.ROLLER
        mode[0, corners["rr"]] = sim.ROLLER
    elif k == "x":
        raise ValueError(kind)
    elif k is not None:
        mode[0, wheel] = k
        if k == sim.WEAK:
            gg[0, wheel] = g
    return mode, gg


def capset_for_arm(arm, asm, truth=None, stub=None, nominal=None, **kw):
    """truth: (rho, drift) (14,) measured on the true health + domain (oracle); stub: (lo, hi[, drift_hi]) (proposed);
    nominal: (rho, drift) measured on the true health under NOMINAL domain parameters (analytic_bayes stub)."""
    if arm == "omni":
        return CAP.omni(asm, **kw)
    if arm == "oracle":
        return CAP.oracle_capset(truth[0], truth[1], asm, **kw)
    if arm == "proposed":
        return CAP.from_intervals(stub[0], stub[1], asm, drift=(stub[2] if len(stub) > 2 else None), name="proposed", **kw)
    if arm == "analytic_bayes":
        return CAP.from_point(nominal[0], asm, nominal[1], name="analytic_bayes", **kw)
    raise ValueError(arm)


def nominal_domain(device="cpu"):
    """B=1 domain at the midpoint of every sim.DR_DEFAULT range (no CG offset, no mounting error): the uncalibrated
    nominal-parameter physics used by the analytic_bayes stub."""
    from .. import sim
    dr = {k: ((v[0] + v[1]) / 2,) * 2 for k, v in sim.DR_DEFAULT.items() if isinstance(v, tuple)}
    dr.update(cg_off=0.0, wheel_yaw_err=0.0)
    return sim.sample_domain(1, np.random.default_rng(0), dr, device=device)


def stub_drift_bound(drift_true, rng, width=0.03, sd=0.015, alpha=0.10):
    """STUB for a calibrated drift upper bound (the C4 contract has none yet; requested): true drift + |noise| + width;
    with probability alpha the bound falls below the truth (miscoverage)."""
    d = np.asarray(drift_true, float)
    b = d + np.abs(rng.normal(0, sd, d.shape)) + width
    miss = rng.uniform(size=d.shape) < alpha
    return np.where(miss, np.maximum(d - width, 0.0), b)


def stub_intervals(rho_true, rng, half_width=0.10, center_sd=0.04, alpha=0.10):
    """STUB for the conformal lower bounds (replace with conformal.capability CapCal intervals): intervals centred on
    the true ratio plus Gaussian error, half-width `half_width`; with probability alpha the centre is shifted by
    2 x half_width (a miscoverage), so per-direction coverage is ~1 - alpha."""
    rho_true = np.asarray(rho_true, float)
    c = rho_true + rng.normal(0, center_sd, rho_true.shape)
    miss = rng.uniform(size=rho_true.shape) < alpha
    c = c + miss * rng.choice([-1, 1], rho_true.shape) * 2 * half_width
    return c - half_width, c + half_width


def simulate_arm(arm, asm, fault, dom, params, scene, truth=None, stub=None, nominal=None, loop="closed", seed=0,
                 device="cpu", capset_kw=None):
    """One docking run of `arm` on assembly `asm` with `fault` = (mode, g) under domain `dom` (B=1 dict from
    sim.sample_domain). See capset_for_arm for truth / stub / nominal."""
    cs = capset_for_arm(arm, asm, truth, stub, nominal, **(capset_kw or {}))
    mode, g = fault
    r = execute(cs, asm, params, scene, dom, mode, g, loop=loop, seed=seed, rng=np.random.default_rng([seed, 99]),
                device=device)
    r["arm"] = arm
    r["capset_summary"] = cs.summary()
    return r


# ============================================================================================== robot executor format
def export_plan(plan, capset, scene, asm, rate_hz=50, ramp=RAMP, rest=REST):
    """JSON-serialisable plan for the ROS 2 cmd_vel executor (docs/DOCKING_PROTOCOL.md section 6). Includes the
    piecewise-constant command list and the same blended twist samples the simulator executes."""
    sched = CommandSchedule(plan.cmd, ramp, rest)
    t = np.arange(0.0, sched.T, 1.0 / rate_hz)
    tw = sched.twist(t)
    return dict(schema="capxfer.dock_plan/v1", arm=capset.name, assembly=asm.name, status=plan.status,
                route=plan.route() if plan.status == "ok" else "none",
                start=list(scene.start), goal=list(scene.goal), ramp_s=ramp, rest_s=rest, rate_hz=rate_hz,
                segments=[dict(vx=float(c[0]), vy=float(c[1]), wz=float(c[2]), T=float(T)) for c, T in plan.cmd],
                samples=np.round(tw, 5).tolist(), duration_s=float(sched.T),
                capset=capset.summary())

"""Component mass budget, CG and inertia for the square mecanum module variants (WS-H).

Run from capxfer/:  .venv/bin/python assets/tools/module_budget.py  > assets/module_budget.json
Frame: module centre at floor level (base_footprint), x forward, y left, z up; metres, kg.
Every component carries a tag: datasheet (URL in docs/HARDWARE_ESTIMATES.md), derived (from STEP/geometry),
or estimate (engineering guess; range given). Nothing here is measured.
"""
import json, math, sys
import numpy as np

# ---- geometry (see docs/HARDWARE_ESTIMATES.md §3 for the derivation) ----
S = 0.320            # side (estimate; >= 0.312 from drivetrain stack)
R = 0.04805          # wheel outer radius from STEP (derived), unloaded
WW = 0.060           # wheel envelope width incl. bolt heads (STEP)
LX = 0.104           # half wheelbase: S/2 - wall 0.004 - clearance 0.004 - R
LY = 0.120           # half track: >= 0.1175 from motor stack; <= 0.122 for S=0.32
E_FACE = 0.012       # motor face plate -> wheel inner envelope (bracket + flange coupling), estimate
MOTOR_L = 0.0725     # datasheet: face plate to encoder cap
MOTOR_D = 0.037
SHAFT_OFF = 0.007    # 37D output shaft offset from motor axis (datasheet drawing, recalled: UNVERIFIED)
CLEAR = 0.020        # ground clearance of the chassis floor, estimate
TOP = 0.110          # chassis top deck height, estimate
WHEEL_M = 0.475      # derived from STEP volumes x densities (Al plates, PU rollers, steel hardware)
WHEEL_IAX = 3.96e-4  # derived from STEP, about the axle

motor_cy = LY - WW / 2 - E_FACE - MOTOR_L / 2

def box(m, sx, sy, sz):
    return np.diag([m * (sy**2 + sz**2), m * (sx**2 + sz**2), m * (sx**2 + sy**2)]) / 12.0

def cyl_y(m, r, l):          # cylinder with axis along y
    a = m * (3 * r * r + l * l) / 12.0
    return np.diag([a, m * r * r / 2, a])

def shell_sq(m, s, h):       # four thin walls, square, side s, height h (no top/bottom)
    mw = m / 4; I = np.zeros((3, 3))
    for k in range(4):
        th = k * math.pi / 2; c = np.array([math.cos(th), math.sin(th), 0]) * s / 2
        Il = box(mw, 0.003, s, h) if k % 2 == 0 else box(mw, s, 0.003, h)
        I += Il + mw * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
    return I

C = []   # (name, mass, (x,y,z), Icm 3x3, tag, note)
def add(name, m, p, I, tag, note=""):
    C.append((name, m, np.array(p, float), I, tag, note))

for sx in (1, -1):
    for sy in (1, -1):
        cn = ("f" if sx > 0 else "r") + ("l" if sy > 0 else "r")
        Iw = np.diag([0.5 * WHEEL_IAX + WHEEL_M * WW**2 / 12, WHEEL_IAX, 0.5 * WHEEL_IAX + WHEEL_M * WW**2 / 12])
        add(f"wheel_{cn}", WHEEL_M, (sx * LX, sy * LY, R), Iw, "derived", "STEP volumes; Al 2.70, PU 1.20, steel 7.85 g/cm3")
        add(f"motor_{cn}", 0.210, (sx * LX, sy * motor_cy, R + SHAFT_OFF), cyl_y(0.210, MOTOR_D / 2, MOTOR_L), "datasheet", "Pololu #4755 210 g")
        add(f"bracket_{cn}", 0.020, (sx * LX, sy * (LY - WW / 2 - E_FACE / 2), R + 0.01), box(0.02, 0.045, 0.004, 0.06), "estimate", "37D bracket + screws, 0.012-0.030")
        add(f"coupling_{cn}", 0.018, (sx * LX, sy * (LY - WW / 2 - 0.004), R), cyl_y(0.018, 0.012, 0.012), "estimate", "6 mm D-shaft flange coupling, 0.010-0.030")
add("battery", 0.355, (0, 0, CLEAR + 0.003 + 0.0124), box(0.355, 0.0425, 0.154, 0.0247), "datasheet", "Gens ace GEA503S60X6GT 355 g +-20 g, 154x42.5x24.7 mm (lies along y)")
add("driver_g2_front", 0.025, (0.05, 0, 0.078), box(0.025, 0.055, 0.07, 0.012), "estimate", "Pololu #2516 shield; weight not published")
add("driver_g2_rear", 0.025, (-0.05, 0, 0.078), box(0.025, 0.055, 0.07, 0.012), "estimate", "")
add("pico2h", 0.004, (0.0, 0.05, 0.075), box(0.004, 0.021, 0.051, 0.004), "estimate", "51x21 mm board (datasheet), mass estimate")
add("imu", 0.003, (0.0, 0.0, 0.075), box(0.003, 0.02, 0.02, 0.003), "estimate", "")
add("power_wiring", 0.190, (0, 0, 0.060), box(0.19, 0.2, 0.2, 0.04), "estimate", "5 V 5 A buck, switch, fuse, XT60, harness; 0.12-0.30")
add("fasteners_misc", 0.080, (0, 0, 0.065), box(0.08, 0.3, 0.3, 0.08), "estimate", "")
add("dock_hardware", 0.110, (0, 0, 0.060), shell_sq(0.11, S - 0.02, 0.03), "estimate", "8 magnets + 4 latches (servo/hook); 0.06-0.20")
# PETG chassis 0.90 kg: floor, top deck, outer walls, wheel-well walls/ribs
add("petg_floor", 0.25, (0, 0, CLEAR + 0.0015), box(0.25, S, S, 0.003), "estimate", "PETG 1.27 g/cm3, ~85 % fill")
add("petg_top", 0.25, (0, 0, TOP - 0.0015), box(0.25, S, S, 0.003), "estimate", "")
add("petg_walls", 0.22, (0, 0, (CLEAR + TOP) / 2), shell_sq(0.22, S, TOP - CLEAR), "estimate", "")
add("petg_wells_ribs", 0.19, (0, 0, 0.06), box(0.19, 0.26, 0.2, 0.08), "estimate", "")

SENS = {
  "lidar": [
    ("d435", 0.072, (S / 2 - 0.015, 0, 0.125), box(0.072, 0.025, 0.09, 0.025), "datasheet", "Intel D435 ~72 g (+-10 %), 90x25x25 mm"),
    ("rplidar_c1", 0.110, (-0.05, 0, TOP + 0.0207), box(0.11, 0.0556, 0.0556, 0.0413), "datasheet", "Slamtec C1 110 g, 55.6x55.6x41.3 mm"),
    ("opi5pro", 0.062, (-0.06, -0.06, 0.088), box(0.062, 0.089, 0.056, 0.016), "datasheet", "Orange Pi 5 Pro 62 g, 89x56 mm"),
    ("opi_cooling", 0.030, (-0.06, -0.06, 0.098), box(0.03, 0.04, 0.04, 0.01), "estimate", "heatsink + fan"),
    ("sensor_mounts_cables", 0.070, (0.03, 0, 0.115), box(0.07, 0.2, 0.1, 0.02), "estimate", "D435/lidar mounts + USB cables"),
  ],
  "cam": [
    ("ov9282", 0.025, (S / 2 - 0.012, 0, 0.115), box(0.025, 0.02, 0.038, 0.038), "estimate", "Arducam OV9282 board + M12 lens; exact module unconfirmed"),
    ("opi4pro", 0.058, (-0.06, -0.06, 0.088), box(0.058, 0.089, 0.056, 0.016), "datasheet", "Orange Pi 4 Pro 58 g, 89x56 mm"),
    ("opi_cooling", 0.030, (-0.06, -0.06, 0.098), box(0.03, 0.04, 0.04, 0.01), "estimate", ""),
    ("sensor_mounts_cables", 0.030, (0.08, 0, 0.11), box(0.03, 0.1, 0.05, 0.02), "estimate", ""),
  ],
}
LIFTER = ("lifter_arm_stowed", 0.50, (0.09, 0, 0.16), box(0.5, 0.12, 0.10, 0.10), "estimate",
          "design unknown: 1-2 servos + linkage + fork/gripper, 0.30-0.80 kg, stowed over the front half")

def summarize(parts):
    m = sum(p[1] for p in parts)
    cg = sum(p[1] * p[2] for p in parts) / m
    I = np.zeros((3, 3))
    for _, mi, pi, Ii, _, _ in parts:
        d = pi - cg; I += Ii + mi * (d @ d * np.eye(3) - np.outer(d, d))
    by_tag = {}
    for p in parts:
        by_tag[p[4]] = by_tag.get(p[4], 0) + p[1]
    return m, cg, I, by_tag

out = {"geometry": dict(side=S, wheel_radius_unloaded=R, wheel_width=WW, half_wheelbase=LX, half_track=LY,
                         motor_centre_abs_y=motor_cy, ground_clearance=CLEAR, top_deck=TOP,
                         motor_stack_min_half_track=WW / 2 + E_FACE + MOTOR_L + 0.003,
                         side_min=2 * (LY + WW / 2 + 0.008)), "variants": {}}
for sens in ("lidar", "cam"):
    for lift in (False, True):
        extra = SENS[sens] + ([LIFTER] if lift else [])
        parts = C + [(a, b, np.array(c, float), d, e, f) for a, b, c, d, e, f in extra]
        m, cg, I, bt = summarize(parts)
        izz_plate = m * 2 * S * S / 12
        name = f"{sens}{'_lifter' if lift else ''}"
        out["variants"][name] = dict(mass=m, cg=cg.tolist(), inertia_about_cg=I.tolist(), izz=I[2, 2],
                                     yaw_inertia_scale=I[2, 2] / izz_plate, mass_by_tag=bt,
                                     front_axle_load_frac=float(0.5 + cg[0] / (2 * LX)),
                                     components=[dict(name=p[0], mass=p[1], xyz=p[2].round(4).tolist(), tag=p[4], note=p[5]) for p in parts])
print(json.dumps(out, indent=1))

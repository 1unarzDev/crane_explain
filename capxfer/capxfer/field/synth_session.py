"""Synthetic FieldLog v2 session in the capxfer layout, generated from the simulator. Labelled SIMULATION by construction.

For testing the loader and the end-to-end pipeline only. Written through FieldLog's own writer (csvlog_v2.SessionV2, not
duplicated) with use_sim_time=True, a 'sim:' clock source, no --field and no vehicle id, plus explicit mark_simulation
reasons, so the label can never become FIELD (the verifier recomputes it). Nothing here creates FIELD data.

Messages follow the config streams (fieldlog_config): cmd_vel 50 Hz, per-module wheel_state 100 Hz, IMU 100 Hz (IMU-module
frame), rf2o odometry 10 Hz (lidar-module frame, 0.1 s latency), AprilTag truth 30 Hz (0.15 s latency), fault_state echoes,
assembly frames and trial markers as the probe runner publishes them. Receipt times include latency + jitter.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

from .. import assembly as A
from .. import sim as S
from ..probes import Schedule
from . import fieldlog_config as FC
from . import schedule as SCH
from .probe_runner import fault_messages, trial_plan

SIM_REASONS = ("capxfer synthetic session: capxfer.sim via capxfer/field/synth_session.py; never field data",)


def _import_fieldlog():
    if str(FC.FIELDLOG) not in sys.path:
        sys.path.insert(0, str(FC.FIELDLOG))
    import csvlog_v2  # noqa: E402
    return csvlog_v2


class SimClock:
    def __init__(self):
        import time
        self.mono0, self.wall0, self.t = time.monotonic_ns(), time.time_ns(), 0.0

    def now(self):
        return self.mono0 + int(round(self.t * 1e9)), self.wall0 + int(round(self.t * 1e9))

    def mono(self, t):
        return self.mono0 + int(round(t * 1e9))

    def wall(self, t):
        return self.wall0 + int(round(t * 1e9))


def _sim_faults(trials, asm):
    """Schedule trials -> sim mode/g (B,n), payload dict, mu_factor (B,)."""
    B, n = len(trials), asm.n
    mode = torch.zeros(B, n, dtype=torch.long)
    g = torch.ones(B, n)
    pm, px, py = torch.zeros(B), torch.zeros(B), torch.zeros(B)
    mu = torch.ones(B)
    code = {"weak": S.WEAK, "coast": S.COAST, "brake": S.BRAKE, "roller_locked": S.ROLLER}
    mods = [m for m, *_ in asm.modules]
    for b, tr in enumerate(trials):
        for f in tr["faults"]:
            if f["mode"] in code:
                w = asm.wheel_corner.index(f["wheel"])
                mode[b, w] = code[f["mode"]]
                if f["mode"] == "weak":
                    g[b, w] = float(f["g"])
            elif f["mode"] == "payload":
                k = mods.index(f["module"])
                pm[b] = float(f["mass_kg"])
                px[b], py[b] = float(asm.module_pos[k, 0]), float(asm.module_pos[k, 1])
            elif f["mode"] == "low_mu":
                mu[b] = 0.3
    payload = dict(mass=pm, x=px, y=py) if (pm > 0).any() else None
    return mode, g, payload, (mu if (mu < 1).any() else None)


def current_stream(R, params, rng, mode=None, sensor_dr=None):
    """Motor-current readings (B,T,n) for simulator output R with the same model and per-trial sensor randomisation
    as `sensors.to_raw` (receive-time delay excluded: the session writer places messages by receive time itself).
    mode: (B,n) sim modes (default R["mode"]). Driven wheels read |I_free tanh(omega/0.2) + tau_motor/kt|, either
    PWM-synchronised or scaled by |duty * g|, times a gain; BRAKE/COAST wheels read 0; then offset + white noise,
    clipped at 0."""
    from .. import sensors as SN
    B, T, n = R["omega"].shape
    sd = SN.sample_sensor_domain(B, rng, params, sensor_dr)
    m = params["motor"]
    kt = float(m["stall_torque_nm_12v"]) / float(m["stall_current_a_12v"])
    mode = torch.as_tensor(R["mode"] if mode is None else mode).cpu()
    driven = ((mode != S.COAST) & (mode != S.BRAKE))[:, None, :]
    om = torch.as_tensor(R["omega"]).float().cpu()
    i_mot = (SN.I_FREE_A * torch.tanh(om / 0.2) + torch.as_tensor(R["tau_motor"]).float().cpu() / kt).abs()
    duty_applied = (torch.as_tensor(R["duty"]).float().cpu() * torch.as_tensor(R["g"]).float().cpu()[:, None, :]).abs()
    syn = torch.as_tensor(sd["current_synced"])[:, None, None]
    i_read = (syn * i_mot + (1 - syn) * duty_applied * i_mot) * torch.as_tensor(sd["current_gain"])[:, None, None]
    cur = torch.where(driven, i_read, torch.zeros_like(i_read))
    noise = torch.as_tensor(rng.standard_normal((B, T, n)), dtype=torch.float32)
    cur = (cur + torch.as_tensor(sd["current_offset"])[:, None, None]
           + torch.as_tensor(sd["current_noise"])[:, None, None] * noise).clamp(min=0)
    return cur.numpy()


def build_domain(B, rng, dom=None, dr=None, device="cpu"):
    """The simulated "true" plant for B trials (FIX M6), via sim's public API only.

    dr:  a domain-randomisation spec for `sim.sample_domain` (ranges override sim.DR_DEFAULT; e.g. {"mu": (0.5, 0.6)}).
    dom: a domain dict whose entries override the sampled ones. Values may be scalars (one fixed plant for every trial),
         length-B sequences / arrays / tensors (per trial), or a callable dom(B, rng) returning such a dict.
         Keys sim does not know are refused. A dict covering every key is used as is (no sampling needed).
    Neither given -> sim.sample_domain(B, rng) (the training DR), exactly as before."""
    if callable(dom):
        dom = dom(B, rng)
    base = S.sample_domain(B, rng, dr) if dr is not None else S.sample_domain(B, rng)
    if not dom:
        return base
    unknown = set(dom) - set(base)
    if unknown:
        raise ValueError(f"unknown domain keys {sorted(unknown)}; sim.sample_domain keys are {sorted(base)}")
    out = dict(base)
    for k, v in dom.items():
        if k == "wheel_yaw_err_std":
            out[k] = float(v)
            continue
        t = torch.as_tensor(np.asarray(v.detach().cpu() if torch.is_tensor(v) else v, dtype=np.float32))
        if t.ndim == 0:
            t = t.expand(B).clone()
        if t.shape != (B,):
            raise ValueError(f"domain key {k}: shape {tuple(t.shape)}, expected () or ({B},)")
        out[k] = t.to(device=base[k].device if torch.is_tensor(base[k]) else device)
    return out


def make_session(out_root, assembly="pair", n_trials=4, seed=0, params=None, gap=2.0, current_wired=False, schedule=None,
                 codebook_dir=None, device="cpu", dt=0.005, dom=None, dr=None, trial_numbers=None, mode_override=None):
    """Simulate `n_trials` scheduled trials of one assembly and write them as a SIMULATION FieldLog v2 session.
    dom / dr: the true plant (see `build_domain`; default = a draw from the training DR).
    Returns (session_dir, truth) where truth holds the simulator ground truth per trial for tests (truth['dom'] is
    the domain actually simulated).
    trial_numbers (WS-P2): simulate exactly these scheduled trials (all of one session entry, i.e. one layout, and one
    probe set; e.g. a masking episode's split-phase trials of M2 alone) instead of the first n_trials of `assembly`.
    mode_override: optional callable(trials, asm) -> (mode, g) replacing the scheduled health (tests only)."""
    V2 = _import_fieldlog()
    params = params or A.load_params(Path(__file__).resolve().parents[2] / "params" / "module_measured.yaml")
    rng = np.random.default_rng(seed)
    sched = schedule or SCH.generate(f"{seed:032x}", assemblies=(assembly,), params=params)
    if trial_numbers is not None:
        want = [int(x) for x in trial_numbers]
        found = {int(t["trial"]): (s, t) for s, _, t in SCH.iter_trials(sched)}
        missing = [x for x in want if x not in found]
        if missing:
            raise KeyError(f"trials {missing[:5]} not in the schedule")
        ents = {id(found[x][0]) for x in want}
        if len(ents) != 1:
            raise ValueError("trial_numbers must belong to one schedule session entry (one layout)")
        session = found[want[0]][0]
        assembly = session["assembly"]
        trials = [found[x][1] for x in want]
    else:
        session = [s for s in sched["sessions"] if s["assembly"] == assembly][0]
        trials = [t for _, _, t in SCH.iter_trials(sched, assembly)][:n_trials]
    asm = A.build(assembly, [tuple(m) for m in session["modules"]], params)
    if len({SCH.trial_pids(t) for t in trials}) != 1:
        raise ValueError("all simulated trials must run the same probe set")
    sch = SCH.trial_schedule(asm, trials[0])
    mode, g, payload, mu = _sim_faults(trials, asm)
    if mode_override is not None:
        mode, g = mode_override(trials, asm)
    dom = build_domain(len(trials), rng, dom=dom, dr=dr)
    R = S.simulate(asm, params, dom, mode, g, schedule=sch, dt=dt, device=device, payload=payload, mu_factor=mu, seed=seed)
    R = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in R.items()}

    out_root = Path(out_root)
    cb_path = FC.overlay_codebook(codebook_dir or out_root / "_codebook")
    rs_dir = out_root / "_runsheets"
    rs_dir.mkdir(parents=True, exist_ok=True)
    if trial_numbers is not None:
        rows = [{"row": k, "trial_id": f"gnd_{FC.PORTION_TOKEN}_{k:03d}", "portion_code": FC.PORTION_CODE, "nn": k,
                 "split_code": 1, "arm_code": SCH.FAULT_CODE.get(str(t["fault_id"])[:2], 0), "block_code": 0, "amp": 0,
                 "sign": 0, "dur_s": 0, "gap_s": 0, "extra_wait_s": 0, "axis_code": 0, "thr_idx": 0, "n_steps": 0}
                for k, t in enumerate(trials, start=1)]
    else:
        rows = SCH.runsheet_rows(sched, assembly)[:n_trials]
    rs_path = SCH.write_runsheet(rs_dir / f"runsheet_{assembly}_{seed}.csv", rows)
    plan = {"study_codes": [0], "session_ordinal": 0, "session_role_code": 1, "vehicle_code": 0, "vehicle_config_code": 0,
            "runsheet_path": str(rs_path)}
    clock = SimClock()
    sess = V2.SessionV2(out_root, "gnd", plan, codebook=cb_path, runsheet_path=str(rs_path), roster_dir=out_root / "_no_roster",
                        use_sim_time=True, clock_source="sim:capxfer", field=False, vehicle_id=None, now=clock.now,
                        operators=((901, 1), (902, 2)), config_path=str(FC.CONFIG_PATH) if FC.CONFIG_PATH.is_file() else None)
    for r in SIM_REASONS:
        sess.mark_simulation(r)

    sp, tp, fw = params["sensors"], params["truth"], params["firmware"]
    cpr = float(params["motor"]["counts_per_output_rev"])
    imu_m, lidar_m = SCH.sensor_modules(asm)
    mods = [m for m, *_ in asm.modules]
    yaw_of = {fr["module"]: math.radians(fr["yaw_deg"]) for fr in session["frames"]}
    pos_of = {fr["module"]: (fr["x"], fr["y"]) for fr in session["frames"]}
    tR = R["t"].numpy()
    # DX fix: the current stream follows sensors.to_raw's model (unidirectional CS, ~0 on BRAKE/COAST wheels,
    # synced/unsynced read, per-trial gain/offset/noise from sensors.SENSOR_DR), not a signed torque/kt.
    cur_all = current_stream(R, params, np.random.default_rng([seed, 0xC0]), mode) if current_wired else None
    dtR = float(tR[1] - tR[0])
    T = sch.T
    asm_code = FC.ASM_TYPE_CODES[assembly.lower()]
    t_cursor = 1.0
    truth = []
    for b, (tr, row) in enumerate(zip(trials, rows)):
        msgs = []                                                    # (t_receipt, src, values, t_src_wall or None)
        jit = lambda: float(rng.uniform(0, 0.001))                   # noqa: E731
        ix = lambda tau: int(np.clip(round(tau / dtR), 0, len(tR) - 1))   # noqa: E731
        t0 = t_cursor
        p = trial_plan(tr, session, params)
        msgs.append((t0 + 0.0003, FC.SRC["TRIAL"], {"trial_num": tr["trial"], "trial_event_code": 1, "asm_type_code": asm_code},
                     clock.wall(t0)))
        for k, fr in enumerate(session["frames"]):
            m = fr["module"]
            msgs.append((t0 + 0.001 + 1e-4 * k, FC.SRC[f"ASM_{m}"],
                         {f"asm_x_m_{FC.mk(m)}": fr["x"], f"asm_y_m_{FC.mk(m)}": fr["y"], f"asm_yaw_deg_{FC.mk(m)}": fr["yaw_deg"]}, None))

        def fs(m, d, t):
            return (t, FC.SRC[f"FLT_{m}"], {f"fs_wheel_code_{FC.mk(m)}": FC.WHEEL_CODES.get(d["wheel"], 9),
                                             f"fs_mode_code_{FC.mk(m)}": FC.FAULT_MODE_CODES[d["mode"]],
                                             f"fs_g_{FC.mk(m)}": d.get("g"), f"fs_trial_{FC.mk(m)}": d["trial"],
                                             f"fs_tick_{FC.mk(m)}": int(t * 1e6)}, None)
        for k, (m, d) in enumerate(fault_messages(tr, session)):
            msgs.append(fs(m, d, t0 + 0.005 + 1e-4 * k))
        # cmd_vel as published, 50 Hz
        for k in range(int(np.ceil(T * 50)) + 1):
            tau = min(k / 50.0, T)
            xi = sch.twist(np.array([tau]))[0]
            msgs.append((t0 + tau + 0.0005 + jit(), FC.SRC["CTL"], {"cmd_vx": xi[0], "cmd_vy": xi[1], "cmd_wz": xi[2]}, clock.wall(t0 + tau)))
        # wheels / current, 100 Hz per module
        ang, om, du, tq = R["wheel_angle"][b].numpy(), R["omega"][b].numpy(), R["duty"][b].numpy(), R["torque"][b].numpy()
        cur_b = cur_all[b] if current_wired else None
        for k in range(int(T * 100)):
            tau = k / 100.0
            i = ix(tau)
            for mi, m in enumerate(mods):
                ws = slice(4 * mi, 4 * mi + 4)
                v = {}
                for c, a_, o_, d_ in zip(FC.CORNERS, ang[i, ws], om[i, ws], du[i, ws]):
                    v[f"cnt_{c}_{FC.mk(m)}"] = math.floor(a_ * cpr / (2 * math.pi))
                    v[f"wspd_{c}_{FC.mk(m)}"] = o_ + 0.01 * rng.normal()
                    v[f"duty_{c}_{FC.mk(m)}"] = float(np.clip(d_ + 0.002 * rng.normal(), -1, 1))
                msgs.append((t0 + tau + 0.002 + jit(), FC.SRC[f"WHL_{m}"], v, clock.wall(t0 + tau)))
                if current_wired:
                    msgs.append((t0 + tau + 0.0025 + jit(), FC.SRC[f"CUR_{m}"],
                                 {f"cur_{c}_{FC.mk(m)}": float(q) for c, q in zip(FC.CORNERS, cur_b[i, ws])}, None))
        # IMU 100 Hz in the IMU module frame
        tw, ac, ps = R["twist"][b].numpy(), R["accel"][b].numpy(), R["pose"][b].numpy()
        cy, sy = math.cos(yaw_of[imu_m]), math.sin(yaw_of[imu_m])
        bias = 0.002 * rng.normal()
        for k in range(int(T * 100)):
            tau = k / 100.0
            i = ix(tau)
            ax, ay = ac[i]
            msgs.append((t0 + tau + 0.003 + jit(), FC.SRC["IMU"],
                         {"wz_radps": tw[i, 2] + bias + float(sp["imu_gyro_noise_radps"]) * rng.normal(),
                          "ax_mps2": cy * ax + sy * ay + float(sp["imu_acc_noise_mps2"]) * rng.normal(),
                          "ay_mps2": -sy * ax + cy * ay + float(sp["imu_acc_noise_mps2"]) * rng.normal()}, clock.wall(t0 + tau)))
        # rf2o odometry 10 Hz at the lidar module centre, module frame
        if lidar_m is not None:
            lat = float(sp["lidar_odom_latency_s"])
            ln = float(sp["lidar_odom_noise_mps"])
            cl, sl = math.cos(yaw_of[lidar_m]), math.sin(yaw_of[lidar_m])
            rx, ry = pos_of[lidar_m]
            for k in range(int(T * 10)):
                tau = k / 10.0
                i = ix(tau)
                vx, vy, wz = tw[i]
                vxa, vya = vx - wz * ry, vy + wz * rx                    # velocity of the lidar point, assembly frame
                msgs.append((t0 + tau + lat + jit(), FC.SRC["LODOM"],
                             {"lo_vx_mps": cl * vxa + sl * vya + ln * rng.normal(), "lo_vy_mps": -sl * vxa + cl * vya + ln * rng.normal(),
                              "lo_wz_radps": wz + 2 * ln * rng.normal()}, clock.wall(t0 + tau)))
        # AprilTag truth 30 Hz, latency 0.15 s
        lat = float(tp["apriltag_latency_s"])
        pn, yn = float(tp["apriltag_pos_noise_m"]), float(tp["apriltag_yaw_noise_rad"])
        world = (0.3 * rng.normal(), 0.3 * rng.normal(), rng.uniform(-math.pi, math.pi))  # arena placement of the start pose
        cw, sw = math.cos(world[2]), math.sin(world[2])
        for k in range(int(T * 30)):
            tau = k / 30.0
            i = ix(tau)
            x, y, th = ps[i]
            X, Y, TH = world[0] + cw * x - sw * y + pn * rng.normal(), world[1] + sw * x + cw * y + pn * rng.normal(), world[2] + th + yn * rng.normal()
            msgs.append((t0 + tau + lat + jit(), FC.SRC["TRUTH"],
                         {"gt_x_m": X, "gt_y_m": Y, "gt_qz": math.sin(TH / 2), "gt_qw": math.cos(TH / 2)}, clock.wall(t0 + tau)))
        # end of trial
        for k, (m, d) in enumerate(fault_messages(tr, session, restore=True)):
            msgs.append(fs(m, d, t0 + T + 0.005 + 1e-4 * k))
        msgs.append((t0 + T + 0.01, FC.SRC["TRIAL"], {"trial_num": tr["trial"], "trial_event_code": 2, "asm_type_code": asm_code},
                     clock.wall(t0 + T)))
        msgs.sort(key=lambda r: r[0])
        clock.t = t0 - 0.05
        sess.begin_trial(int(row["row"]), t_mono_ns=clock.mono(t0 - 0.05))
        for t_r, src, vals, ts in msgs:
            sess.write(src, vals, t_src_ns=ts, t_mono_ns=clock.mono(t_r))
        clock.t = msgs[-1][0] + 0.05
        sess.end_trial(1, t_mono_ns=clock.mono(clock.t))
        truth.append(dict(trial=tr["trial"], mode=mode[b].clone(), g=g[b].clone(), ood=tr["ood"]))
        t_cursor = clock.t + gap
        clock.t = t_cursor - gap / 2
        sess.tick()
    clock.t = t_cursor
    d = sess.close()
    return d, dict(trials=truth, R=R, asm=asm, schedule=sch, dom=dom)

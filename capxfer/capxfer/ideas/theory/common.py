"""WS-TH shared helpers: the quantities that appear in docs/THEORY.md (rows, excitation, masking bound).

Notation follows docs/THEORY.md and capxfer/ideas/i2/theory.py:
  g_i = J_i^T r_i (drive row), h_i = J_i^T n_i (lateral row), J_i xi = contact velocity of wheel i.
  q_d = W xi_d / |xi_d|_W^2 so that rho_d(xi) = <q_d, xi>.
  M = sum_{j != i} N_j g_j g_j^T  (healthy load-weighted row Gram matrix)
  kappa_i = N_i max_{j != i} |g_j|_{M^-1} sum_k |a_ik|_{M^-1}           (dimensionless, mu- and s0-free)
  beta_i(d) = N_i |q_d|_{M^-1} sum_k |a_ik|_{M^-1}                     (units 1/(m/s))
  bound    |Delta rho_d| <= s0 * beta_i(d) * atanh(kappa_i)/kappa_i     if kappa_i < 1        (Prop. 3)
  first-order estimate      s0 * N_i |<q_d, M^-1 g_i>|                  (saturated passive drive fault)
"""
from __future__ import annotations

import json
import os

import numpy as np

from capxfer import assembly as A
from capxfer.ideas.i2 import theory as T
from capxfer.sim import WEAK, COAST, BRAKE, ROLLER

OUT = os.path.expanduser("~/.cache/capxfer/TH")
os.makedirs(OUT, exist_ok=True)
MODE_NAME = {WEAK: "weak", COAST: "coast", BRAKE: "brake", ROLLER: "roller"}


def params():
    return A.load_params("params/module_measured.yaml")


def lateral_w_row(asm):
    """a_w = J^T w (rolling-direction row) for each wheel -> (n,3)."""
    p, w = asm.wheel_pos, asm.w_hat
    return np.stack([w[:, 0], w[:, 1], p[:, 0] * w[:, 1] - p[:, 1] * w[:, 0]], 1)


def fault_rows(asm, i, mode):
    """Rows along which a fault of `mode` on wheel i can transmit friction force (c_r = 0 idealisation)."""
    g, h = T.rows(asm)
    if mode == ROLLER:
        return np.stack([lateral_w_row(asm)[i], h[i]])
    return g[i][None]


def q_vecs(asm):
    """(14,3) q_d = W xi_d / |xi_d|_W^2 for the 14 legs."""
    L = T.legs(asm)
    Lc = asm.char_length()
    W = np.array([1.0, 1.0, Lc * Lc])
    return L * W / (L * L * W).sum(1, keepdims=True)


def masking_quantities(asm, i, mode, N=None):
    """-> dict(kappa, beta (14,), bound (14,) or inf, first_order (14,))."""
    g, _ = T.rows(asm)
    N = T.normal_loads(asm) if N is None else N
    others = [j for j in range(asm.n) if j != i]
    M = (N[others, None, None] * g[others, :, None] * g[others, None, :]).sum(0)
    Minv = np.linalg.inv(M)
    nrm = lambda x: np.sqrt(np.einsum("...i,ij,...j->...", x, Minv, x))
    a = fault_rows(asm, i, mode)
    sa = nrm(a).sum()
    kappa = N[i] * nrm(g[others]).max() * sa
    q = q_vecs(asm)
    beta = N[i] * nrm(q) * sa
    s0 = T.NOM["s0"]
    if kappa < 1:
        fac = np.arctanh(kappa) / kappa if kappa > 1e-12 else 1.0
        bound = s0 * beta * fac
    else:
        bound = np.full(14, np.inf)
    # first-order (linear-response) estimate: healthy stiffness H0 = (mu/s0) M, faulty-wheel force at zeroth order
    xi = T.legs(asm)                                                       # (14,3)
    if mode == ROLLER:
        aw, hh = a
        s_w = xi @ aw - (xi @ g[i]) / float(asm.w_hat[i] @ asm.r_hat[i])  # rolling-direction slip at xi_d
        s_n = xi @ hh
        F = -(np.tanh(s_w / s0)[:, None] * aw + np.tanh(s_n / s0)[:, None] * hh)   # / (mu N_i)
    else:
        F = -np.tanh((xi @ g[i]) / s0)[:, None] * g[i]                    # locked drive wheel (omega_i = 0)
    first = s0 * N[i] * np.abs(np.einsum("ki,ij,kj->k", q, Minv, F))
    return dict(kappa=float(kappa), beta=beta, bound=bound, first_order=first)


def rebuild(r, P):
    asm = A.build(r["name"], [tuple(m) for m in r["modules"]], P)
    assert np.allclose(asm.wheel_pos, r["wheel_pos"])
    return asm


def load_sweeps():
    import torch
    out = []
    for f in ("sweep_n16.pt", "sweep_n20_24_cluster.pt"):
        d = torch.load(os.path.expanduser(f"~/.cache/capxfer/I2/{f}"), weights_only=False)
        out += [r for r in d["results"] if r is not None]
    return out


def dump(name, obj):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, "w"), indent=1, default=float)
    print("wrote", p)

"""WS-F: vendored Fleet capplan subset imports and is unmodified apart from recorded import rewrites."""
import hashlib
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "..", "capxfer", "vendor")


def _sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_vendor_imports():
    from capxfer.vendor.capplan.planners import base
    from capxfer.vendor.capplan import world, _shim
    assert hasattr(base, "Lattice") and hasattr(base, "MotionModel") and hasattr(world, "World")
    assert _shim.wrap(3 * math.pi) == -math.pi or abs(abs(_shim.wrap(3 * math.pi)) - math.pi) < 1e-12


def test_vendor_hashes_recorded():
    text = open(os.path.join(VENDOR, "VENDOR.md")).read()
    assert "bf4fdc538cff39954067fd2740d5ab8c0961ef6f" in text
    for rel in ("capplan/world.py", "capplan/planners/base.py", "capplan/README.md",
                "capplan/capability_descriptor_v0.md", "capplan/_shim.py"):
        assert _sha(os.path.join(VENDOR, rel)) in text, rel


def test_vendored_lattice_plans_in_open_world():
    from capxfer.vendor.capplan.planners.base import Lattice, MotionModel, Prim
    from capxfer.vendor.capplan.world import World, _blank
    w = World(_blank(3.0, 3.0), 0.05)
    pts = np.array([[0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1], [0.1, -0.1]])
    prims = [Prim(np.array([0.2, 0, 0]), 1.0), Prim(np.array([0, 0.2, 0]), 1.0), Prim(np.array([0, -0.2, 0]), 1.0)]
    lat = Lattice(w, MotionModel(prims, pts, 0.02, 16, 0.2, 0.5, 0.1))
    pl = lat.search((0.5, 0.5, 0.0), [(1.3, 1.1, 0.0)], pos_tol=0.05, head_tol=0.1)
    assert pl.status == "ok" and len(pl.segments) >= 2

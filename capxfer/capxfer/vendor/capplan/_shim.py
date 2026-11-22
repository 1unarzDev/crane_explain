"""Glue for the vendored subset (NOT a Fleet file). The definitions below are copied verbatim from the Fleet sources at
commit bf4fdc538cff39954067fd2740d5ab8c0961ef6f so that planners/base.py does not pull in capplan.common (which imports
recover_under_load and the Fleet profile JSON) or capplan.tasks:
  rot, wrap          <- Tools/Fleet/capplan/common.py
  POS_TOL, HEAD_TOL  <- Tools/Fleet/capplan/tasks.py
"""
import math

import numpy as np

POS_TOL, HEAD_TOL = 0.15, 0.2


def rot(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

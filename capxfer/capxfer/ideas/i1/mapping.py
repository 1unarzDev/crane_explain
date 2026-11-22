"""Map WS-C prediction groups (set/gi) onto devW shard groups by matching true health arrays."""
import pickle, pathlib
import numpy as np, torch

META = pathlib.Path.home() / ".cache/capxfer/I1/devW_meta.pkl"


def build(d):
    G = pickle.load(open(META, "rb"))["_groups"]
    key = {}
    for gi, g in enumerate(G):
        k = (g["mode"].shape, g["mode"].tobytes(), np.round(g["g"], 4).tobytes())
        key[k] = gi
    gid = np.array(d["group_id"])
    mt, gt, nw = d["mode_true"].numpy(), d["g_true"].numpy(), d["n_wheels"].numpy()
    out = np.full(len(gid), -1)
    for u in np.unique(gid):
        idx = np.flatnonzero(gid == u)
        n = nw[idx[0]]
        m = mt[idx, :n].astype(np.int64)
        g = np.round(gt[idx, :n].astype(np.float32), 4)
        k = (m.shape, m.tobytes(), g.tobytes())
        if k in key:
            out[idx] = key[k]
    return G, out

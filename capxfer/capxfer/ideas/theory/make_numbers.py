"""Collect every th_* number quoted in paper/sections/theory.tex into ~/.cache/capxfer/TH/th_numbers.json
(flat {key: {"tex": preformatted string, "value": float, "source": file, "path": json path}}). WS-FIG binds from it.

  PYTHONPATH=. .venv/bin/python -m capxfer.ideas.theory.make_numbers
"""
from __future__ import annotations

import json
import os

from capxfer.ideas.theory.common import OUT, dump


def get(fname, path):
    d = json.load(open(os.path.join(OUT, fname)))
    for p in path.split("/"):
        d = d[p]
    return d


def main():
    M = "check_masking.json"
    V = "check_masking_violations.json"
    spec = {
        # key: (file, path, format, transform)
        "th_exc_sim_n_unexc": ("check_excitation.json", "sim/n_unexcited", "{:,}", None),
        "th_exc_sim_unexc_drho_max": ("check_excitation.json", "sim/unexcited_drho_max", "{:.4f}", None),
        "th_exc_ideal_unexc_max": ("check_excitation.json", "qs_idealised/unexcited_max_signal_change", "{:.0e}", None),
        "th_mask_kappa_single": (M, "idealised/by_n/4/kappa_med", "{:.2f}", None),
        "th_mask_ideal_n_faults": (M, "idealised/n_kappa_lt_1", "{:,}", None),
        "th_mask_ideal_holds": (M, "idealised/frac_bound_holds", "{:.0%}", None),
        "th_mask_ideal_ratio_max": (M, "idealised/max_ratio_measured_over_bound", "{:.3f}", None),
        "th_mask_ideal_ratio_med": (M, "idealised/median_ratio_measured_over_bound", "{:.2f}", None),
        "th_mask_ideal_sat_ratio_max": (M, "idealised/incl_saturated_legs_max_ratio", "{:.1f}", None),
        "th_mask_C": (M, "measured/passive/C_over_nm1", "{:.2f}", None),
        "th_mask_roller_ratio_med": (M, "measured/roller/by_n/24/sim_over_bound_med", "{:.2f}", None),
        "th_mask_sim_exceed_unsat_max": (V, "passive/8/exceed_frac_unsat_legs", "{:.0%}", None),
        "th_mask_sim_excess_unsat_max": (V, "summary/max_abs_excess_unsat", "{:.4f}", None),
        "th_mask_sim_exceed_other": (V, "summary/by_class/other", "{:d}", None),
        "th_split_f4_child_z": ("check_split_double.json", "summary_nominal/child_z_sim_max", "{:.3f}", None),
        "th_split_f4_parent_dz_ideal": ("check_split_double.json", "summary_ideal/dz_par_absmax", "{:.0e}", None),
        "th_split_masked_passive": ("check_split.json", "passive/parent_masked_frac", "{:.0%}", None),
        "th_split_child_unsafe08_given_masked": ("check_split.json", "passive/child_unsafe_given_parent_masked_0.8",
                                                 "{:.0%}", None),
        "th_split_spearman": ("check_split.json", "all/spearman_parent_worst_vs_child_loss", "{:.2f}", None),
        "th_bs_n_asm": ("check_blindspot.json", "n_asm", "{:,}", None),
        "th_obl_margin": ("check_blindspot.json", "oblique_max_val", "{:.3f}", None),
        "th_cert_max_miscov": ("check_cert.json", "P1P5P2/max_miscov", "{:.3f}", None),
    }
    out = {}
    for k, (f, p, fmt, _) in spec.items():
        v = get(f, p)
        if fmt.endswith("e}") and abs(v) < 1e-12:
            tex = r"$<10^{-12}$"                                         # solver precision
        else:
            tex = fmt.format(v).replace("%", r"\%").replace(",", "{,}")
        out[k] = dict(tex=tex, value=v, source=os.path.join("~/.cache/capxfer/TH", f), path=p)
    dump("th_numbers.json", out)
    for k, v in out.items():
        print(f"{k:40s} {v['tex']:>10s}  {v['source']}:{v['path']}")


if __name__ == "__main__":
    main()

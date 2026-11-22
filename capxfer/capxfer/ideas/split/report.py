"""Print SPLIT tables from results_*.json (SIMULATION ONLY)."""
import json
import sys

from capxfer.ideas.split import common as C

LABEL = dict(comp_se="(a) COMP structured_ens", post_se="(a') post-cal structured_ens", comp_ab="(d) COMP analytic_bayes",
             post_ab="(d') post-cal analytic_bayes", bb_carry="(b) black-box carry-over", bb_post="(b') black-box post-cal",
             healthy="(c) assume healthy", orc_h="oracle health, unknown plant", oracle="(e) oracle",
             pa_ens="(f) probe-after ensemble", pa_se="(f) probe-after structured_ens")


def main(path):
    R = json.loads(open(path).read())
    T = R["tables"]
    meths = list(dict.fromkeys(k.split("|")[2] for k in T))
    groups = list(dict.fromkeys(k.split("|")[1] for k in T))
    print(f"# {path}  alpha={R['alpha']}")
    for tau in R["taus"]:
        print(f"\n## tau={tau}: unsafe-go (go & z<tau) / P(go|unsafe) / cert / P(go|safe)  [n, n_unsafe]")
        for g in groups:
            t0 = T[f"{tau}|{g}|{meths[0]}"]
            print(f"\n### {g}  n={t0['n']} unsafe={t0['n_unsafe']}")
            print("| method | unsafe-go | P(go given unsafe) | cert | P(go given safe) | lb cov(z) | cov pt | cov strafe | width |")
            print("|---|---|---|---|---|---|---|---|---|")
            for m in meths:
                t = T[f"{tau}|{g}|{m}"]
                print(f"| {LABEL.get(m, m)} | {t['unsafe_go']:.4f} | {t['false_go']:.3f} | {t['cert']:.3f} | {t['useful']:.3f} "
                      f"| {t['lb_cov']:.3f} | {t['cov_pt']:.3f} | {t['cov_strafe']:.3f} | {t['width']:.3f} |")
    for k in ("stage1_module_cov", "pre", "task"):
        if k in R:
            print(f"\n## {k}")
            for kk, v in R[k].items():
                print(kk, json.dumps(v))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(C.OUT / "results_singles_P1P5.json"))

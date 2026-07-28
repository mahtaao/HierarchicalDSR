"""
Cross-generator comparison for rebuttal item #2.

Overlays the failure-axis sweep under three mixing families that share the same
source dynamics (make_sources) and differ only in how sources are mapped to
sensors:
    learned W   — shPLRNN observation matrix (the paper's headline generator)
    random  W   — random unit-norm rows (existing control)
    forward L   — MNE 3-layer sphere EEG leadfield (physics, untrained)

Question it answers: is the per-method failure pattern an artifact of the one
learned generator, or does it survive a completely different inductive bias?

Reads:
    results/xgen/learned_W.npy
    results/xgen/random_W.npy
    results/forward_model/all_results.npy
Writes:
    results/figures/fig_cross_generator.{png,pdf}
    results/figures/fig_cross_generator_mvsd.{png,pdf}
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["PCA", "FastICA", "TCL", "iVAE"]
MCOLORS = {"PCA": "#4C72B0", "FastICA": "#55A868", "TCL": "#C44E52", "iVAE": "#8172B3"}

GEN = [
    ("learned W", "results/xgen/learned_W.npy"),
    ("random W",  "results/xgen/random_W.npy"),
    ("forward L", "results/forward_model/all_results.npy"),
]

# shared axes (same configs across all three generators)
AXES = [
    ("1f_snr",   "1/f SNR (dB)",              lambda c: c),
    ("nonlin",   "Mixing nonlinearity",        lambda c: c),
    ("coupling", "Source coupling",            lambda c: c),
    ("ns",       "Nonstationarity type",       lambda c: c),
]


def load(path):
    return np.load(path, allow_pickle=True).item()


def cfg_list(d, axis):
    return list(d[axis][METHODS[0]].keys())


def mean_std(d, axis, method):
    cfgs = cfg_list(d, axis)
    mu = np.array([np.nanmean(d[axis][method][c]) for c in cfgs])
    sd = np.array([np.nanstd(d[axis][method][c]) for c in cfgs])
    return cfgs, mu, sd


def main():
    os.makedirs("results/figures", exist_ok=True)
    data = {name: load(p) for name, p in GEN}

    # ── main grid: rows = generators, cols = shared axes ──────────────────────
    nrow, ncol = len(GEN), len(AXES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.8 * nrow),
                             sharey=True)
    for r, (gname, _) in enumerate(GEN):
        d = data[gname]
        for c, (akey, alabel, _) in enumerate(AXES):
            ax = axes[r, c]
            cfgs = cfg_list(d, akey)
            x = np.arange(len(cfgs))
            for m in METHODS:
                _, mu, sd = mean_std(d, akey, m)
                ax.plot(x, mu, "-o", ms=4, color=MCOLORS[m], label=m)
                ax.fill_between(x, mu - sd, mu + sd, color=MCOLORS[m], alpha=0.15)
            ax.set_xticks(x)
            ax.set_xticklabels([str(cc) for cc in cfgs], fontsize=7, rotation=30)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title(alabel, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{gname}\nMCC", fontsize=9)
    axes[0, -1].legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.suptitle("Failure axes under three mixing generators (shared source dynamics)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(f"results/figures/fig_cross_generator.{ext}",
                    dpi=200 if ext == "png" else 300, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/figures/fig_cross_generator.{png,pdf}")

    # ── sources-vs-sensors panel (x differs per generator; forward L reaches M>>D)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6.5, 4.2))
    styles = {"learned W": ":", "random W": "--", "forward L": "-"}
    for gname, _ in GEN:
        d = data[gname]
        cfgs = cfg_list(d, "m_vs_d")
        Ks = np.array([float(k) for k in cfgs])
        for m in METHODS:
            _, mu, _ = mean_std(d, "m_vs_d", m)
            ax2.plot(Ks, mu, styles[gname] + "o", ms=4, color=MCOLORS[m],
                     label=f"{m} ({gname})" if gname == "forward L" else None)
    ax2.axvline(22, color="k", lw=0.8, ls=":", alpha=0.5)
    ax2.text(22, 1.0, "M=D", fontsize=8, ha="left")
    ax2.set_xlabel("M (sources), D=22 sensors fixed")
    ax2.set_ylabel("MCC")
    ax2.set_title("Sources vs sensors: forward L reaches the M >> D regime")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7, ncol=2, title="line: — forward L  -- random  ·· learned")
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(f"results/figures/fig_cross_generator_mvsd.{ext}",
                     dpi=200 if ext == "png" else 300, bbox_inches="tight")
    plt.close(fig2)
    print("Saved results/figures/fig_cross_generator_mvsd.{png,pdf}")


if __name__ == "__main__":
    main()

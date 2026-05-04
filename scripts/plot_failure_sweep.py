#!/usr/bin/env python3
"""Publication-quality figures from the failure-axis sweep.
Reads results/failure_axis/all_results.npy (5 seeds × 5 axes × 4 methods).
Saves PDFs to results/figures/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── rcParams ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# Wong colorblind-safe palette
COLORS  = {"PCA": "#999999", "FastICA": "#0072B2", "TCL": "#E69F00", "iVAE": "#CC79A7"}
MARKERS = {"PCA": "o",       "FastICA": "s",       "TCL": "^",       "iVAE": "D"}
METHODS = ["PCA", "FastICA", "TCL", "iVAE"]

os.makedirs("results/figures", exist_ok=True)
d = np.load("results/failure_axis/all_results.npy", allow_pickle=True).item()

# ── Axis metadata (in paper-friendly order) ───────────────────────────────────
AXES = [
    {
        "key": "1f_snr",
        "title": "(a) Pink-noise SNR",
        "xlabel": "SNR (dB)",
        "cfgs": [30, 15, 10, 5, 0],
        "xvals": [30, 15, 10, 5, 0],
        "xticks": ["30", "15", "10", "5", "0"],
        "numeric": True,
        "jitter_scale": 0.7,
    },
    {
        "key": "nonlin",
        "title": "(b) Mixing nonlinearity",
        "xlabel": "Nonlinearity",
        "cfgs": ["linear", "cube", "tanh"],
        "xvals": [0, 1, 2],
        "xticks": ["linear", "cube", "tanh"],
        "numeric": False,
        "jitter_scale": 0.12,
    },
    {
        "key": "coupling",
        "title": "(c) Source coupling",
        "xlabel": "Coupling $\\gamma$",
        "cfgs": [0.0, 0.1, 0.3, 0.6, 0.9],
        "xvals": [0.0, 0.1, 0.3, 0.6, 0.9],
        "xticks": ["0", "0.1", "0.3", "0.6", "0.9"],
        "numeric": True,
        "jitter_scale": 0.015,
    },
    {
        "key": "ns",
        "title": "(d) Nonstationarity type",
        "xlabel": "NS regime",
        "cfgs": ["stationary", "source_aligned", "confounder_aligned"],
        "xvals": [0, 1, 2],
        "xticks": ["stationary", "source-\naligned", "confounder-\naligned"],
        "numeric": False,
        "jitter_scale": 0.12,
    },
    {
        "key": "m_vs_d",
        "title": "(e) Sources $M$ vs channels $D{=}22$",
        "xlabel": "# sources $M$",
        "cfgs": [2, 4, 8, 16, 22],
        "xvals": [2, 4, 8, 16, 22],
        "xticks": ["2", "4", "8", "16", "22"],
        "numeric": True,
        "jitter_scale": 0.3,
    },
]

rng = np.random.default_rng(42)


# ════════════════════════════════════════════════════════════════════════════════
# Figure 1 — 5-panel line + seed-dot figure
# ════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(12, 2.9), constrained_layout=True)

for ax, meta in zip(axes, AXES):
    axis_data = d[meta["key"]]
    xvals = np.array(meta["xvals"], dtype=float)

    # method-level offset so seed dots don't fully overlap across methods
    n_methods = len(METHODS)
    method_offsets = np.linspace(-meta["jitter_scale"] * 0.7,
                                  meta["jitter_scale"] * 0.7, n_methods)

    for i, method in enumerate(METHODS):
        method_data = axis_data[method]
        means, stds, seed_matrix = [], [], []
        for cfg in meta["cfgs"]:
            vals = np.array(method_data[cfg], dtype=float)
            means.append(vals.mean())
            stds.append(vals.std())
            seed_matrix.append(vals)

        means = np.array(means)
        stds  = np.array(stds)
        seeds = np.array(seed_matrix)  # (n_cfgs, 5)

        c = COLORS[method]
        m = MARKERS[method]

        # Mean line + std band
        ax.plot(xvals, means, color=c, marker=m, markersize=5, linewidth=1.6,
                zorder=3, clip_on=False, markeredgewidth=0.5, markeredgecolor="white")
        ax.fill_between(xvals, means - stds, means + stds,
                        color=c, alpha=0.15, linewidth=0, zorder=2)

        # Individual seed dots (jittered)
        for j, (xv, vals) in enumerate(zip(xvals, seeds)):
            jitter = rng.uniform(-meta["jitter_scale"], meta["jitter_scale"], len(vals))
            jitter_x = xv + method_offsets[i] + jitter * 0.3
            ax.scatter(jitter_x, vals, color=c, alpha=0.35, s=8,
                       linewidths=0, zorder=1)

    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(xvals)
    ax.set_xticklabels(meta["xticks"], fontsize=8 if not meta["numeric"] else 8.5)
    ax.set_xlabel(meta["xlabel"], labelpad=4)
    ax.set_title(meta["title"], pad=5)
    ax.axhline(0.25, color="#cccccc", linewidth=0.7, linestyle="--", zorder=0)  # chance reference

    if ax is axes[0]:
        ax.set_ylabel("MCC", labelpad=4)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

# Shared legend
legend_handles = [
    Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], linewidth=1.6,
           markersize=5, markeredgewidth=0.5, markeredgecolor="white", label=m)
    for m in METHODS
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04),
           columnspacing=1.2, handletextpad=0.4)

fig.savefig("results/figures/fig1_failure_axes.pdf", bbox_inches="tight", dpi=300)
fig.savefig("results/figures/fig1_failure_axes.png", bbox_inches="tight", dpi=200)
print("Saved: fig1_failure_axes.pdf/.png")


# ════════════════════════════════════════════════════════════════════════════════
# Figure 2 — Mean vs Std scatter (reliability map)
# Shows: FastICA = high mean / low std (reliable); iVAE = high mean / high std
# ════════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)

for method in METHODS:
    all_means, all_stds = [], []
    for meta in AXES:
        axis_data = d[meta["key"]][method]
        for cfg in meta["cfgs"]:
            vals = np.array(axis_data[cfg], dtype=float)
            all_means.append(vals.mean())
            all_stds.append(vals.std())

    all_means = np.array(all_means)
    all_stds  = np.array(all_stds)
    ax2.scatter(all_means, all_stds, color=COLORS[method], marker=MARKERS[method],
                s=22, alpha=0.7, label=method, linewidths=0.3, edgecolors="white")

ax2.set_xlabel("Mean MCC across 5 seeds")
ax2.set_ylabel("Std MCC across 5 seeds")
ax2.set_xlim(-0.02, 1.05)
ax2.set_ylim(-0.01, 0.35)
ax2.legend(frameon=False, loc="upper left")

# Annotate quadrants with light text
ax2.text(0.82, 0.30, "reliable\n(high, consistent)", fontsize=7, color="#555",
         ha="center", style="italic")
ax2.text(0.82, 0.02, "high but\nunstable", fontsize=7, color="#555",
         ha="center", style="italic")
ax2.text(0.10, 0.02, "consistently\nfails", fontsize=7, color="#555",
         ha="center", style="italic")
ax2.axhline(0.1, color="#dddddd", linewidth=0.7, linestyle=":", zorder=0)
ax2.axvline(0.5, color="#dddddd", linewidth=0.7, linestyle=":", zorder=0)

fig2.savefig("results/figures/fig2_reliability_scatter.pdf", bbox_inches="tight", dpi=300)
fig2.savefig("results/figures/fig2_reliability_scatter.png", bbox_inches="tight", dpi=200)
print("Saved: fig2_reliability_scatter.pdf/.png")


# ════════════════════════════════════════════════════════════════════════════════
# Figure 3 — Heatmap: mean MCC across all conditions
# Rows = methods, cols = one representative condition per axis (the hardest)
# ════════════════════════════════════════════════════════════════════════════════
conditions = [
    ("SNR 0 dB",       "1f_snr",   0),
    ("tanh mixing",    "nonlin",   "tanh"),
    ("coup. 0.9",      "coupling", 0.9),
    ("stationary",     "ns",       "stationary"),
    ("M=22 (=D)",      "m_vs_d",   22),
    # also include easy baselines for contrast
    ("SNR 30 dB",      "1f_snr",   30),
    ("linear mixing",  "nonlin",   "linear"),
    ("coup. 0.0",      "coupling", 0.0),
    ("src-aligned",    "ns",       "source_aligned"),
    ("M=2",            "m_vs_d",   2),
]

hm_data = np.zeros((len(METHODS), len(conditions)))
for ci, (_, axis_key, cfg) in enumerate(conditions):
    for mi, method in enumerate(METHODS):
        vals = np.array(d[axis_key][method][cfg], dtype=float)
        hm_data[mi, ci] = vals.mean()

col_labels = [c[0] for c in conditions]
col_groups = ["Hard conditions"] * 5 + ["Easy baselines"] * 5

fig3, ax3 = plt.subplots(figsize=(8.5, 2.5), constrained_layout=True)

import matplotlib.colors as mcolors
cmap = "RdYlGn"
im = ax3.imshow(hm_data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

ax3.set_yticks(range(len(METHODS)))
ax3.set_yticklabels(METHODS)
ax3.set_xticks(range(len(conditions)))
ax3.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)

# Annotate cells
for mi in range(len(METHODS)):
    for ci in range(len(conditions)):
        val = hm_data[mi, ci]
        color = "white" if val < 0.35 or val > 0.85 else "#333"
        ax3.text(ci, mi, f"{val:.2f}", ha="center", va="center",
                 fontsize=7.5, color=color, fontweight="bold")

# Divider between hard/easy groups
ax3.axvline(4.5, color="white", linewidth=2)

cbar = fig3.colorbar(im, ax=ax3, shrink=0.85, pad=0.01)
cbar.set_label("Mean MCC", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Group labels above
ax3.annotate("Hard conditions", xy=(2/len(conditions), 1.07), xycoords="axes fraction",
             fontsize=8.5, color="#555", ha="center", style="italic")
ax3.annotate("Easy baselines", xy=(7.5/len(conditions), 1.07), xycoords="axes fraction",
             fontsize=8.5, color="#555", ha="center", style="italic")

fig3.savefig("results/figures/fig3_heatmap.pdf", bbox_inches="tight", dpi=300)
fig3.savefig("results/figures/fig3_heatmap.png", bbox_inches="tight", dpi=200)
print("Saved: fig3_heatmap.pdf/.png")


# ════════════════════════════════════════════════════════════════════════════════
# Figure 4 — Seed-to-seed variance by method (boxplot of per-condition stds)
# One data point per condition (21 total), 4 methods on x-axis.
# Key message: iVAE is highly unstable; FastICA is stable (where it works).
# ════════════════════════════════════════════════════════════════════════════════

# Collect std-per-condition for every method
method_stds = {m: [] for m in METHODS}
method_means = {m: [] for m in METHODS}
for meta in AXES:
    for method in METHODS:
        axis_data = d[meta["key"]][method]
        for cfg in meta["cfgs"]:
            vals = np.array(axis_data[cfg], dtype=float)
            method_stds[method].append(vals.std())
            method_means[method].append(vals.mean())

fig4, ax4 = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

positions = np.arange(len(METHODS))
for i, method in enumerate(METHODS):
    stds = np.array(method_stds[method])
    bp = ax4.boxplot(stds, positions=[i], widths=0.45,
                     patch_artist=True, notch=False,
                     medianprops=dict(color="white", linewidth=2),
                     whiskerprops=dict(color=COLORS[method], linewidth=1.2),
                     capprops=dict(color=COLORS[method], linewidth=1.2),
                     flierprops=dict(marker="", linestyle="none"),
                     boxprops=dict(facecolor=COLORS[method], alpha=0.45,
                                   edgecolor=COLORS[method], linewidth=1.2))
    # Jittered seed dots
    jx = i + rng.uniform(-0.18, 0.18, len(stds))
    ax4.scatter(jx, stds, color=COLORS[method], s=18, alpha=0.75,
                zorder=3, linewidths=0, marker=MARKERS[method])

ax4.set_xticks(positions)
ax4.set_xticklabels(METHODS)
ax4.set_ylabel("Std MCC across 5 seeds\n(one point = one condition)", labelpad=4)
ax4.set_ylim(-0.01, 0.36)
ax4.set_yticks([0, 0.1, 0.2, 0.3])
ax4.axhline(0.1, color="#dddddd", linewidth=0.8, linestyle="--", zorder=0)
ax4.set_title("Seed-to-seed variability across all conditions (N=21 per method)",
              pad=6, fontsize=9.5)

fig4.savefig("results/figures/fig4_variance_boxplot.pdf", bbox_inches="tight", dpi=300)
fig4.savefig("results/figures/fig4_variance_boxplot.png", bbox_inches="tight", dpi=200)
print("Saved: fig4_variance_boxplot.pdf/.png")

print("\nAll figures saved to results/figures/")

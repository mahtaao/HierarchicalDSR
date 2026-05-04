"""
Tier-3 biological correspondence check: equivalent dipole fit for each
obs_matrix column using MNE's sphere forward model.

Reproduces the DIPFIT criterion (Delorme et al., 2012): a source component
is considered biologically plausible if its scalp topography is explained
by a single equivalent dipole with residual variance (RV) < 15%.

Usage:
    python scripts/dipole_check.py \
        --model_path trained_models/bci/bci/linear_M4/001/model_500.pt \
        --out_dir results/dipole_check

Output:
    results/dipole_check/topomaps.png   — scalp topographies (obs_matrix cols)
    results/dipole_check/rv_table.txt   — RV per component
    results/dipole_check/rv_bar.png     — bar chart of RV values
"""

import argparse
import os
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
from mne.datasets import eegbci
mne.set_log_level("WARNING")

# BCI Competition IV-2a channel names (22 EEG, no EOG)
BCI_CH_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]
N_CH = len(BCI_CH_NAMES)  # 22
SFREQ = 250.0


def build_info():
    info = mne.create_info(ch_names=BCI_CH_NAMES, sfreq=SFREQ, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    # Drop channels not in our set (montage has ~94 channels)
    montage_ch = set(montage.ch_names)
    keep = [ch for ch in BCI_CH_NAMES if ch in montage_ch]
    missing = [ch for ch in BCI_CH_NAMES if ch not in montage_ch]
    if missing:
        warnings.warn(f"Channels not found in standard_1020 montage: {missing}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info.set_montage(montage, match_case=True, on_missing="warn")
    return info


def fit_dipole_sphere(topo, info):
    """
    Fit a single equivalent dipole to a scalp topography using MNE sphere model.
    Returns (gof_percent, rv).  RV = 1 - GOF/100.
    """
    sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info,
                                   verbose=False)
    evoked = mne.EvokedArray(topo.reshape(-1, 1), info, tmin=0)
    evoked.set_eeg_reference("average", projection=True, verbose=False)
    evoked.apply_proj(verbose=False)
    noise_cov = mne.make_ad_hoc_cov(info, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dip = mne.fit_dipole(evoked, noise_cov, sphere, verbose=False)[0]
    gof = float(dip.gof[0])
    rv = 1.0 - gof / 100.0
    pos = dip.pos[0]
    ori = dip.ori[0]
    amp = float(dip.amplitude[0])
    return gof, rv, pos, ori, amp


def plot_topomaps(obs_matrix, info, out_path, rv_vals):
    """Plot scalp topographies for all components."""
    n_comp = obs_matrix.shape[0]
    fig, axes = plt.subplots(1, n_comp, figsize=(3 * n_comp, 3.5))
    if n_comp == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        topo = obs_matrix[i]  # (22,)
        mne.viz.plot_topomap(
            topo, info, axes=ax, show=False,
            contours=6, extrapolate="head",
        )
        rv_pct = rv_vals[i] * 100
        color = "green" if rv_vals[i] < 0.15 else "red"
        ax.set_title(f"Comp {i+1}\nRV={rv_pct:.1f}%", fontsize=10, color=color)
    fig.suptitle("shPLRNN obs_matrix: scalp topographies\n"
                 "(green = RV<15%, dipole-plausible)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rv_bar(rv_vals, out_path):
    n = len(rv_vals)
    fig, ax = plt.subplots(figsize=(max(4, n * 1.2), 3.5))
    colors = ["green" if rv < 0.15 else "red" for rv in rv_vals]
    bars = ax.bar(range(1, n + 1), [rv * 100 for rv in rv_vals], color=colors,
                  edgecolor="black", linewidth=0.8)
    ax.axhline(15, color="black", linestyle="--", linewidth=1.2,
               label="15% RV threshold (Delorme et al. 2012)")
    ax.set_xlabel("Component", fontsize=12)
    ax.set_ylabel("Residual Variance (%)", fontsize=12)
    ax.set_title("Equivalent dipole fit: residual variance per component", fontsize=12)
    ax.set_xticks(range(1, n + 1))
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    for bar, rv in zip(bars, rv_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rv*100:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--out_dir", default="results/dipole_check")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load obs_matrix
    ckpt = torch.load(args.model_path, map_location="cpu")
    obs_matrix = ckpt["obs_matrix"].numpy()  # (dz, dx) = (4, 22)
    n_comp, n_ch = obs_matrix.shape
    print(f"obs_matrix shape: {obs_matrix.shape} (n_comp={n_comp}, n_ch={n_ch})")
    assert n_ch == N_CH, f"Expected {N_CH} channels, got {n_ch}"

    # Build MNE info with standard montage
    info = build_info()

    # Fit dipole to each component
    results = []
    for i in range(n_comp):
        topo = obs_matrix[i].copy()
        gof, rv, pos, ori, amp = fit_dipole_sphere(topo, info)
        plausible = rv < 0.15
        print(f"  Comp {i+1}: GOF={gof:.1f}%  RV={rv*100:.1f}%  "
              f"pos={pos}  {'PASS' if plausible else 'FAIL'}")
        results.append(dict(comp=i+1, gof=gof, rv=rv, pos=pos, ori=ori,
                            amp=amp, plausible=plausible))

    # Save table
    rv_vals = [r["rv"] for r in results]
    table_path = os.path.join(args.out_dir, "rv_table.txt")
    with open(table_path, "w") as f:
        f.write("Component  GOF(%)  RV(%)  x(m)    y(m)    z(m)    Plausible(<15%)\n")
        f.write("-" * 70 + "\n")
        for r in results:
            x, y, z = r["pos"]
            f.write(f"  {r['comp']:5d}    {r['gof']:5.1f}  {r['rv']*100:5.1f}"
                    f"  {x:+.3f}  {y:+.3f}  {z:+.3f}  {'YES' if r['plausible'] else 'NO'}\n")
        n_pass = sum(r["plausible"] for r in results)
        f.write(f"\n{n_pass}/{n_comp} components with RV < 15%\n")
        f.write("Reference: Delorme et al. (2012), doi:10.3389/fnins.2012.00007\n")
    print(f"\nTable saved to {table_path}")
    with open(table_path) as f:
        print(f.read())

    # Plots
    topo_path = os.path.join(args.out_dir, "topomaps.png")
    plot_topomaps(obs_matrix, info, topo_path, rv_vals)
    print(f"Topomaps saved to {topo_path}")

    bar_path = os.path.join(args.out_dir, "rv_bar.png")
    plot_rv_bar(rv_vals, bar_path)
    print(f"RV bar chart saved to {bar_path}")

    n_pass = sum(r["plausible"] for r in results)
    print(f"\nSummary: {n_pass}/{n_comp} components pass dipole criterion (RV < 15%)")


if __name__ == "__main__":
    main()

"""
Biophysical forward-model generator for the failure-axis sweep.

Motivation (rebuttal item #2): the failure-axis results in the paper are driven by
mixing matrices that come from a *learned* shPLRNN observation matrix (or a random
matrix). A reviewer can object that the failure pattern is an artifact of that one
learned generator. This script swaps in a mixing family with a completely different
inductive bias — a physics-based EEG leadfield (MNE 3-layer sphere model, K discrete
current dipoles, no training at all) — while keeping the *source dynamics* identical
(`make_sources` from failure_axis_sweep). Only the mixing family changes, so it is a
clean ablation.

Because the number of dipoles K is a free knob (no retraining), the same script also
covers the M >> D regime (K up to 128 with D = 22 sensors), which the learned
generator could not reach without retraining per source count.

Usage:
    # smoke test (1 seed, few configs)
    python scripts/forward_model_generator.py --quick

    # full 5-axis sweep under the forward-model generator
    python scripts/forward_model_generator.py --mode axes

    # sources-vs-sensors sweep including M >> D
    python scripts/forward_model_generator.py --mode mvsd

Output:
    results/forward_model/all_results.npy   — same dict layout as failure_axis_sweep
    results/forward_model/axis_*.png
    results/forward_model/leadfield.png     — leadfield topographies + gain spectrum
"""

import argparse
import os
import sys
import warnings

import numpy as np
# numpy>=2 on Apple Accelerate/OpenBLAS emits spurious divide/overflow/invalid
# RuntimeWarnings from matmul on finite inputs (verified: results are finite).
np.seterr(divide="ignore", over="ignore", invalid="ignore")
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from scipy.optimize import linear_sum_assignment

import mne
mne.set_log_level("WARNING")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from failure_axis_sweep import (      # noqa: E402
    METHODS, D, T, N_SEG,
    make_sources, make_segment_labels, mix, add_noise,
    fit_tcl, fit_ivae, plot_axis,
)
from dipole_check import build_info   # noqa: E402

RESULTS_DIR = "./results/forward_model"


# ── leadfield ────────────────────────────────────────────────────────────────

def sample_dipoles(K, head_radius, r0, rng, r_min_frac=0.25, r_max_frac=0.80):
    """Sample K dipole positions inside the brain compartment and unit orientations.

    Positions are uniform in volume between r_min_frac and r_max_frac of the head
    radius (the sphere model's brain layer ends at 0.90 * head_radius; staying below
    0.80 keeps every dipole safely inside it). Orientations are uniform on the sphere.
    """
    u = rng.uniform(r_min_frac ** 3, r_max_frac ** 3, K) ** (1.0 / 3.0)
    radii = u * head_radius
    dirs = rng.standard_normal((K, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rr = r0[None, :] + radii[:, None] * dirs

    nn = rng.standard_normal((K, 3))
    nn /= np.linalg.norm(nn, axis=1, keepdims=True)
    return rr, nn


def build_leadfield(K, seed=0, row_normalize=False, return_meta=False):
    """Physics-based EEG leadfield for K dipoles and the 22 BCI IV-2a channels.

    Returns L with shape (K, D) so it drops straight into `mix(z, W)`, which computes
    x = z @ W with z of shape (T, K). This is the transpose of the usual (D, K)
    leadfield convention.

    Scaling: by default the whole matrix is divided by the mean row norm. This keeps
    the *relative* row scales — i.e. the depth-dependent gain falloff, which is a real
    property of volume conduction and part of what makes the physics generator
    different from a random matrix. `row_normalize=True` instead forces every row to
    unit norm, matching the random-W baseline exactly (used as a control).
    """
    rng = np.random.default_rng(seed)
    info = build_info()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info,
                                       verbose=False)
    r0 = np.asarray(sphere["r0"], dtype=float)
    head_radius = float(sphere["layers"][-1]["rad"])

    rr, nn = sample_dipoles(K, head_radius, r0, rng)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        src = mne.setup_volume_source_space(
            pos=dict(rr=rr, nn=nn), sphere_units="m", verbose=False)
        fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere,
                                        eeg=True, meg=False, verbose=False)
        fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=False,
                                           verbose=False)

    L_DK = fwd["sol"]["data"]          # (D, K), fixed orientation = nn
    assert L_DK.shape == (D, K), f"leadfield shape {L_DK.shape}, expected {(D, K)}"
    L = L_DK.T.astype(np.float32)      # (K, D)

    row_norms = np.linalg.norm(L, axis=1, keepdims=True)
    if row_normalize:
        L = L / (row_norms + 1e-12)
    else:
        L = L / (row_norms.mean() + 1e-12)

    if return_meta:
        depths = np.linalg.norm(rr - r0[None, :], axis=1)
        return L, dict(info=info, rr=rr, nn=nn, depths=depths,
                       row_norms=row_norms.ravel(), r0=r0, head_radius=head_radius)
    return L


# ── MCC that tolerates K != n_components ─────────────────────────────────────

def mcc_rect(s_true, s_hat):
    """MCC over the *true* sources: matched |corr| summed, divided by K.

    When K == n_components this is identical to failure_axis_sweep.mcc. When K > D the
    methods can only return D components, and the K - D unrecoverable sources score 0
    — which is the honest accounting for the M >> D regime (taking the mean over the
    matched subset only would hide the missing sources).
    """
    K = s_true.shape[1]
    C = np.abs(np.corrcoef(s_true.T, s_hat.T)[:K, K:])   # (K, n)
    C = np.nan_to_num(C)
    row, col = linear_sum_assignment(-C)
    return C[row, col].sum() / K


# ── one cell ─────────────────────────────────────────────────────────────────

def run_cell(z, x, labels, device="cpu", epochs=200):
    """PCA / FastICA / TCL / iVAE on x, scored against z. n_comp = min(K, D)."""
    K = z.shape[1]
    n_comp = min(K, x.shape[1])
    results = {}

    s_pca = PCA(n_components=n_comp).fit_transform(x)
    results["PCA"] = mcc_rect(z, s_pca)

    try:
        s_ica = FastICA(n_components=n_comp, max_iter=500,
                        random_state=0).fit_transform(x)
        results["FastICA"] = mcc_rect(z, s_ica)
    except Exception:
        results["FastICA"] = np.nan

    s_tcl = fit_tcl(x, labels, latent_dim=n_comp, epochs=epochs, device=device)
    results["TCL"] = mcc_rect(z, s_tcl)

    s_ivae = fit_ivae(x, labels, latent_dim=n_comp, epochs=epochs, device=device)
    results["iVAE"] = mcc_rect(z, s_ivae)

    return results


def sweep_axis(name, configs, make_data_fn, L, seeds, device, epochs):
    """Same contract as failure_axis_sweep.sweep_axis, with seeds/epochs exposed."""
    print(f"\n{'='*60}")
    print(f"Axis: {name}")
    results = {m: {c: [] for c in configs} for m in METHODS}

    for cfg in configs:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            z, x, labels = make_data_fn(cfg, rng, L)
            cell = run_cell(z, x, labels, device=device, epochs=epochs)
            for m in METHODS:
                results[m][cfg].append(cell.get(m, np.nan))
        line = " | ".join(f"{m}={np.nanmean(results[m][cfg]):.3f}" for m in METHODS)
        print(f"  {name}={cfg}: {line}")

    return results


# ── leadfield diagnostics ────────────────────────────────────────────────────

def plot_leadfield(L, meta, out_path):
    """Topographies of the first few leadfield rows + gain/conditioning summary."""
    info = meta["info"]
    n_show = min(4, L.shape[0])
    fig, axes = plt.subplots(1, n_show + 2, figsize=(3 * (n_show + 2), 3.2))
    for i in range(n_show):
        mne.viz.plot_topomap(L[i], info, axes=axes[i], show=False,
                             contours=6, extrapolate="head")
        axes[i].set_title(f"dipole {i+1}\ndepth={meta['depths'][i]*100:.1f} cm",
                          fontsize=9)

    ax = axes[n_show]
    ax.scatter(meta["depths"] * 100, meta["row_norms"], s=12)
    ax.set_xlabel("dipole eccentricity (cm)")
    ax.set_ylabel("row norm of L")
    ax.set_title("depth-dependent gain", fontsize=9)

    ax = axes[n_show + 1]
    sv = np.linalg.svd(L, compute_uv=False)
    ax.semilogy(np.arange(1, len(sv) + 1), sv, "o-", ms=3)
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.set_title(f"spectrum (cond={sv[0]/sv[-1]:.1e})", fontsize=9)

    fig.suptitle(f"MNE sphere forward model, K={L.shape[0]} dipoles, D={L.shape[1]} sensors",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── axes ─────────────────────────────────────────────────────────────────────

def run_axes(L, seeds, device, epochs, quick, out_dir):
    """The 5 failure axes of failure_axis_sweep, with L as the mixing matrix."""
    M = L.shape[0]
    all_results = {}

    snr_configs = [30, 10, 0] if quick else [30, 15, 10, 5, 0]

    def make_snr(snr_db, rng, W):
        z = make_sources(T, M, N_SEG, coupling=0.0, ns_mode="source_aligned", rng=rng)
        x = mix(z, W, nonlin="linear")
        x = add_noise(x, snr_db, noise_type="pink", rng=rng)
        return z, x, make_segment_labels(T, N_SEG)

    res = sweep_axis("1/f_SNR_dB", snr_configs, make_snr, L, seeds, device, epochs)
    all_results["1f_snr"] = res
    plot_axis("1/f SNR (forward model)", snr_configs, res,
              "SNR (dB, higher = cleaner)", f"{out_dir}/axis_snr.png")

    nonlin_configs = ["linear", "cube", "tanh"]

    def make_nonlin(nl, rng, W):
        z = make_sources(T, M, N_SEG, coupling=0.0, ns_mode="source_aligned", rng=rng)
        x = mix(z, W, nonlin=nl)
        x = add_noise(x, 20, noise_type="gaussian", rng=rng)
        return z, x, make_segment_labels(T, N_SEG)

    res = sweep_axis("Mixing_nonlinearity", nonlin_configs, make_nonlin, L, seeds,
                     device, epochs)
    all_results["nonlin"] = res
    plot_axis("Mixing nonlinearity (forward model)", nonlin_configs, res,
              "Nonlinearity type", f"{out_dir}/axis_nonlinearity.png")

    coupling_configs = [0.0, 0.3, 0.9] if quick else [0.0, 0.1, 0.3, 0.6, 0.9]

    def make_coupling(coup, rng, W):
        z = make_sources(T, M, N_SEG, coupling=coup, ns_mode="source_aligned", rng=rng)
        x = mix(z, W, nonlin="linear")
        x = add_noise(x, 20, noise_type="gaussian", rng=rng)
        return z, x, make_segment_labels(T, N_SEG)

    res = sweep_axis("Source_coupling", coupling_configs, make_coupling, L, seeds,
                     device, epochs)
    all_results["coupling"] = res
    plot_axis("Source coupling (forward model)", coupling_configs, res,
              "Coupling strength (0=independent)", f"{out_dir}/axis_coupling.png")

    ns_configs = ["stationary", "source_aligned", "confounder_aligned"]

    def make_ns(ns_mode, rng, W):
        z = make_sources(T, M, N_SEG, coupling=0.0, ns_mode=ns_mode, rng=rng)
        x = mix(z, W, nonlin="linear")
        x = add_noise(x, 15, noise_type="pink", rng=rng)
        return z, x, make_segment_labels(T, N_SEG)

    res = sweep_axis("NS_informativeness", ns_configs, make_ns, L, seeds, device, epochs)
    all_results["ns"] = res
    plot_axis("Nonstationarity type (forward model)", ns_configs, res, "NS mode",
              f"{out_dir}/axis_ns.png")

    return all_results


def run_mvsd(k_list, seeds, device, epochs, row_normalize, out_dir):
    """Sources-vs-sensors axis: a fresh K-dipole leadfield per K, D=22 fixed.

    No retraining is involved at any K — the generator is the same physics for every
    source count, which is exactly the objection this answers.
    """
    print(f"\n{'='*60}")
    print("Axis: sources vs sensors (forward model, D=22 fixed)")
    results = {m: {k: [] for k in k_list} for m in METHODS}

    for K in k_list:
        for seed in seeds:
            L = build_leadfield(K, seed=seed, row_normalize=row_normalize)
            rng = np.random.default_rng(seed)
            z = make_sources(T, K, N_SEG, coupling=0.0, ns_mode="source_aligned",
                             rng=rng)
            x = mix(z, L, nonlin="linear")
            x = add_noise(x, 20, noise_type="gaussian", rng=rng)
            labels = make_segment_labels(T, N_SEG)
            cell = run_cell(z, x, labels, device=device, epochs=epochs)
            for m in METHODS:
                results[m][K].append(cell.get(m, np.nan))
        line = " | ".join(f"{m}={np.nanmean(results[m][K]):.3f}" for m in METHODS)
        print(f"  K={K}: {line}")

    plot_axis("Sources vs sensors (forward model)", k_list, results,
              "K (dipoles, D=22 fixed)", f"{out_dir}/axis_m_vs_d.png")
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["axes", "mvsd", "both"], default="both")
    p.add_argument("--K", type=int, default=4,
                   help="dipoles for the 5 failure axes (matches M=4 in the paper)")
    p.add_argument("--k_list", type=int, nargs="+",
                   default=[2, 4, 8, 16, 22, 32, 64, 128],
                   help="source counts for the sources-vs-sensors axis")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=200,
                   help="TCL/iVAE training epochs per cell")
    p.add_argument("--row_normalize", action="store_true",
                   help="force unit-norm leadfield rows (control: removes depth gain)")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 1 seed, fewer configs, 20 epochs")
    p.add_argument("--out_dir", type=str, default=RESULTS_DIR)
    p.add_argument("--device", type=str,
                   default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    seeds, epochs, k_list = args.seeds, args.epochs, args.k_list
    if args.quick:
        seeds = seeds[:1]
        epochs = 20
        k_list = [4, 22, 64]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {args.device} | seeds: {seeds} | epochs: {epochs}")

    all_results = {}

    if args.mode in ("axes", "both"):
        L, meta = build_leadfield(args.K, seed=0, row_normalize=args.row_normalize,
                                  return_meta=True)
        print(f"Leadfield: {L.shape} (K={args.K} dipoles, D={D} sensors), "
              f"cond={np.linalg.cond(L):.2e}")
        plot_leadfield(L, meta, f"{args.out_dir}/leadfield.png")
        all_results.update(run_axes(L, seeds, args.device, epochs, args.quick,
                                    args.out_dir))

    if args.mode in ("mvsd", "both"):
        all_results["m_vs_d"] = run_mvsd(k_list, seeds, args.device, epochs,
                                         args.row_normalize, args.out_dir)

    out = f"{args.out_dir}/all_results.npy"
    np.save(out, all_results)
    print(f"\nSaved raw results to {out}")


if __name__ == "__main__":
    main()

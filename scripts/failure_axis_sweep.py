"""
Failure-axis sweep for NeurIPS position paper:
"Identifiability for Brain Dynamics is Untestable Without Intervention"

For each of 5 failure axes, we:
  1. Generate M=4 synthetic sources with controlled nonstationarity
  2. Mix them (using shPLRNN's learned W or a random W)
  3. Add axis-specific confounds (1/f noise, nonlinearity, coupling, etc.)
  4. Run PCA, FastICA, TCL, iVAE on the observed mixtures
  5. Compute MCC between recovered and true sources

Usage:
    python scripts/failure_axis_sweep.py --model_path trained_models/bci/bci/linear_M4/run_1
    python scripts/failure_axis_sweep.py --random_W   # skip model, use random W
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "./results/failure_axis"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── reproducibility ──────────────────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4]
M = 4       # number of sources (latent_size)
D = 22      # number of channels (obs_size)
T = 2000    # timesteps per trial
N_SEG = 20  # segments for TCL nonstationarity

# ── MCC ─────────────────────────────────────────────────────────────────────

def mcc(s_true, s_hat):
    """Mean correlation coefficient with optimal permutation matching.
    s_true, s_hat: (T, M) numpy arrays.
    Returns scalar MCC in [0, 1].
    """
    M_ = s_true.shape[1]
    C = np.abs(np.corrcoef(s_true.T, s_hat.T)[:M_, M_:])   # (M, M)
    row, col = linear_sum_assignment(-C)
    return C[row, col].mean()


# ── pink noise ───────────────────────────────────────────────────────────────

def pink_noise(shape, rng):
    """Generate 1/f (pink) noise via spectral shaping."""
    T_, D_ = shape
    freqs = np.fft.rfftfreq(T_)
    freqs[0] = 1.0   # avoid div-zero
    psd = 1.0 / freqs
    psd[0] = 0.0
    white = rng.standard_normal((T_, D_)) + 1j * rng.standard_normal((T_, D_))
    shaped = np.fft.rfft(rng.standard_normal((T_, D_)), axis=0) * np.sqrt(psd[:, None])
    return np.fft.irfft(shaped, n=T_, axis=0).astype(np.float32)


# ── source generation ────────────────────────────────────────────────────────

def make_sources(T, M, N_seg, coupling=0.0, ns_mode="source_aligned", rng=None):
    """Generate M nonstationary sources of length T with N_seg segments.

    coupling: strength of inter-source coupling (0 = independent)
    ns_mode: 'source_aligned' | 'confounder_aligned' | 'stationary'
    """
    if rng is None:
        rng = np.random.default_rng(0)
    seg_len = T // N_seg
    z = np.zeros((T, M), dtype=np.float32)

    if ns_mode == "stationary":
        z = rng.standard_normal((T, M)).astype(np.float32)
    elif ns_mode == "source_aligned":
        # Each segment has independent per-source variances → TCL can discriminate
        for seg in range(N_seg):
            t0, t1 = seg * seg_len, min((seg + 1) * seg_len, T)
            lambdas = rng.exponential(1.0, M)
            z[t0:t1] = rng.standard_normal((t1 - t0, M)) * np.sqrt(lambdas)
    elif ns_mode == "confounder_aligned":
        # All sources share a single segment-specific global variance (TCL grabs this
        # global confound instead of source-specific patterns)
        base = rng.standard_normal((T, M)).astype(np.float32)
        for seg in range(N_seg):
            t0, t1 = seg * seg_len, min((seg + 1) * seg_len, T)
            global_scale = rng.exponential(1.0)
            base[t0:t1] *= global_scale
        z = base

    if coupling > 0.0:
        # Coupled AR(1): z_t = (1-coupling)*z_t + coupling * A @ z_{t-1}
        A = rng.standard_normal((M, M)).astype(np.float32)
        A /= (np.linalg.norm(A, 2) + 1e-6)   # spectral norm 1
        for t in range(1, T):
            z[t] = (1 - coupling) * z[t] + coupling * (A @ z[t - 1])

    return z   # (T, M)


def make_segment_labels(T, N_seg):
    seg_len = T // N_seg
    labels = np.zeros(T, dtype=np.int64)
    for seg in range(N_seg):
        t0, t1 = seg * seg_len, min((seg + 1) * seg_len, T)
        labels[t0:t1] = seg
    return labels


# ── mixing ───────────────────────────────────────────────────────────────────

def mix(z, W, nonlin="linear"):
    """x = z @ W (z: T×M, W: M×D → x: T×D), then apply nonlinearity."""
    x = z @ W   # (T, D)
    if nonlin == "tanh":
        x = np.tanh(x)
    elif nonlin == "cube":
        # mild polynomial nonlinearity
        x = x + 0.1 * x ** 3
    return x.astype(np.float32)


def add_noise(x, snr_db, noise_type="gaussian", rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    sig_pow = np.mean(x ** 2)
    noise_pow = sig_pow * 10 ** (-snr_db / 10.0)
    if noise_type == "pink":
        n = pink_noise(x.shape, rng) * np.sqrt(noise_pow / (np.mean(pink_noise(x.shape, rng) ** 2) + 1e-12))
    else:
        n = rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(noise_pow)
    return x + n


# ── TCL ──────────────────────────────────────────────────────────────────────

class TCLNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_segments):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.classifier = nn.Linear(latent_dim, n_segments)

    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h), h


def fit_tcl(x, labels, latent_dim, hidden_dim=64, epochs=200, lr=1e-3, device="cpu"):
    """Fit TCL and return recovered sources (T, latent_dim)."""
    n_seg = int(labels.max()) + 1
    X = torch.tensor(x, dtype=torch.float32, device=device)
    Y = torch.tensor(labels, dtype=torch.long, device=device)

    model = TCLNet(x.shape[1], hidden_dim, latent_dim, n_seg).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device)
        for i in range(0, len(X), 256):
            idx = perm[i:i+256]
            logits, _ = model(X[idx])
            loss = loss_fn(logits, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        _, h = model(X)
    return h.cpu().numpy()


# ── iVAE ─────────────────────────────────────────────────────────────────────

class iVAENet(nn.Module):
    def __init__(self, data_dim, latent_dim, aux_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(data_dim + aux_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.prior_net = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, data_dim),
        )

    def reparameterize(self, mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    def forward(self, x, u):
        enc = self.encoder(torch.cat([x, u], dim=-1))
        z_mu, z_lv = enc.chunk(2, dim=-1)
        p = self.prior_net(u)
        p_mu, p_lv = p.chunk(2, dim=-1)
        z = self.reparameterize(z_mu, z_lv)
        x_hat = self.decoder(z)
        return x_hat, z, z_mu, z_lv, p_mu, p_lv


def fit_ivae(x, labels, latent_dim, hidden_dim=64, epochs=200, lr=1e-3, device="cpu"):
    n_seg = int(labels.max()) + 1
    X = torch.tensor(x, dtype=torch.float32, device=device)
    u_onehot = torch.zeros(len(x), n_seg, device=device)
    u_onehot[torch.arange(len(x)), torch.tensor(labels, device=device)] = 1.0

    model = iVAENet(x.shape[1], latent_dim, n_seg, hidden_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device)
        for i in range(0, len(X), 256):
            idx = perm[i:i+256]
            x_hat, z, z_mu, z_lv, p_mu, p_lv = model(X[idx], u_onehot[idx])
            # reconstruction
            recon = ((x_hat - X[idx]) ** 2).mean()
            # KL(q(z|x,u) || p(z|u))
            kl = -0.5 * (1 + z_lv - p_lv
                         - (z_mu - p_mu).pow(2) / p_lv.exp()
                         - z_lv.exp() / p_lv.exp()).sum(dim=-1).mean()
            loss = recon + kl
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        _, z, z_mu, _, _, _ = model(X, u_onehot)
    return z_mu.cpu().numpy()


# ── run one cell ─────────────────────────────────────────────────────────────

def run_cell(z, x, labels, device="cpu", n_comp=None):
    """Run all 4 methods on observed x given true z. Returns dict of MCCs."""
    if n_comp is None:
        n_comp = z.shape[1]
    results = {}

    # PCA
    s_pca = PCA(n_components=n_comp).fit_transform(x)
    results["PCA"] = mcc(z, s_pca)

    # FastICA
    try:
        s_ica = FastICA(n_components=n_comp, max_iter=500, random_state=0).fit_transform(x)
        results["FastICA"] = mcc(z, s_ica)
    except Exception:
        results["FastICA"] = np.nan

    # TCL
    s_tcl = fit_tcl(x, labels, latent_dim=n_comp, device=device)
    results["TCL"] = mcc(z, s_tcl)

    # iVAE
    s_ivae = fit_ivae(x, labels, latent_dim=n_comp, device=device)
    results["iVAE"] = mcc(z, s_ivae)

    return results


# ── failure axes ─────────────────────────────────────────────────────────────

METHODS = ["PCA", "FastICA", "TCL", "iVAE"]


def sweep_axis(name, configs, make_data_fn, W, device):
    """Run the sweep for one failure axis.
    make_data_fn(config, seed) → (z, x, labels)
    """
    print(f"\n{'='*60}")
    print(f"Axis: {name}")
    results = {m: {c: [] for c in configs} for m in METHODS}

    for cfg in configs:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            z, x, labels = make_data_fn(cfg, rng, W)
            cell = run_cell(z, x, labels, device=device)
            for m in METHODS:
                results[m][cfg].append(cell.get(m, np.nan))
        line = " | ".join(f"{m}={np.nanmean(results[m][cfg]):.3f}" for m in METHODS)
        print(f"  {name}={cfg}: {line}")

    return results


def plot_axis(name, configs, results, xlabel, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"PCA": "#1f77b4", "FastICA": "#ff7f0e", "TCL": "#2ca02c", "iVAE": "#d62728"}
    x_pos = np.arange(len(configs))
    for m in METHODS:
        means = [np.nanmean(results[m][c]) for c in configs]
        stds  = [np.nanstd(results[m][c]) for c in configs]
        ax.plot(x_pos, means, "o-", label=m, color=colors[m])
        ax.fill_between(x_pos,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.15, color=colors[m])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(c) for c in configs], rotation=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MCC")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def get_W(model_path, device):
    """Load trained shPLRNN and return obs_matrix as numpy (M, D)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from model import shallowPLRNN

    # find latest checkpoint
    ckpt_dir = model_path
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt") and "model" in f])
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    ckpt = os.path.join(ckpt_dir, ckpts[-1])
    print(f"Loading checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    W = state["obs_matrix"].numpy()    # (M, D) = (4, 22)
    print(f"obs_matrix shape: {W.shape}")
    return W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model checkpoint directory")
    parser.add_argument("--random_W", action="store_true",
                        help="Use a random orthogonal W instead of trained model")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")

    # Load or generate mixing matrix W (M, D)
    if args.random_W or args.model_path is None:
        rng_W = np.random.default_rng(42)
        W_rand = rng_W.standard_normal((M, D)).astype(np.float32)
        W_rand /= np.linalg.norm(W_rand, axis=1, keepdims=True)
        W = W_rand
        print("Using random W")
    else:
        W = get_W(args.model_path, device)

    all_results = {}

    # ─── Axis 1: 1/f SNR ────────────────────────────────────────────────────
    snr_configs = [30, 15, 10, 5, 0]   # dB; 0 dB = noise power = signal power

    def make_snr(snr_db, rng, W):
        z = make_sources(T, M, N_SEG, coupling=0.0, ns_mode="source_aligned", rng=rng)
        x = mix(z, W, nonlin="linear")
        x = add_noise(x, snr_db, noise_type="pink", rng=rng)
        labels = make_segment_labels(T, N_SEG)
        return z, x, labels

    res_snr = sweep_axis("1/f_SNR_dB", snr_configs, make_snr, W, device)
    all_results["1f_snr"] = res_snr
    plot_axis("1/f SNR", snr_configs, res_snr, "SNR (dB, higher = cleaner)",
              f"{RESULTS_DIR}/axis_snr.png")

    # ─── Axis 2: Mixing nonlinearity ─────────────────────────────────────────
    nonlin_configs = ["linear", "cube", "tanh"]

    def make_nonlin(nl, rng, W):
        z = make_sources(T, M, N_SEG, coupling=0.0, ns_mode="source_aligned", rng=rng)
        x = mix(z, W, nonlin=nl)
        x = add_noise(x, 20, noise_type="gaussian", rng=rng)
        labels = make_segment_labels(T, N_SEG)
        return z, x, labels

    res_nl = sweep_axis("Mixing_nonlinearity", nonlin_configs, make_nonlin, W, device)
    all_results["nonlin"] = res_nl
    plot_axis("Mixing nonlinearity", nonlin_configs, res_nl, "Nonlinearity type",
              f"{RESULTS_DIR}/axis_nonlinearity.png")

    # ─── Axis 3: Source coupling ──────────────────────────────────────────────
    coupling_configs = [0.0, 0.1, 0.3, 0.6, 0.9]

    def make_coupling(coup, rng, W):
        z = make_sources(T, M, N_SEG, coupling=coup, ns_mode="source_aligned", rng=rng)
        x = mix(z, W, nonlin="linear")
        x = add_noise(x, 20, noise_type="gaussian", rng=rng)
        labels = make_segment_labels(T, N_SEG)
        return z, x, labels

    res_coup = sweep_axis("Source_coupling", coupling_configs, make_coupling, W, device)
    all_results["coupling"] = res_coup
    plot_axis("Source coupling", coupling_configs, res_coup, "Coupling strength (0=independent)",
              f"{RESULTS_DIR}/axis_coupling.png")

    # ─── Axis 4: Nonstationarity informativeness ──────────────────────────────
    ns_configs = ["stationary", "source_aligned", "confounder_aligned"]

    def make_ns(ns_mode, rng, W):
        z = make_sources(T, M, N_SEG, coupling=0.0, ns_mode=ns_mode, rng=rng)
        x = mix(z, W, nonlin="linear")
        x = add_noise(x, 15, noise_type="pink", rng=rng)
        labels = make_segment_labels(T, N_SEG)
        return z, x, labels

    res_ns = sweep_axis("NS_informativeness", ns_configs, make_ns, W, device)
    all_results["ns"] = res_ns
    plot_axis("Nonstationarity type", ns_configs, res_ns, "NS mode",
              f"{RESULTS_DIR}/axis_ns.png")

    # ─── Axis 5: Sources vs sensors ───────────────────────────────────────────
    # Vary M while keeping D=22 fixed
    M_configs = [2, 4, 8, 16, 22]

    def make_m_vs_d(m_src, rng, W_unused):
        # Generate a fresh W for this M
        W_m = rng.standard_normal((m_src, D)).astype(np.float32)
        W_m /= np.linalg.norm(W_m, axis=1, keepdims=True)
        z = make_sources(T, m_src, N_SEG, coupling=0.0, ns_mode="source_aligned", rng=rng)
        x = mix(z, W_m, nonlin="linear")
        x = add_noise(x, 20, noise_type="gaussian", rng=rng)
        labels = make_segment_labels(T, N_SEG)
        return z, x, labels

    m_results = {m: {cfg: [] for cfg in M_configs} for m in METHODS}
    for m_src in M_configs:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            z, x, labels = make_m_vs_d(m_src, rng, W)
            cell = run_cell(z, x, labels, device=device, n_comp=m_src)
            for meth in METHODS:
                m_results[meth][m_src].append(cell.get(meth, np.nan))
        line = " | ".join(f"{meth}={np.nanmean(m_results[meth][m_src]):.3f}" for meth in METHODS)
        print(f"  M={m_src}: {line}")
    all_results["m_vs_d"] = m_results
    plot_axis("Sources vs Sensors", M_configs, m_results, "M (sources, D=22 fixed)",
              f"{RESULTS_DIR}/axis_m_vs_d.png")

    # ─── Save raw results ─────────────────────────────────────────────────────
    np.save(f"{RESULTS_DIR}/all_results.npy", all_results)
    print(f"\nAll results saved to {RESULTS_DIR}/")

    # ─── Summary table ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY (mean MCC across seeds, realistic EEG-like config)")
    print("="*60)
    print(f"{'Axis':<28} {'PCA':>6} {'ICA':>6} {'TCL':>6} {'iVAE':>6}")
    print("-"*60)
    summaries = [
        ("1/f SNR @ 0 dB",   res_snr,  0),
        ("Nonlin=tanh",       res_nl,   "tanh"),
        ("Coupling=0.6",      res_coup, 0.6),
        ("NS=confounder",     res_ns,   "confounder_aligned"),
        ("M=22 (=D)",         m_results, 22),
    ]
    for label, res, cfg in summaries:
        row = [np.nanmean(res[m][cfg]) for m in METHODS]
        print(f"{label:<28} {row[0]:>6.3f} {row[1]:>6.3f} {row[2]:>6.3f} {row[3]:>6.3f}")


if __name__ == "__main__":
    main()

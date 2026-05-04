# EEG Reproduction & Improvement Plan

## What we reproduced (Section 4.4, Andrzejak 2001)

Binary healthy-vs-ictal classification via unsupervised hierarchical DSR:

| Script | Purpose |
|---|---|
| `scripts/prepare_eeg_data.py` | Download Andrzejak EEG dataset, z-score, save as `data/eeg/data.pt` |
| `scripts/run_eeg.sh` | Train 10 runs of hierarchical shPLRNN, save to `trained_models/eeg/` |
| `scripts/eval_eeg_clustering.py` | Extract p-vectors, fit 2-GMM, report clustering accuracy |
| `scripts/eeg_baselines.py` | Catch-22, tsfresh, ROCKET, MiniRocket baselines for Table 7 |

**Target:** 92.6 ± 1.2% clustering accuracy.

**Hyperparameters used:**
- obs_size=1, latent_size=1, hidden_size=20 (L=20M default)
- num_individual_params=2 (2D feature space → clean 2D scatter plot)
- lr_group=1e-4, lr_individual=1e-3 (paper Sec. 3.2)
- 5000 epochs, seq_len=500, 10 independent runs

---

## Open questions / things to verify

- [ ] Confirm whether Zhang et al. (2022) refers to a preprocessed version of the Andrzejak data or a different dataset entirely — check full paper PDF.
- [ ] Check whether the paper uses Z+O (200 healthy) vs S (100 ictal) → 300 subjects, or just Z vs S → 200. Adjust `prepare_eeg_data.py` accordingly.
- [ ] Confirm `num_individual_params` used in the paper (not stated explicitly; try 2, 3, 4 and compare).
- [ ] Verify `seq_len` used — paper mentions Tmax=1000 in some experiments; try 500 and 1000.

---

## TODO: Improvements for later

### Data & preprocessing
- [ ] **Multi-class EEG**: Extend to all 5 Andrzejak classes (Z, O, N, F, S) with a 5-component GMM — richer test of unsupervised structure discovery.
- [ ] **Multi-channel EEG**: Use a dataset with multiple EEG channels (e.g., TUH EEG Corpus or CHB-MIT scalp EEG) to stress-test the model at obs_size > 1.
- [ ] **Bandpass filtering**: Add 1–40 Hz bandpass filter before z-scoring; this is standard preprocessing that may help the DSR focus on relevant dynamics.
- [ ] **Longer segments**: Test with full 4096-sample segments (23.6 s) vs. shorter windows to assess sensitivity to segment length.

### Model architecture
- [ ] **Multi-channel observation model**: When obs_size > 1, the identity obs_model forces latent_size=obs_size=N_channels. A learned linear obs_model (obs_model=linear) with latent_size < obs_size would allow dimensionality reduction — potentially better for high-channel EEG.
- [ ] **Higher latent_size**: Try latent_size=3–5 even for single-channel EEG (overcomplete latent space); the extra latent dims might capture nonlinear structure better than M=1.
- [ ] **Learnable noise covariance**: Enable `--learn_noise_cov` flag — may help on noisy real EEG data where noise is non-stationary across subjects.
- [ ] **Clipped PLRNN**: Try `--clipped` variant; clipping may regularize epileptic segments that have high-amplitude bursts.

### Training & optimization
- [ ] **Gradient clipping sensitivity**: Systematic sweep of `clip_grad_norm` (0, 1, 5, 10) — epileptic ictal segments can cause gradient spikes.
- [ ] **GTF alpha schedule**: Compare fixed alpha (alpha_start = alpha_end) vs. annealing from 0.9→0 — the paper's GTF may be especially important for the fast dynamics of ictal EEG.
- [ ] **Hierarchical regularization (lam > 0)**: The `--lam` flag adds a hierarchical loss term; sweep lam ∈ {0, 0.01, 0.1} to see if it compresses the p-vector space and improves cluster separation.

### Evaluation
- [ ] **Silhouette score**: In addition to GMM accuracy, compute silhouette score in p-vector space as a label-free clustering quality measure.
- [ ] **PCA visualization**: Plot 2D PCA of p-vectors colored by true label — reproduces the paper's Figure showing "distinct clusters clearly separated along the first PC."
- [ ] **t-SNE / UMAP visualization**: Complement PCA with t-SNE/UMAP for a richer view of the feature manifold.
- [ ] **Varying GMM components**: Fit GMMs with k=2..5 components; use BIC/AIC to check whether the model recovers >2 natural clusters (Andrzejak has 5 classes).
- [ ] **Fixed-point analysis (SCYFI)**: Run `eval/scyfi.py` on the trained EEG models — fixed points in the 1D latent space should differ qualitatively between healthy and ictal dynamics.

### Baselines (Table 7 completion)
- [ ] **Attention-based convolutional autoencoder**: Not yet implemented in `eeg_baselines.py` — requires a small PyTorch model trained on the EEG data.
- [ ] **Supervised upper bound**: Train a simple 1D CNN or ROCKET + SVM with true labels to get a supervised ceiling for comparison.

### Infrastructure
- [ ] **SLURM job script**: Write `scripts/eeg_slurm.sh` to submit 10 runs in a SLURM array on Mila (`--array=1-10`, L40S nodes, 1 GPU each).
- [ ] **WandB logging**: Replace/augment TensorBoard with WandB for easier tracking of 10-run sweeps and hyperparameter comparisons.
- [ ] **ubermain.py integration**: Add EEG config to `ubermain.py` for grid-search over (num_individual_params × seq_len × lam) in one command.

"""
Download and prepare BCI Competition IV-2a dataset for multi-channel shPLRNN training.

Dataset: 9 subjects, 22 EEG channels, motor imagery (4-class: left/right hand, feet, tongue).
Source: https://bnci-horizon-2020.eu/database/data-sets/001-2014/

Output:
  data/bci/data.pt    — shape (N_subjects * N_trials, T, 22), z-scored per-channel per-subject
  data/bci/labels.pt  — shape (N_subjects * N_trials,), class labels 1-4
  data/bci/meta.pt    — dict with subject_ids, trial_lengths, channel_names

The model trains across subjects, learning shared dynamics with per-subject p-vectors.
Each trial (T=1000 samples @ 250 Hz = 4 s) becomes one "subject" in the hierarchical model.
"""

import os
import urllib.request
import numpy as np
import torch
import scipy.io

DATA_DIR = "./data/bci"
os.makedirs(DATA_DIR, exist_ok=True)

# BCI IV-2a subjects: A01..A09 (training set T files)
SUBJECTS = [f"A{i:02d}T" for i in range(1, 10)]
BASE_URL = "https://bnci-horizon-2020.eu/database/data-sets/001-2014/"

# EEG channels only (first 22 of 25 channels; last 3 are EOG)
N_EEG_CH = 22
# Sampling rate: 250 Hz; trial window: [0.5s, 4.5s] post-cue → 1000 samples
TRIAL_START = 125   # 0.5 s * 250 Hz
TRIAL_LEN = 1000    # 4 s * 250 Hz
TARGET_T = 750      # crop to 3 s for speed while keeping enough temporal context


def download(subject):
    fname = f"{DATA_DIR}/{subject}.mat"
    if os.path.exists(fname):
        print(f"  {subject}.mat already exists, skipping download.")
        return fname
    url = f"{BASE_URL}{subject}.mat"
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, fname)
    return fname


def load_trials(fname):
    """Load EEG trials from a BCI IV-2a .mat file.
    Returns:
        X: (n_trials, T, 22) float32
        y: (n_trials,) int labels 1-4
    """
    mat = scipy.io.loadmat(fname, squeeze_me=True, struct_as_record=False)
    data = mat["data"]

    trials_X, trials_y = [], []
    for run in data:
        try:
            X_run = run.X.astype(np.float32)       # (T_run, 25)
            trial_pos = run.trial.astype(int)       # onset samples
            y_run = run.y.astype(int)               # labels 1-4
        except AttributeError:
            continue
        for onset, label in zip(trial_pos, y_run):
            seg = X_run[onset + TRIAL_START : onset + TRIAL_START + TARGET_T, :N_EEG_CH]
            if seg.shape[0] < TARGET_T:
                continue
            trials_X.append(seg)
            trials_y.append(label)

    return np.stack(trials_X), np.array(trials_y)


def zscore(X):
    """Z-score each channel independently across time."""
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sigma


all_X, all_y, all_subj = [], [], []

for subj_idx, subj in enumerate(SUBJECTS):
    print(f"\n[{subj_idx+1}/9] {subj}")
    fname = download(subj)
    try:
        X, y = load_trials(fname)                  # (n_trials, T, 22)
    except Exception as e:
        print(f"  ERROR loading {fname}: {e}, skipping.")
        continue
    X = zscore(X)
    print(f"  {X.shape[0]} trials, shape {X.shape}")
    all_X.append(X)
    all_y.append(y)
    all_subj.extend([subj_idx] * len(y))

X_all = np.concatenate(all_X, axis=0)   # (N, T, 22)
y_all = np.concatenate(all_y, axis=0)   # (N,)

X_t = torch.from_numpy(X_all)           # float32
y_t = torch.from_numpy(y_all).long()
subj_t = torch.tensor(all_subj).long()

torch.save(X_t, f"{DATA_DIR}/data.pt")
torch.save(y_t, f"{DATA_DIR}/labels.pt")
torch.save(subj_t, f"{DATA_DIR}/subjects.pt")

print(f"\nSaved:")
print(f"  {DATA_DIR}/data.pt     shape={tuple(X_t.shape)}")
print(f"  {DATA_DIR}/labels.pt   shape={tuple(y_t.shape)}")
print(f"  {DATA_DIR}/subjects.pt shape={tuple(subj_t.shape)}")
print(f"  Label distribution: {dict(zip(*np.unique(y_all, return_counts=True)))}")

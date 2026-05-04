"""
Baseline feature extraction + GMM clustering on the Andrzejak EEG dataset.
Reproduces Table 7 baselines from the paper:
  - tsfresh (catch-22 subset for speed; full tsfresh optional)
  - Catch-22
  - ROCKET
  - MiniRocket

All methods extract features, then a 2-component GMM clusters subjects.
Clustering accuracy (best permutation) is reported.

Usage:
  python scripts/eeg_baselines.py --data_path ./data/eeg/data.pt \
                                   --labels_path ./data/eeg/labels.pt

Dependencies (install if missing):
  pip install tsfresh catch22 rocket-python scikit-learn
  # or: pip install tsfresh pyts sktime
"""

import argparse
import numpy as np
import torch
from itertools import permutations
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2) -> float:
    best = 0.0
    for perm in permutations(range(n_classes)):
        remapped = np.array([perm[c] for c in y_pred])
        best = max(best, accuracy_score(y_true, remapped))
    return best


def gmm_cluster(features: np.ndarray, labels: np.ndarray, n_init: int = 20) -> float:
    scaler   = StandardScaler()
    features = scaler.fit_transform(features)
    # drop NaN/Inf columns (tsfresh can produce these)
    mask     = np.isfinite(features).all(axis=0)
    features = features[:, mask]
    gmm  = GaussianMixture(n_components=2, n_init=n_init, covariance_type="full", random_state=42)
    pred = gmm.fit_predict(features)
    return cluster_accuracy(labels, pred) * 100


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def run_catch22(X: np.ndarray) -> np.ndarray:
    """X: (n_subjects, T)"""
    try:
        import catch22
    except ImportError:
        raise ImportError("pip install catch22")
    feats = []
    for x in X:
        res = catch22.catch22_all(x.tolist())
        feats.append(res["values"])
    return np.array(feats, dtype=np.float32)


def run_tsfresh(X: np.ndarray) -> np.ndarray:
    """X: (n_subjects, T) — uses minimal feature set for speed."""
    try:
        import pandas as pd
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters
    except ImportError:
        raise ImportError("pip install tsfresh")

    n, T   = X.shape
    ids    = np.repeat(np.arange(n), T)
    times  = np.tile(np.arange(T), n)
    values = X.ravel()
    df = pd.DataFrame({"id": ids, "time": times, "value": values})
    feats = extract_features(df, column_id="id", column_sort="time",
                             default_fc_parameters=MinimalFCParameters(),
                             disable_progressbar=True)
    return feats.values.astype(np.float32)


def run_rocket(X: np.ndarray) -> np.ndarray:
    """X: (n_subjects, T)"""
    try:
        from sktime.transformations.panel.rocket import Rocket
        import pandas as pd
    except ImportError:
        raise ImportError("pip install sktime")

    # sktime expects (n, 1, T)
    X3d  = X[:, None, :]
    rocket = Rocket(num_kernels=10000, random_state=42)
    rocket.fit(X3d)
    return rocket.transform(X3d).astype(np.float32)


def run_minirocket(X: np.ndarray) -> np.ndarray:
    """X: (n_subjects, T)"""
    try:
        from sktime.transformations.panel.rocket import MiniRocket
    except ImportError:
        raise ImportError("pip install sktime")

    X3d = X[:, None, :]
    mr  = MiniRocket(random_state=42)
    mr.fit(X3d)
    return mr.transform(X3d).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",   type=str, default="./data/eeg/data.pt")
    p.add_argument("--labels_path", type=str, default="./data/eeg/labels.pt")
    return p.parse_args()


METHODS = {
    "Catch-22":   run_catch22,
    "tsfresh":    run_tsfresh,
    "ROCKET":     run_rocket,
    "MiniRocket": run_minirocket,
}


def main():
    args   = parse_args()
    data   = torch.load(args.data_path,   map_location="cpu").numpy()  # (200, T, 1)
    labels = torch.load(args.labels_path, map_location="cpu").numpy()  # (200,)

    X = data[:, :, 0]  # (200, T) — single channel
    print(f"Data shape: {X.shape}, labels: {np.bincount(labels)}")
    print()

    results = {}
    for name, fn in METHODS.items():
        print(f"Running {name} …", end=" ", flush=True)
        try:
            feats = fn(X)
            acc   = gmm_cluster(feats, labels)
            results[name] = acc
            print(f"{acc:.1f}%")
        except ImportError as e:
            print(f"SKIPPED ({e})")
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n--- Summary (GMM clustering accuracy) ---")
    for name, acc in results.items():
        print(f"  {name:<20s}  {acc:.1f}%")
    print(f"  {'HierDSR (paper)':<20s}  92.6 ± 1.2%")


if __name__ == "__main__":
    main()

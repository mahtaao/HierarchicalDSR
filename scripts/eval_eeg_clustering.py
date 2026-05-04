"""
Evaluate the learned p-vectors from the hierarchical DSR on the EEG task.

For each trained run:
  1. Load the final checkpoint.
  2. Extract model.p_vector  (shape: 200 × dp).
  3. Fit a 2-component GMM in p-vector space.
  4. Assign each subject to the cluster with the highest posterior.
  5. Compute clustering accuracy (permutation-matched to true labels).

Aggregates accuracy over all runs and prints mean ± std (paper reports 92.6 ± 1.2%).

Usage:
  python scripts/eval_eeg_clustering.py \
      --model_path ./trained_models/eeg/eeg \
      --labels_path ./data/eeg/labels.pt
"""

import os
import argparse
import glob
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from itertools import permutations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_p_vectors(run_dir: str) -> np.ndarray | None:
    """Load p_vector from the latest checkpoint in a run directory."""
    # checkpoints saved as model_<epoch>.pt by saving.py
    ckpts = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")))
    if not ckpts:
        print(f"  No checkpoints found in {run_dir}")
        return None
    ckpt_path = ckpts[-1]  # latest
    state = torch.load(ckpt_path, map_location="cpu")
    if "p_vector" in state:
        p = state["p_vector"].numpy()
    elif "model_state_dict" in state:
        p = state["model_state_dict"]["p_vector"].numpy()
    else:
        # try reading directly
        p = None
        for k, v in state.items():
            if "p_vector" in k:
                p = v.numpy()
                break
    if p is None:
        print(f"  Could not find p_vector in {ckpt_path}")
    return p


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2) -> float:
    """Best permutation accuracy for clustering (Hungarian / brute-force for 2 classes)."""
    best = 0.0
    for perm in permutations(range(n_classes)):
        remapped = np.array([perm[c] for c in y_pred])
        acc = accuracy_score(y_true, remapped)
        best = max(best, acc)
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  type=str, default="./trained_models/eeg/eeg",
                   help="Directory containing numbered run subdirs (001, 002, …)")
    p.add_argument("--labels_path", type=str, default="./data/eeg/labels.pt",
                   help="Path to labels.pt produced by prepare_eeg_data.py")
    p.add_argument("--n_components", type=int, default=2,
                   help="Number of GMM components (= number of classes)")
    p.add_argument("--n_gmm_init",   type=int, default=20,
                   help="GMM n_init for robustness")
    return p.parse_args()


def main():
    args  = parse_args()
    labels = torch.load(args.labels_path, map_location="cpu").numpy()  # (200,)

    # find run directories: named 001, 002, … or projection/001, …
    run_dirs = sorted(glob.glob(os.path.join(args.model_path, "**", "[0-9][0-9][0-9]"),
                                recursive=True))
    if not run_dirs:
        # fall back: numbered dirs directly under model_path
        run_dirs = sorted(glob.glob(os.path.join(args.model_path, "[0-9]*")))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {args.model_path}")

    print(f"Found {len(run_dirs)} run(s).")
    accuracies = []

    for run_dir in run_dirs:
        p_vec = load_p_vectors(run_dir)
        if p_vec is None:
            continue

        gmm = GaussianMixture(
            n_components=args.n_components,
            n_init=args.n_gmm_init,
            covariance_type="full",
            random_state=42,
        )
        pred = gmm.fit_predict(p_vec)
        acc  = cluster_accuracy(labels, pred, n_classes=args.n_components)
        accuracies.append(acc * 100)
        print(f"  {os.path.basename(run_dir)}: accuracy = {acc*100:.1f}%")

    if accuracies:
        mean_ = np.mean(accuracies)
        std_  = np.std(accuracies)
        print(f"\nClustering accuracy: {mean_:.1f} ± {std_:.1f}%  (n={len(accuracies)} runs)")
        print(f"Paper reports:       92.6 ± 1.2%")
    else:
        print("No runs could be evaluated.")


if __name__ == "__main__":
    main()

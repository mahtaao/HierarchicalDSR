"""
Prepare the Andrzejak et al. (2001) Bonn EEG dataset for the hierarchical DSR model.

Source ZIP (one file, ~3MB):
  https://raw.githubusercontent.com/RYH2077/EEG-Epilepsy-Datasets/master/Bonn%20EEG%20dataset.zip
  Structure inside: A_Z/ B_O/ C_N/ D_F/ E_S/  — each with 100 .txt EEG segments

  Set A_Z – healthy, eyes open       → label 0
  Set B_O – healthy, eyes closed     → label 0  (optional, see USE_SETS below)
  Set C_N – epileptic interictal opp → label 1  (optional)
  Set D_F – epileptic interictal foc → label 1  (optional)
  Set E_S – epileptic ictal (seizure)→ label 1

Binary task used in the paper: A_Z (healthy) vs E_S (ictal), 200 subjects, 1 channel.

Output (saved to data/eeg/):
  data.pt   – shape (200, SEQ_LEN, 1), z-scored per segment
               subjects 0..99 = healthy (Z), 100..199 = epileptic (S)
  labels.pt – shape (200,), 0 = healthy, 1 = epileptic
"""

import os
import io
import zipfile
import urllib.request
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "eeg")
SEQ_LEN  = 4096   # full segment; main.py sub-samples with --seq_len during training

ZIP_URL  = ("https://raw.githubusercontent.com/RYH2077/EEG-Epilepsy-Datasets"
            "/master/Bonn%20EEG%20dataset.zip")
ZIP_CACHE = "/tmp/bonn_eeg.zip"

# Folders inside the ZIP and their labels (0=healthy, 1=epileptic)
USE_SETS = {
    "A_Z": 0,   # healthy
    "E_S": 1,   # ictal seizure
}


def download_zip() -> bytes:
    if os.path.exists(ZIP_CACHE):
        print(f"  Using cached ZIP at {ZIP_CACHE}")
        with open(ZIP_CACHE, "rb") as f:
            return f.read()
    print(f"  Downloading from {ZIP_URL} …")
    with urllib.request.urlopen(ZIP_URL, timeout=120) as resp:
        data = resp.read()
    with open(ZIP_CACHE, "wb") as f:
        f.write(data)
    print(f"  Saved to {ZIP_CACHE} ({len(data)//1024} KB)")
    return data


def extract_set(zf: zipfile.ZipFile, folder: str, label: int) -> tuple[np.ndarray, np.ndarray]:
    txt_files = sorted([n for n in zf.namelist()
                        if n.startswith(folder + "/") and n.endswith(".txt")])
    assert len(txt_files) == 100, f"Expected 100 files in {folder}, got {len(txt_files)}"
    segments = []
    for fname in txt_files:
        with zf.open(fname) as f:
            vals = np.array(f.read().decode().split(), dtype=np.float32)
        segments.append(vals[:SEQ_LEN])
    segs   = np.stack(segments)                        # (100, SEQ_LEN)
    labels = np.full(len(segs), label, dtype=np.int64)
    print(f"  {folder}: {len(segs)} segments, label={label}")
    return segs, labels


def zscore(x: np.ndarray) -> np.ndarray:
    mu  = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mu) / std


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = download_zip()

    all_segs, all_labels = [], []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for folder, label in USE_SETS.items():
            segs, labs = extract_set(zf, folder, label)
            all_segs.append(segs)
            all_labels.append(labs)

    segments = np.concatenate(all_segs,   axis=0)  # (200, SEQ_LEN)
    labels   = np.concatenate(all_labels, axis=0)  # (200,)

    segments = zscore(segments)

    # shape expected by MultiSubjectDataset: (num_subjects, T, obs_size)
    data_tensor   = torch.tensor(segments[:, :, None], dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    data_path   = os.path.join(OUT_DIR, "data.pt")
    labels_path = os.path.join(OUT_DIR, "labels.pt")

    torch.save(data_tensor,   data_path)
    torch.save(labels_tensor, labels_path)

    print(f"\nSaved:")
    print(f"  {data_path}   shape={tuple(data_tensor.shape)}")
    print(f"  {labels_path} shape={tuple(labels_tensor.shape)}")
    print(f"  Labels: {labels_tensor.sum().item()} epileptic, "
          f"{(labels_tensor==0).sum().item()} healthy")


if __name__ == "__main__":
    main()

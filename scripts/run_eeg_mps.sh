#!/usr/bin/env bash
# Run hierarchical DSR on the Andrzejak EEG dataset on Apple Silicon (MPS).
# Mirrors run_eeg.sh but with --use_gpu (triggers MPS via get_device) and
# --plots "" (skips per-epoch trajectory saves that caused ~39 min overhead).
# Reduced seq_len=200 for local runs; use seq_len=500 to match paper.
#
# Run from repo root:  bash scripts/run_eeg_mps.sh

set -euo pipefail

DATA="./data/eeg/data.pt"
SAVE="./trained_models/eeg"
N_RUNS=${N_RUNS:-3}
SEQ_LEN=${SEQ_LEN:-200}

for RUN in $(seq 1 $N_RUNS); do
  echo "=== Run $RUN / $N_RUNS (MPS, seq_len=$SEQ_LEN) ==="
  python main.py \
    --data_path              "$DATA"      \
    --obs_size               1           \
    --latent_size            1           \
    --hidden_size            20          \
    --obs_model              identity    \
    --hierarchisation_scheme projection  \
    --num_individual_params  2           \
    --seq_len                $SEQ_LEN    \
    --train_set_size         3500        \
    --num_epochs             5000        \
    --batch_size             128         \
    --batches_per_epoch      50          \
    --learning_rate          1e-4        \
    --individual_learning_rate 1e-3      \
    --tf_alpha_start         0.5        \
    --tf_alpha_end           0.0        \
    --clip_grad_norm         10.0       \
    --lam                    0.0        \
    --num_workers            4          \
    --metrics                           \
    --plots                             \
    --save_path              "$SAVE"     \
    --experiment             eeg        \
    --name                   projection  \
    --run                    $RUN        \
    --use_gpu                           \
    "$@"
done

echo "Done. Models saved to $SAVE"
echo "Run: python scripts/eval_eeg_clustering.py --model_path $SAVE/eeg"

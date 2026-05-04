#!/usr/bin/env bash
# Run the hierarchical DSR on the Andrzejak EEG dataset.
# Run from the repo root:  bash scripts/run_eeg.sh
#
# Hyperparameters follow the paper defaults:
#   M  = obs_size = 1  (single EEG channel)
#   L  = 20*M    = 20  (hidden size)
#   dp = 2             (feature / p-vector dimension; small for interpretability)
#   lr_group = 1e-4, lr_individual = 1e-3
#   10 runs for reproducible mean ± std

set -euo pipefail

DATA="./data/eeg/data.pt"
SAVE="./trained_models/eeg"
N_RUNS=10

for RUN in $(seq 1 $N_RUNS); do
  echo "=== Run $RUN / $N_RUNS ==="
  python main.py \
    --data_path            "$DATA" \
    --obs_size             1       \
    --latent_size          1       \
    --hidden_size          20      \
    --obs_model            identity \
    --hierarchisation_scheme projection \
    --num_individual_params 2      \
    --seq_len              500     \
    --train_set_size       3500    \
    --num_epochs           5000    \
    --batch_size           128     \
    --batches_per_epoch    50      \
    --learning_rate        1e-4    \
    --individual_learning_rate 1e-3 \
    --tf_alpha_start       0.5    \
    --tf_alpha_end         0.0    \
    --clip_grad_norm       10.0   \
    --lam                  0.0    \
    --num_workers          4      \
    --metrics              \
    --plots                hier   \
    --save_path            "$SAVE" \
    --experiment           eeg    \
    --name                 projection \
    --run                  $RUN   \
    "$@"   # pass extra args from command line if needed
done

echo "Done. Models saved to $SAVE"
echo "Run: python scripts/eval_eeg_clustering.py --model_path $SAVE/eeg"

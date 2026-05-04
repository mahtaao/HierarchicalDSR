#!/usr/bin/env bash
# Train multi-channel shPLRNN on BCI IV-2a (22-channel, motor imagery).
# obs_model=linear lets latent_size differ from obs_size.
# Latents are the "synthetic sources" used in the NeurIPS position paper experiments.
#
# Run from repo root:  bash scripts/run_bci_mps.sh

set -euo pipefail

DATA="./data/bci/data.pt"
SAVE="./trained_models/bci"
RUN=${RUN:-1}

python main.py \
  --data_path              "$DATA"      \
  --obs_size               22          \
  --latent_size            4           \
  --hidden_size            20          \
  --obs_model              linear      \
  --hierarchisation_scheme projection  \
  --num_individual_params  4           \
  --seq_len                50          \
  --train_set_size         600         \
  --subjects_per_batch     32          \
  --num_epochs             500         \
  --batch_size             64          \
  --batches_per_epoch      10          \
  --learning_rate          1e-4        \
  --individual_learning_rate 1e-3      \
  --tf_alpha_start         0.5        \
  --tf_alpha_end           0.0        \
  --clip_grad_norm         10.0       \
  --num_workers            0          \
  --metrics                           \
  --plots                             \
  --save_path              "$SAVE"     \
  --experiment             bci        \
  --name                   linear_M4   \
  --run                    $RUN        \
  --use_gpu                           \
  "$@"

echo "Done. Model saved to $SAVE/bci/linear_M4/run_$RUN"

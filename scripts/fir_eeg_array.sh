#!/usr/bin/env bash
#SBATCH --job-name=hierDSR_eeg
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --array=1-10
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# Submit from $SCRATCH:
#   cd $SCRATCH && sbatch ~/research/HierarchicalDSR/scripts/fir_eeg_array.sh

set -euo pipefail

REPO="$HOME/research/HierarchicalDSR"
VENV="$HOME/envs/hierarchical_dsr"
DATA="$SCRATCH/hierarchical_dsr/data/eeg"
SAVE="$SCRATCH/hierarchical_dsr/trained_models/eeg"
RUN=$SLURM_ARRAY_TASK_ID

echo "=== Job $SLURM_JOB_ID  array=$RUN  node=$(hostname) ==="

module --force purge
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6
source "$VENV/bin/activate"

# Stage data to fast local storage
LOCAL_DATA="$SLURM_TMPDIR/eeg"
mkdir -p "$LOCAL_DATA"
cp "$DATA/data.pt"   "$LOCAL_DATA/"
cp "$DATA/labels.pt" "$LOCAL_DATA/"

mkdir -p "$SAVE"

cd "$REPO"

python main.py \
    --data_path            "$LOCAL_DATA/data.pt" \
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
    --plots                \
    --save_path            "$SAVE" \
    --experiment           eeg    \
    --name                 projection \
    --run                  $RUN   \
    --use_gpu

echo "Run $RUN done."

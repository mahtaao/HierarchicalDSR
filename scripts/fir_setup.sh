#!/usr/bin/env bash
# Run ONCE on Fir login node to create the virtualenv and install deps.
# Usage: bash scripts/fir_setup.sh
set -euo pipefail

REPO_DIR="$HOME/research/HierarchicalDSR"
VENV_DIR="$HOME/envs/hierarchical_dsr"
SCRATCH_DATA="$SCRATCH/hierarchical_dsr/data/eeg"

echo "=== Setting up HierarchicalDSR on Fir ==="
echo "Repo   : $REPO_DIR"
echo "Venv   : $VENV_DIR"
echo "Data   : $SCRATCH_DATA"
echo

# ---- virtualenv ----
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv..."
    module purge
    module load python/3.11
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
module purge
module load python/3.11
module load cuda/12.4

echo "Installing packages..."
# Use Alliance wheelhouse first, then PyPI fallback
pip install --upgrade pip --quiet

# Core scientific stack (usually in wheelhouse)
pip install --no-index torch torchvision 2>/dev/null || \
    pip install torch --index-url https://download.pytorch.org/whl/cu124 --quiet

pip install --no-index numpy scipy scikit-learn matplotlib 2>/dev/null || \
    pip install numpy scipy scikit-learn matplotlib --quiet

pip install --no-index tensorboard 2>/dev/null || \
    pip install tensorboard --quiet

pip install --no-index tqdm 2>/dev/null || \
    pip install tqdm --quiet

python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

# ---- data ----
mkdir -p "$SCRATCH_DATA"
if [ ! -f "$SCRATCH_DATA/data.pt" ]; then
    echo "Downloading & preparing EEG data..."
    cd "$REPO_DIR"
    python scripts/prepare_eeg_data.py
    cp data/eeg/data.pt   "$SCRATCH_DATA/"
    cp data/eeg/labels.pt "$SCRATCH_DATA/"
else
    echo "Data already exists at $SCRATCH_DATA"
fi

echo ""
echo "=== Setup complete ==="
echo "Submit jobs with: cd \$SCRATCH && sbatch $REPO_DIR/scripts/fir_eeg_array.sh"

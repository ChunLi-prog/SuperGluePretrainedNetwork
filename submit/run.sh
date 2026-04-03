#!/bin/bash
# MCAP GT Generation - L20 Cluster Run Script
# Called by aidi-inf-cli inside the Docker container.
#
# Usage: bash run.sh 0,1 [--parking-ids P1 P2]
set -e

echo "=== MCAP GT Generation Pipeline ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "PWD: $(pwd)"

cd /running_package/

# Create virtualenv and install deps
virtualenv -p /usr/local/bin/python3 env_3
source env_3/bin/activate

cd SuperGluePretrainedNetwork

echo "=== Installing dependencies ==="
pip3 install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://pypi.hobot.cc/simple \
    --extra-index-url https://pypi.hobot.cc/hobot-local/simple 2>&1 | tail -5

pip3 install opencv-python-headless numpy pyyaml tqdm scipy \
    -i https://pypi.hobot.cc/simple \
    --extra-index-url https://pypi.hobot.cc/hobot-local/simple 2>&1 | tail -5

# Install cmdk from bundled wheel
if ls wheels/cmdk*.whl 1>/dev/null 2>&1; then
    pip3 install wheels/cmdk*.whl 2>&1 | tail -3
    echo "cmdk installed from wheel"
else
    echo "WARNING: No cmdk wheel found in wheels/"
fi

# Verify GPU
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Parse GPU IDs
GPUID=${1:-0}
shift || true  # remaining args passed to orchestrator
export CUDA_VISIBLE_DEVICES=$GPUID
echo "CUDA_VISIBLE_DEVICES=$GPUID"

# Count GPUs
num_commas=$(echo $GPUID | tr -cd ',' | wc -c)
num_gpu=$((num_commas + 1))
echo "Number of GPUs: $num_gpu"

# Run orchestrator
echo "=== Starting Pipeline ==="
python3 orchestrator.py \
    --config config/mcap_pipeline_l20.yaml \
    --stages stage1 stage2 \
    "$@"

echo "=== Pipeline Complete ==="
echo "Date: $(date)"

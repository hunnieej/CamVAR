#!/bin/bash
set -ex

#=============================================================================
# 2-GPU DDP Training Script for STAR-T2I Ray Adaptation
#=============================================================================
# Usage:
#   ./train_2gpu.sh [config_file]
#
# Example:
#   ./train_2gpu.sh configs/config_phase_c_2gpu.json
#
# This script launches distributed training on 2 local GPUs using torchrun.
# Batch size calculation: bs=8, ac=2, num_gpus=2 → per_gpu_batch = 8/2/2 = 2
#=============================================================================

# Activate conda environment
source /home/mmai6k_jh/anaconda3/bin/activate SMGD

# Use both GPUs (0 and 1)
export CUDA_VISIBLE_DEVICES=0,1

# DDP environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# NCCL settings for 2-GPU training (local communication)
export NCCL_DEBUG=INFO  # Change to WARN for less verbose output
export NCCL_IB_DISABLE=1  # Disable InfiniBand for local GPUs
export NCCL_P2P_DISABLE=0  # Enable P2P for better performance

# Configuration file (default or passed as argument)
CONFIG_FILE=${1:-configs/config_phase_c_2gpu.json}

# Verify GPUs are available
echo "=========================================="
echo "2-GPU DDP Training Setup"
echo "=========================================="
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Config: $CONFIG_FILE"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Launch distributed training with torchrun
# torchrun is the modern PyTorch distributed launcher (replaces torch.distributed.launch)
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --config=$CONFIG_FILE

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code: $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE

#!/bin/bash
#=============================================================================
# Quick verification script for 2-GPU DDP setup
#=============================================================================

set -e

echo "=========================================="
echo "2-GPU DDP Setup Verification"
echo "=========================================="
echo ""

# Check if files exist
echo "1. Checking required files..."
FILES=(
    "configs/config_phase_c_2gpu.json"
    "train_2gpu.sh"
    "train.py"
    "dist.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file exists"
    else
        echo "   ✗ $file missing!"
        exit 1
    fi
done
echo ""

# Check GPUs
echo "2. Checking GPU availability..."
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "   Found $GPU_COUNT GPU(s)"

if [ $GPU_COUNT -lt 2 ]; then
    echo "   ⚠ Warning: Less than 2 GPUs available!"
    echo "   2-GPU training requires at least 2 GPUs."
    exit 1
else
    echo "   ✓ Sufficient GPUs for 2-GPU training"
fi

echo ""
echo "3. GPU Details:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Check config differences
echo "4. Verifying config differences..."
BS_SINGLE=$(grep '"bs"' configs/config_phase_c.json | grep -o '[0-9]*' | head -1)
BS_DOUBLE=$(grep '"bs"' configs/config_phase_c_2gpu.json | grep -o '[0-9]*' | head -1)

echo "   Single-GPU config: bs=$BS_SINGLE"
echo "   2-GPU config: bs=$BS_DOUBLE"

if [ "$BS_DOUBLE" -eq "$((BS_SINGLE * 2))" ]; then
    echo "   ✓ Batch size correctly doubled"
else
    echo "   ⚠ Warning: Batch size not exactly doubled"
fi
echo ""

# Check Python environment
echo "5. Checking Python environment..."
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "   ✗ Python not found!"
    exit 1
fi

echo "   Python: $($PYTHON_CMD --version)"

# Check if conda env is activated
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "   ✓ Conda environment: $CONDA_DEFAULT_ENV"
else
    echo "   ⚠ No conda environment activated"
    echo "     Run: source /home/mmai6k_jh/anaconda3/bin/activate SMGD"
fi
echo ""

# Check PyTorch and CUDA
echo "6. Checking PyTorch setup..."
cat > /tmp/check_torch.py << 'EOF'
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check if distributed is available
    import torch.distributed as dist
    print(f"   ✓ torch.distributed available")
    
except ImportError as e:
    print(f"   ✗ Error: {e}")
    exit(1)
EOF

$PYTHON_CMD /tmp/check_torch.py
rm /tmp/check_torch.py
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "✓ All checks passed!"
echo ""
echo "Ready to launch 2-GPU training with:"
echo "  ./train_2gpu.sh"
echo ""
echo "Or with custom config:"
echo "  ./train_2gpu.sh configs/config_phase_c_2gpu.json"
echo "=========================================="

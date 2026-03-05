# 2-GPU DDP Training Guide

## Overview

This setup enables **2-GPU Distributed Data Parallel (DDP)** training for STAR-T2I Ray Adaptation, doubling throughput while maintaining the same per-GPU memory footprint.

---

## Files Created

### 1. `configs/config_phase_c_2gpu.json`
**2-GPU configuration with doubled global batch size**

**Key Changes from Single-GPU Config:**
- `bs: 4 → 8` (global batch size doubled)
- `wandb_run_name: "phase_c_2gpu_bs8"` (tracking clarity)
- `local_out_dir_path: "outputs/260305_2gpu_bs8"` (separate output dir)

**Batch Size Calculation:**
```
Global batch size (bs) = 8
Accumulation steps (ac) = 2
Number of GPUs = 2

Per-GPU batch = bs / ac / num_gpus
             = 8 / 2 / 2
             = 2 samples per GPU per step

Effective batch size = per_gpu_batch × num_gpus × accumulation
                     = 2 × 2 × 2
                     = 8
```

**Result:** Same per-GPU memory usage as single-GPU, but **2x throughput**!

---

### 2. `train_2gpu.sh`
**Launch script using modern `torchrun` launcher**

**Features:**
- Automatic GPU detection and verification
- Proper NCCL configuration for local 2-GPU training
- Error handling and exit codes
- Verbose startup information

---

## Usage

### Quick Start

```bash
# Launch 2-GPU training with default config
./train_2gpu.sh

# Or specify a custom config
./train_2gpu.sh configs/config_phase_c_2gpu.json
```

### Verify GPU Setup Before Training

```bash
# Check GPUs are visible and free
nvidia-smi

# Check CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
```

---

## Single-GPU vs 2-GPU Comparison

### Single-GPU (`train_single_gpu.sh`)
```bash
Config: configs/config_phase_c.json
bs = 4, ac = 2, num_gpus = 1
Per-GPU batch = 4/2/1 = 2 samples/step
Effective batch = 2 × 1 × 2 = 4
Output: outputs/260305_at3/
```

### 2-GPU DDP (`train_2gpu.sh`)
```bash
Config: configs/config_phase_c_2gpu.json
bs = 8, ac = 2, num_gpus = 2
Per-GPU batch = 8/2/2 = 2 samples/step
Effective batch = 2 × 2 × 2 = 8
Output: outputs/260305_2gpu_bs8/
```

**Key Advantage:** 2x training throughput with identical per-GPU VRAM usage!

---

## Technical Details

### DDP Architecture

The codebase uses PyTorch's `DistributedDataParallel`:

```python
# From train.py
var: DDP = (DDP if dist.initialized() else NullDDP)(
    var_wo_ddp, 
    device_id=dist.get_local_rank(),
    ...
)
```

**When DDP is initialized:**
- Each GPU gets `1/world_size` of the data
- Gradients are automatically synchronized via `AllReduce`
- Model parameters stay in sync across GPUs

### NCCL Configuration

The script sets optimal NCCL settings for 2-GPU local training:

```bash
export NCCL_DEBUG=INFO           # Verbose logging (change to WARN for less output)
export NCCL_IB_DISABLE=1         # Disable InfiniBand (not needed for local)
export NCCL_P2P_DISABLE=0        # Enable P2P for better performance
```

### WandB Logging

**Already DDP-compatible!** ✅

- Logger only initializes on rank 0 (master process)
- Config saved once to avoid conflicts
- All metrics automatically gathered from all ranks

---

## Troubleshooting

### Problem: "Address already in use"

```bash
ERROR: [Errno 98] Address already in use
```

**Solution:** Change the master port in `train_2gpu.sh`:
```bash
export MASTER_PORT=12356  # Change from 12355
```

---

### Problem: NCCL timeout or hanging

```bash
ERROR: NCCL timeout
```

**Solution 1:** Increase NCCL timeout:
```bash
export NCCL_TIMEOUT=1800  # 30 minutes
```

**Solution 2:** Check GPU communication:
```bash
nvidia-smi topo -m  # Check GPU topology
```

---

### Problem: OOM (Out of Memory)

```bash
RuntimeError: CUDA out of memory
```

**Solution:** Reduce batch size in config:
```json
{
  "bs": 6,  // 8 → 6 (reduce by 25%)
  "ac": 2
}
```

This gives: `6/2/2 = 1.5 → 1 sample per GPU` (rounds down)

---

### Problem: Different GPUs load imbalance

```bash
GPU 0: 90% utilization
GPU 1: 30% utilization
```

**Solution:** Ensure data sampler is properly distributed:
- Check `DistInfiniteBatchSampler` is used (it is by default)
- Verify `world_size` is correctly detected: Add debug print in `dist.py`

---

## Performance Expectations

### Training Speed

**Single-GPU (bs=4):**
- ~6000 iterations per epoch (1500 panos × 12 views / 3 saves × batch_size)
- ~X seconds per iteration
- Total: ~Y hours per epoch

**2-GPU DDP (bs=8):**
- ~3000 iterations per epoch (same data, double throughput)
- ~X seconds per iteration (similar, slight DDP overhead)
- Total: ~Y/2 hours per epoch (**2x faster!**)

### Memory Usage

Both configs use **~2 samples per GPU**, so VRAM usage should be identical:
- Single-GPU: 2 samples → ~Z GB VRAM
- 2-GPU: 2 samples per GPU → ~Z GB VRAM per GPU

**Safe headroom:** RTX 6000 Ada has 49GB, so plenty of room for scaling if needed.

---

## Scaling to Larger Batch Sizes

If you want to increase batch size further (after verifying 2-GPU works):

### Option: 4 samples per GPU (bs=16)
```json
{
  "bs": 16,              // 8 → 16
  "ac": 2,
  "wandb_run_name": "phase_c_2gpu_bs16"
}
```
- Per-GPU: 16/2/2 = 4 samples
- Effective: 4 × 2 × 2 = 16
- **Risk:** Higher VRAM usage, test first!

### Option: Remove accumulation (bs=8, ac=1)
```json
{
  "bs": 8,
  "ac": 1,               // 2 → 1 (faster updates)
  "wandb_run_name": "phase_c_2gpu_bs8_ac1"
}
```
- Per-GPU: 8/1/2 = 4 samples
- Effective: 4 × 2 = 8
- **Benefit:** Faster gradient updates, may improve convergence

---

## Next Steps

1. **Verify setup:**
   ```bash
   # Check GPUs
   nvidia-smi
   
   # Dry run (exit after 1 iteration)
   # Add --debug flag if available
   ./train_2gpu.sh
   ```

2. **Monitor training:**
   - WandB dashboard: Check both GPUs are being used
   - `nvidia-smi -l 1`: Live GPU monitoring
   - Check logs for DDP initialization messages

3. **Compare results:**
   - Single-GPU run: `outputs/260305_at3/`
   - 2-GPU run: `outputs/260305_2gpu_bs8/`
   - Compare training curves in WandB

4. **Scale if successful:**
   - Try larger batch sizes
   - Adjust learning rate if needed (linear scaling rule: `lr ∝ batch_size`)

---

## Files Reference

```
configs/
├── config_phase_c.json         # Single-GPU (bs=4)
└── config_phase_c_2gpu.json    # 2-GPU (bs=8) ← NEW

train_single_gpu.sh              # Single-GPU launcher
train_2gpu.sh                    # 2-GPU DDP launcher ← NEW

outputs/
├── 260305_at3/                  # Single-GPU outputs
└── 260305_2gpu_bs8/             # 2-GPU outputs ← NEW
```

---

## FAQ

**Q: Can I use GPUs 1 and 2 instead of 0 and 1?**

A: Yes! Edit `train_2gpu.sh`:
```bash
export CUDA_VISIBLE_DEVICES=1,2
```

**Q: Will checkpoints be compatible between single-GPU and 2-GPU?**

A: Yes! Model weights are identical, only the training process differs.

**Q: Do I need to adjust learning rate for larger batch size?**

A: Generally yes, use **linear scaling**: If you double batch size, consider doubling learning rate. But test first!

**Q: Can I switch between single-GPU and 2-GPU mid-training?**

A: Yes! Just make sure to use compatible checkpoints and configs.

**Q: How do I know DDP is actually working?**

A: Check logs for:
```
[dist initialize] lrk=0, rk=0  # Rank 0 (master)
[dist initialize] lrk=1, rk=1  # Rank 1 (worker)
```

And monitor both GPUs with `nvidia-smi -l 1` - both should show activity.

---

## Contact

If you encounter issues, check:
1. PyTorch version: `python -c "import torch; print(torch.__version__)"`
2. NCCL version: `python -c "import torch; print(torch.cuda.nccl.version())"`
3. GPU communication: `nvidia-smi topo -m`

Happy training! 🚀

# GPU HPC Toolkit

GPU monitoring demo using simulated data (no GPU hardware available),
and cluster-convention jobscripts for NVIDIA GPU workloads.
Developed on Apple M1 (no CUDA); jobscripts checked against real cluster documentation.

## Contents

### scripts/
- gpu_monitor_sim.py - GPU monitoring demo, 1 GPU/node (RTX 6000 Ada 48 GB), simulated

### examples/
- pytorch-single-gpu.sh - PyTorch training on 1 GPU
- gromacs-gpu.sh - GROMACS MD simulation with GPU offloading

**Note:** Lyra nodes have a single GPU each; scaling beyond one GPU means multi-node DDP
(torchrun --nnodes). Not included here, as I have no multi-GPU hardware to validate.

## GPU monitoring

Simulated dashboard (for learning and demo):

    python3 scripts/gpu_monitor_sim.py --once
    python3 scripts/gpu_monitor_sim.py --interval 2

On a real cluster with NVIDIA GPUs:

    nvidia-smi
    nvidia-smi dmon -s u
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv

## Key Slurm GPU directives

    #SBATCH --partition=gpu        # GPU partition
    #SBATCH --gres=gpu:1           # Request 1 GPU
    #SBATCH --gres=gpu:GPUTYPE:2   # Request a specific GPU type by name (e.g., v100, a100)

Check GPU allocation in job:

    echo $CUDA_VISIBLE_DEVICES    # which GPUs were allocated
    nvidia-smi                    # GPU status

## GPU vocabulary for HPC support

| Term | Definition |
|------|-----------|
| CUDA cores | Parallel processing units on NVIDIA GPU |
| VRAM | GPU memory - separate from CPU RAM |
| SM (Streaming Multiprocessor) | GPU compute unit grouping CUDA cores |
| cuDNN | NVIDIA library optimizing deep learning ops |
| NCCL | NVIDIA library for multi-GPU communication |
| DDP | PyTorch Distributed Data Parallel - multi-GPU training |
| MIG | Multi-Instance GPU - partition one GPU for multiple users |
| GRES | Generic Resources - how Slurm manages GPUs |

## Common GPU support issues

**GPU at 5% utilization:**
DataLoader bottleneck - GPU waiting for CPU to prepare data.
Fix: increase num_workers, use faster storage (local scratch).

**CUDA out of memory:**
Reduce batch size or use gradient checkpointing.
Check: nvidia-smi, reduce --batch-size.

**Job allocated GPU but not using it:**
Check CUDA_VISIBLE_DEVICES, verify module load CUDA.

## Note on development environment

This toolkit was developed on Apple M1 which uses Metal (not CUDA).
GPU jobscripts follow CECI/ULB cluster conventions.
Monitoring dashboard uses simulated data for demonstration (no CUDA / Apple M1 Metal).

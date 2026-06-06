#!/bin/bash
#SBATCH --job-name=pytorch_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # CPU workers for DataLoader
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu          # GPU partition
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- Modules ---
module purge
module load CUDA/12.2
module load Python/3.11

mkdir -p logs

# --- Verify GPU allocation ---
echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# --- Environment ---
source $HOME/envs/pytorch/bin/activate

# --- Critical: limit CPU threads to avoid oversubscription ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Run training ---
python3 train.py \
    --epochs 100 \
    --batch-size 256 \
    --num-workers $SLURM_CPUS_PER_TASK \
    --output results/

echo "Job finished: $(date)"

# --- GPU utilization summary ---
# Check after job: was the GPU actually used?
# sacct -j $SLURM_JOB_ID --format=JobID,GPUUtil,MaxRSS

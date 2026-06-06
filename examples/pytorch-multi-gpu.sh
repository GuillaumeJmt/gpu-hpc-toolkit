#!/bin/bash
#SBATCH --job-name=pytorch_multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32       # 8 workers per GPU
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4             # Request 4 GPUs on same node
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load CUDA/12.2
module load Python/3.11

mkdir -p logs

echo "Job started: $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

source $HOME/envs/pytorch/bin/activate

export OMP_NUM_THREADS=8

# PyTorch DDP (Distributed Data Parallel) on 4 GPUs
# torchrun replaces torch.distributed.launch
torchrun \
    --standalone \
    --nproc_per_node=4 \
    train_ddp.py \
    --epochs 50 \
    --batch-size 512

echo "Job finished: $(date)"

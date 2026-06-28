#!/bin/bash
#SBATCH --job-name=gromacs_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load GROMACS/2023.3-CUDA-12.2
mkdir -p logs

echo "Job started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Single node, single GPU: use the thread-MPI build (gmx), where -ntmpi is valid.
# (gmx_mpi + srun would be the multi-node form, and must NOT use -ntmpi.)
gmx mdrun -s topol.tpr \
    -nb gpu -pme gpu \
    -ntmpi 1 -ntomp $SLURM_CPUS_PER_TASK \
    -o traj.trr -c confout.gro -e ener.edr -g md.log

echo "Job finished: $(date)"

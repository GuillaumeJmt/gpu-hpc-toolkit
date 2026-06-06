#!/bin/bash
#SBATCH --job-name=gromacs_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4               # 4 MPI ranks
#SBATCH --cpus-per-task=4        # 4 OpenMP threads per rank
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1             # 1 GPU for PME offloading
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load GROMACS/2023.3-CUDA-12.2

mkdir -p logs

echo "Job started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GMX_GPU_ID=0

# GROMACS hybrid MPI+OpenMP+GPU
# -nb gpu: non-bonded forces on GPU
# -pme gpu: PME electrostatics on GPU
srun gmx_mpi mdrun \
    -s topol.tpr \
    -nb gpu \
    -pme gpu \
    -ntmpi $SLURM_NTASKS \
    -ntomp $SLURM_CPUS_PER_TASK \
    -gpu_id 0 \
    -o traj.trr \
    -c confout.gro \
    -e ener.edr \
    -g md.log

echo "Job finished: $(date)"

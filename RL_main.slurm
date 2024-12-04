#!/bin/bash
#SBATCH --job-name="e1_G95"
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1G
#SBATCH --account=research-ME-dcsc

module load 2024r1
module load miniconda3

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate /scratch/smsaad/RL-MPC-for-autonomous-greenhouse-control/.conda/envs/rlmpc_murray/

# Run the job
srun python RL_main.py 1 0.95

# Deactivate conda
conda deactivate
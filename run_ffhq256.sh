#!/bin/sh
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8 
#SBATCH --mem-per-gpu=32GB
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --partition=gpu-cluster
#SBATCH --time=72:00:00

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python run_ffhq256.py
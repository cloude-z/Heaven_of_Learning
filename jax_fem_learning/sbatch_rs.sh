#!/bin/sh
#SBATCH --job-name="jax_fem_simulation"
#SBATCH --partition=gpu
#SBATCH --time=03:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --account=research-ceg-he
#SBATCH --output=logs/output.%j.out
#SBATCH --error=logs/error.%j.err

module load 2024r1
module load openmpi/4.1.6
module load python/3.10.13
module load openblas/0.3.24_threads_none
module load petsc/3.20.1

source ~/workspace/venv/venv_jax_fem/bin/activate
srun python ~/workspace/projects/Heaven_of_Learning/jax_fem_learning/run_simulation.py

#!/bin/bash
#SBATCH --job-name=shear-analysis         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH --time=00:29:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH -o /global/homes/p/pecom/logs/anacal/unrec-analysis.%A.out
#SBATCH -e /global/homes/p/pecom/logs/anacal/unrec-analysis.%A.err

module load conda
conda activate /pscratch/sd/p/pecom/envs/anacal_mpi

# export OMP_PROC_BIND=spread
# export OMP_PLACES=threads
# export OMP_NUM_THREADS=1

cd /global/homes/p/pecom/anacal_scripts/analysis/
# srun --mpi=pmix -n 128 python mpi_constant_analysis.py
srun --mpi=pmix -n 128 python mpi_redshift_analysis.py -z 1
srun --mpi=pmix -n 128 python mpi_redshift_analysis.py -z 2

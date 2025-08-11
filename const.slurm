#!/bin/bash
#SBATCH --job-name=const-shear         # create a short name for your job
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=256
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -o /global/homes/p/pecom/logs/anacal/const.%A.out
#SBATCH -e /global/homes/p/pecom/logs/anacal/const.%A.err

module load conda
conda activate /pscratch/sd/p/pecom/envs/anacal_mpi

# export OMP_PROC_BIND=spread
# export OMP_PLACES=threads
# export OMP_NUM_THREADS=2

cd /global/homes/p/pecom/anacal_scripts/
srun --mpi=pmix -n 1024 python constant_shear.py -m 0
srun --mpi=pmix -n 1024 python constant_shear.py -m 1

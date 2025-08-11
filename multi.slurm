#!/bin/bash
#SBATCH --job-name=g-shear         # create a short name for your job
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=256
#SBATCH --cpus-per-task=1
#SBATCH --time=03:30:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -o /global/homes/p/pecom/logs/anacal/bin.%A.out
#SBATCH -e /global/homes/p/pecom/logs/anacal/bin.%A.err

module load conda
conda activate /pscratch/sd/p/pecom/envs/anacal_mpi

cd /global/homes/p/pecom/anacal_scripts/
srun --mpi=pmix -n 1024 python binned_shear.py -m 3 -b g
srun --mpi=pmix -n 1024 python binned_shear.py -m 1 -b g

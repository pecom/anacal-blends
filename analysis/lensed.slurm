#!/bin/bash
#SBATCH --job-name=unrec-wl         # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=256
#SBATCH --cpus-per-task=1
#SBATCH --time=00:29:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH -o /global/homes/p/pecom/logs/anacal/blend-label.%A.out
#SBATCH -e /global/homes/p/pecom/logs/anacal/blend-label.%A.err

module load conda
conda activate /pscratch/sd/p/pecom/envs/anacal_mpi

# export OMP_PROC_BIND=spread
# export OMP_PLACES=threads
# export OMP_NUM_THREADS=1

cd /global/homes/p/pecom/anacal_scripts/analysis/

# srun --mpi=pmix -n 256 python mpi_lensed_multi_unrec.py -b 1

# srun --mpi=pmix -n 512 python mpi_lensed_multi_unrec.py -b 1 -q 1 -p 10
# srun --mpi=pmix -n 512 python mpi_lensed_multi_unrec.py -b 2 -q 1 -p 10

srun --mpi=pmix -n 512 python mpi_lensed_multi_unrec.py -b 1 -q 1 -p 9
srun --mpi=pmix -n 512 python mpi_lensed_multi_unrec.py -b 2 -q 1 -p 9

#!/bin/bash
#SBATCH --job-name=rf-analysis         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=256
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH -o /global/homes/p/pecom/logs/anacal/rf-analysis.%A.out
#SBATCH -e /global/homes/p/pecom/logs/anacal/rf-analysis.%A.err

module load conda
conda activate /pscratch/sd/p/pecom/envs/anacal_mpi

# export OMP_PROC_BIND=spread
# export OMP_PLACES=threads
# export OMP_NUM_THREADS=1

cd /global/homes/p/pecom/anacal_scripts/analysis/

# for i in 0.8 0.9 
# do
# 	srun --mpi=pmix -n 2048 python mpi_rf_unrec_analysis.py -z 1 -q 1 -s $i
# 	srun --mpi=pmix -n 2048 python mpi_rf_unrec_analysis.py -z 2 -q 1 -s $i
# 
# 	srun --mpi=pmix -n 2048 python mpi_rf_unrec_analysis.py -z 1 -q 1 -s $i -r
# 	srun --mpi=pmix -n 2048 python mpi_rf_unrec_analysis.py -z 2 -q 1 -s $i -r
# done

srun --mpi=pmix -n 256 python mpi_rf_const_analysis.py -q 1 -s 0.8
srun --mpi=pmix -n 256 python mpi_rf_const_analysis.py -q 1 -s 0.8 -r

srun --mpi=pmix -n 256 python mpi_rf_const_analysis.py -q 1 -s 0.9
srun --mpi=pmix -n 256 python mpi_rf_const_analysis.py -q 1 -s 0.9 -r


# Need to run: 
# srun --mpi=pmix -n 2048 python mpi_rf_unrec_analysis.py -z 2 -q 1 -s 0.4 -r

# srun --mpi=pmix -n 256 python mpi_rf_unrec_analysis.py -z 1 -q 1 -s .005
# srun --mpi=pmix -n 256 python mpi_rf_unrec_analysis.py -z 2 -q 1 -s .005
# srun --mpi=pmix -n 256 python mpi_rf_unrec_analysis.py -z 1 -q 1 -s .005 -r
# srun --mpi=pmix -n 256 python mpi_rf_unrec_analysis.py -z 2 -q 1 -s .005 -r

# srun --mpi=pmix -n 128 python mpi_rf_unrec_analysis.py -z 1 -q 1 -s .9 
# srun --mpi=pmix -n 128 python mpi_rf_unrec_analysis.py -z 2 -q 1 -s .9 
# 
# srun --mpi=pmix -n 128 python mpi_rf_unrec_analysis.py -z 1 -q 1 -s .59 
# srun --mpi=pmix -n 128 python mpi_rf_unrec_analysis.py -z 2 -q 1 -s .59

# srun --mpi=pmix -n 128 python mpi_rf_unrec_analysis.py -z 1 -q 1 -s .21
# srun --mpi=pmix -n 128 python mpi_rf_unrec_analysis.py -z 2 -q 1 -s .21
# 
# srun --mpi=pmix -n 128 python mpi_rf_unrec_analysis.py -z 1 -q 1 -s .005
# srun --mpi=pmix -n 128 python mpi_rf_unrec_analysis.py -z 2 -q 1 -s .005
#

#!/bin/bash
#SBATCH --job-name=shear-analysis         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH --time=00:16:00          # total run time limit (HH:MM:SS)
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
# srun --mpi=pmix -n 128 python mpi_redshift_unrec_analysis.py -z 1
# srun --mpi=pmix -n 128 python mpi_redshift_unrec_analysis.py -z 2

# for i in 1 2 3 4 5 6 7 8 9 10
# do
# 	srun --mpi=pmix -n 128 python mpi_redshift_analysis.py -z 1 -p $i
# 	srun --mpi=pmix -n 128 python mpi_redshift_analysis.py -z 2 -p $i
# done


# srun --mpi=pmix -n 128 python mpi_unlensed_redshift_analysis.py -z 1 -p 5
# srun --mpi=pmix -n 128 python mpi_unlensed_redshift_analysis.py -z 2 -p 5

# for i in 1 2 3 4 5 6 7 8 9 10
# do
# 	srun --mpi=pmix -n 128 python mpi_unlensed_redshift_analysis.py -z 1 -u -p $i
# 	srun --mpi=pmix -n 128 python mpi_unlensed_redshift_analysis.py -z 2 -u -p $i
# done

# srun --mpi=pmix -n 128 python mpi_rf_unrec_analysis.py -z 1 -q 1
# srun --mpi=pmix -n 128 python mpi_rf_unrec_analysis.py -z 2 -q 1


for i in 6 7 8 9 10
do
	srun --mpi=pmix -n 128 python mpi_unlensed_analysis.py -z 1 -u -p $i
	srun --mpi=pmix -n 128 python mpi_unlensed_analysis.py -z 2 -u -p $i
done

# srun --mpi=pmix -n 128 python mpi_unlensed_analysis.py -z 1 -u -p 5
# srun --mpi=pmix -n 128 python mpi_unlensed_analysis.py -z 2 -u -p 5

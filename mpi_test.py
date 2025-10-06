import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


kale = rank**2
kale = comm.gather(kale, root=0)

if rank==0:
    print("Rank 0: ", kale)



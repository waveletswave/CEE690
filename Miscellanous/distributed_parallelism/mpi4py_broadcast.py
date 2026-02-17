#Example taken from https://mpi4py.readthedocs.io/en/stable/tutorial.html

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data = np.arange(size*5, dtype='i')
else:
    data = np.empty(size*5, dtype='i')
comm.Bcast(data, root=0)
print(f"Rank {rank}",data)



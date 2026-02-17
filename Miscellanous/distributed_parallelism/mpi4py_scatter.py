#Example taken from https://mpi4py.readthedocs.io/en/stable/tutorial.html

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
send_data = None
recv_data = None

if rank == 0:
    send_data = np.arange(size*5, dtype='i')
else:
    recv_data = np.empty(5, dtype='i')
comm.Scatter(send_data,recv_data, root=0)
if rank == 0:
    print(f"Rank {rank}",send_data)
else:
    print(f"Rank {rank}",recv_data)



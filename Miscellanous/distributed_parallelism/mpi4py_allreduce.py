from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Define the data to be reduced (a generic Python integer)
send_data = rank

# Perform the all-reduce operation
result = comm.allreduce(send_data, op=MPI.SUM)

# The result is now available in 'result' on all processes
print(f"Rank {rank}: Result is {result}")



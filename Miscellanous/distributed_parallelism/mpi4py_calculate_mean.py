import time
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def calculate_mean_collective():
    rows, cols = 5000, 5000
    data = None
    
    if rank == 0:
        # MAIN PROCESS: Only rank 0 creates the full matrix
        print(f"Main: Starting collective calculation with {size} processes...")
        np.random.seed(1)
        data = np.random.randn(rows, cols)
        start_time = time.time()
    else:
        # WORKERS: Start with nothing
        data = None

    # 1. SCATTER: Automatically slice the data and distribute it.
    # We send rows // size to each process.
    rows_per_rank = rows // size
    local_data = np.empty((rows_per_rank, cols), dtype=np.float64)
    
    # Note the Upper Case 'S': This is the fast, buffer-optimized version.
    comm.Scatter(data, local_data, root=0)

    # 2. LOCAL WORK: Every process (including Main) does its assigned part
    local_val = np.sum(local_data)
    local_count = local_data.size

    # 3. REDUCE: Combine all local results back to the Main process
    # This replaces the manual loop of 'recv' calls.
    total_val = comm.reduce(local_val, op=MPI.SUM, root=0)
    total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

    # 4. REPORT: Only the Main process reports the final result
    if rank == 0:
        final_mean = total_val / total_count
        print(f"Main: Final Mean = {final_mean:.6f}")
        print(f"Main: Total Time = {time.time() - start_time:.4f}s")

# Execute
calculate_mean_collective()

import numpy as np
from mpi4py import MPI
from numba.openmp import njit, openmp_context as openmp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@njit
def compute_local_chunk(chunk):
    rows, cols = chunk.shape
    local_val = 0.0
    local_count = 0
    
    # OPENMP: Parallelism WITHIN the node
    with openmp("parallel for reduction(+:local_val, local_count) private(j)"):
        for i in range(rows):
            for j in range(cols):
                local_val += chunk[i, j]
                local_count += 1
    return local_val, local_count

def run_hybrid_mean():
    rows, cols = 10000, 10000
    data = None
    
    if rank == 0:
        np.random.seed(1)
        data = np.random.randn(rows, cols)

    # 1. MPI SCATTER: Distribute large chunks to each node
    rows_per_rank = rows // size
    local_chunk = np.empty((rows_per_rank, cols), dtype=np.float64)
    comm.Scatter(data, local_chunk, root=0)

    # 2. LOCAL CALCULATION: Use OpenMP threads on the local chunk
    v, c = compute_local_chunk(local_chunk)

    # 3. MPI REDUCE: Bring the node-level totals back to Main
    total_val = comm.reduce(v, op=MPI.SUM, root=0)
    total_count = comm.reduce(c, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"Hybrid Result: {total_val / total_count}")

run_hybrid_mean()

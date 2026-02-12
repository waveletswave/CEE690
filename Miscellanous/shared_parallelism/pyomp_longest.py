import time
import numpy as np
import numba
from numba.openmp import njit, omp_set_num_threads, omp_get_num_threads, openmp_context, omp_get_thread_num

@njit
def calculate_mean_pyomp(data):
    """
    Using the Numba-OpenMP extension syntax.
    Note: We do NOT use parallel=True in the decorator because 
    the 'with openmp' block handles the parallelism explicitly.
    """
    val = 0.0
    count = 0
    rows, cols = data.shape

    # 1. Parallel Region (Spawn threads)
    with openmp_context("parallel shared(data, val, count)"):

        # Get rank and number of threads
        rank = omp_get_thread_num()
        size = omp_get_num_threads()

        # Every thread gets its own private local counters
        local_val = 0.0
        local_count = 0

        # 2. Work-Sharing Block (Calculate local sums)
        for i in range(rows):
            if i%size != rank:continue
            for j in range(cols):
                local_val += data[i, j]
                local_count += 1

        # 3. Separate Reduction Block (The merge)
        # 'critical' ensures only ONE thread at a time can run this block
        with openmp_context("critical"):
            val += local_val
            count += local_count
                
    return val / count

# 1. Setup Data
niter = 10
np.random.seed(1)
data = np.random.randn(5000, 5000)

# 3. Benchmark Numba-OpenMP
# Warm-up (Compilation happens here)
calculate_mean_pyomp(np.zeros((1, 1)))

for nthreads in [1,2,4,8,16]:
    omp_set_num_threads(nthreads)
    print(f"Running with {nthreads} threads")
    start_p = time.time()
    for i in range(niter):
        res_p = calculate_mean_pyomp(data)
    time_p = (time.time() - start_p)/niter
    print(f"Result:    {res_p} | Time: {time_p:.4f}s")

import time
import numba
import numpy as np

@numba.jit(nopython=True, parallel=True)
def calculate_mean(data):
    val = 0
    count = 0
    for i in numba.prange(data.shape[0]):
        for j in numba.prange(data.shape[1]):
            val += data[i,j]
            count += 1
    return val/count

# 1. Setup Data
niter = 10
np.random.seed(1)
data = np.random.randn(5000, 5000)

# 2. Benchmark
# Warm-up (Compilation happens here)
calculate_mean(np.zeros((1, 1)))

for nthreads in [1,2,4,8,16]:
    numba.set_num_threads(nthreads)
    print(f"Running with {nthreads} threads")
    start_p = time.time()
    for i in range(niter):
        res_p = calculate_mean(data)
    time_p = (time.time() - start_p)/niter
    print(f"Result:    {res_p} | Time: {time_p:.4f}s")


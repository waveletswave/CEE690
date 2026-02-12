import os
import time
# Set these BEFORE importing numpy
nthreads = 16
os.environ["OMP_NUM_THREADS"] = f"{nthreads}"
os.environ["MKL_NUM_THREADS"] = f"{nthreads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{nthreads}"

import numpy as np

#Create data
np.random.seed(1)
A = np.random.randn(5000,5000)
B = np.random.randn(5000,5000)

start_time = time.time()

# 3. The execution
final_result = np.dot(A,B)

duration = time.time() - start_time

print(f"{nthreads} threads")
print(f"Time: {duration:.8f} seconds")



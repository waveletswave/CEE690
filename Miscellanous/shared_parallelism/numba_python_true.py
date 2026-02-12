import time
import numba
import numpy as np
import warnings
from numba.core.errors import NumbaWarning

# Filter out the specific "Object Mode" or "Type Inference" warnings
warnings.simplefilter('ignore', category=NumbaWarning)

@numba.jit(nopython=False, parallel=True, forceobj=True)
def calculate_mean(data):
    val = 0
    count = 0
    for i in numba.prange(data.shape[0]):
        for j in numba.prange(data.shape[1]):
            val += data[i,j]
            count += 1
    return val/count

#Create data
np.random.seed(1)
data = np.random.randn(5000,5000)
    
# 1. The compilation step
calculate_mean(data) 


# 2. Define the number of threads
numba.set_num_threads(4)
print(f"Using {numba.get_num_threads()}",flush=True)
start_time = time.time()

# 3. The execution
final_result = calculate_mean(data)
    
duration = time.time() - start_time

print(f"Result: {final_result}")
print(f"Time: {duration:.8f} seconds")


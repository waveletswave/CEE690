import time
import numpy as np

def calculate_mean(data):
    val = 0
    count = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val += data[i,j]
            count += 1
    return val/count

#Create data
niter = 2
np.random.seed(1)
data = np.random.randn(5000,5000)

start_p = time.time()
for i in range(niter):
    res_p = calculate_mean(data)
time_p = (time.time() - start_p)/niter
print(f"Result:    {res_p} | Time: {time_p:.4f}s")
result = calculate_mean(data)










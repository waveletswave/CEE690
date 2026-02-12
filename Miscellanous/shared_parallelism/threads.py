import time
import numpy as np
import threading

def calculate_mean_thread(data, imin, imax, results, index):
    val = 0
    count = 0
    for i in range(imin,imax):
        for j in range(data.shape[1]):
            val += data[i,j]
            count += 1
    results[index] = (val, count)

# Create data
np.random.seed(1)
data = np.random.randn(5000, 5000)

niter = 2
n_threads = 4
result = 0.0
chunk = data.shape[0] // n_threads
results = [None] * n_threads
threads = []

start = time.time()
for k in range(niter):
    for i in range(n_threads):
        imin = i*chunk
        imax = (i+1)*chunk
        t = threading.Thread(target=calculate_mean_thread, 
                         args=(data, imin, imax, results, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    result += sum(r[0] for r in results) / sum(r[1] for r in results)

time_p = (time.time() - start)/niter
result = result/niter
print(f"{n_threads} threads")
print(f"Result:    {result} | Time: {time_p:.4f}s")







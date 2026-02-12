import numpy as np
from numba.openmp import njit, openmp_context, omp_set_num_threads

@njit
def trigger_race_condition(iterations):
    # This is our shared variable
    counter = 0
    
    # We spawn threads but provide NO protection (no reduction, no critical)
    with openmp_context("parallel shared(counter)"):
        with openmp_context("for"):
            for i in range(iterations):
                counter += 1
                
    return counter

# Setup
omp_set_num_threads(16)
iters = 1_000_000
expected = iters * 4

# Run the experiment
result = trigger_race_condition(iters)

print(f"Expected Result: {expected}")
print(f"Actual Result:   {result}")
print(f"Lost Increments: {expected - result}")



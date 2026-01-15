import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# Configuration variables
INPUT_FILE = 'era_interim_monthly_197901_201512_upscaled_annual.nc'
OUTPUT_FILE = 'out.nc'
VAR_NAME = 't2m'
LAT_MIN = 5
LAT_MAX = 50
LON_MIN = 10
LON_MAX = 100
TIME_MIN = 0
TIME_MAX = 10

# Load dataset
file_pointer_input = nc.Dataset(INPUT_FILE,'r')
t2m_data = file_pointer_input.variables[VAR_NAME][:]

temporal_spatial_mean, temporal_spatial_variance = [], []

# Calculate temporally varying spatial mean
for t in range(len(t2m_data)): 
    if ((t < TIME_MIN) | (t >= TIME_MAX)):
        continue
    
    pixel_count = 0
    total_t2m = 0

    for y in range(len(t2m_data[0])):
        if ((y < LAT_MIN) | (y >= LAT_MAX)):
            continue

        for x in range(len(t2m_data[0][0])):
            if ((x < LON_MIN) | (x >= LON_MAX)):
                continue
            
            pixel_count = pixel_count + 1
            total_t2m = total_t2m + t2m_data[t][y][x]

    temporal_spatial_mean.append(total_t2m / pixel_count)

# Calculate temporally varying spatial standard deviation
for t in range(len(t2m_data)):
    if ((t < TIME_MIN) | (t >= TIME_MAX)):
        continue

    pixel_count = 0
    diff_squared_sum = 0

    for y in range(len(t2m_data[0])):
        if ((y < LAT_MIN) | (y >= LAT_MAX)):
            continue

        for x in range(len(t2m_data[0][0])):
            if ((x < LON_MIN) | (x >= LON_MAX)):
                continue

            pixel_count = pixel_count + 1
            diff = t2m_data[t][y][x] - temporal_spatial_mean[t - TIME_MIN]
            diff_squared_sum = diff_squared_sum + (diff)

    temporal_spatial_variance.append(diff_squared_sum / pixel_count)

#Convert lists to arrays
temporal_spatial_mean = np.array(temporal_spatial_mean)
temporal_spatial_variance = np.array(temporal_spatial_variance)

#Visualize the data
plt.plot(temporal_spatial_mean)
plt.plot(temporal_spatial_variance)
plt.show()

#Output the data to netcdf
file_pointer_output = nc.Dataset('out.nc','w')
file_pointer_output.createDimension('t',TIME_MAX-TIME_MIN)

var_v1 = file_pointer_output.createVariable('temporal_spatial_mean','f4',('t',))
var_v1[:] = temporal_spatial_mean

var_v2 = file_pointer_output.createVariable('temporal_spatial_variance','f4',('t',))
var_v2[:] = temporal_spatial_variance

file_pointer_output.close()
file_pointer_input.close()

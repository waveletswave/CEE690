import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force Matplotlib to not use any X-Windows backend
import matplotlib.pyplot as plt


def calculate_spatial_mean(data,time_min,time_max,lat_min,lat_max,lon_min,lon_max):

    # Define the final variable as a list
    temporal_spatial_mean = []

    # Calculate temporally varying spatial mean
    for t in range(data.shape[0]): 
        if ((t < time_min) | (t >= time_max)):
            continue
        
        pixel_count = 0
        total_data = 0

        for y in range(data.shape[1]):
            if ((y < lat_min) | (y >= lat_max)):
                continue

            for x in range(data.shape[2]):
                if ((x < lon_min) | (x >= lon_max)):
                    continue
                
                pixel_count = pixel_count + 1
                total_data = total_data + data[t][y][x]

        temporal_spatial_mean.append(total_data / pixel_count)

    return np.array(temporal_spatial_mean)

def calculate_spatial_variance(data,time_min,time_max,lat_min,lat_max,lon_min,lon_max,temporal_spatial_mean):

    # Define the final variable as a list
    temporal_spatial_variance = []

    # Calculate temporally varying spatial mean
    for t in range(data.shape[0]): 
        if ((t < time_min) | (t >= time_max)):
            continue
        
        pixel_count = 0
        diff_squared_sum = 0

        for y in range(data.shape[1]):
            if ((y < lat_min) | (y >= lat_max)):
                continue

            for x in range(data.shape[2]):
                if ((x < lon_min) | (x >= lon_max)):
                    continue
                
                pixel_count = pixel_count + 1
                diff =  data[t][y][x] - temporal_spatial_mean[t - time_min]
                diff_squared_sum = diff_squared_sum + (diff)

        temporal_spatial_variance.append(diff_squared_sum / pixel_count)

    return np.array(temporal_spatial_variance)

def load_dataset(input_file,var_name):

    # Load dataset from netcdf file
    file_pointer_input = nc.Dataset(input_file,'r') 
    t2m_data = file_pointer_input.variables[var_name][:]
    file_pointer_input.close()

    return t2m_data

def visualize_data(temporal_spatial_mean,temporal_spatial_variance,plot_file):

    # Plot and save the time series
    plt.plot(temporal_spatial_mean, label="Mean")
    plt.plot(temporal_spatial_variance, label="Variance")
    plt.legend()
    plt.savefig(plot_file)  # Saves directly to disk
    plt.close()

    return

def output_data_to_netcdf(output_file,temporal_spatial_mean,temporal_spatial_variance):

    # Output the data to a netcdf file
    file_pointer_output = nc.Dataset('out.nc','w')
    file_pointer_output.createDimension('t',temporal_spatial_mean.shape[0])

    var_v1 = file_pointer_output.createVariable('temporal_spatial_mean','f4',('t',))
    var_v1[:] = temporal_spatial_mean

    var_v2 = file_pointer_output.createVariable('temporal_spatial_variance','f4',('t',))
    var_v2[:] = temporal_spatial_variance

    file_pointer_output.close()

    return

def main():

    """
    The director of the orchestra. When this function is called, it runs the defined
    sequence of functions. However, it also ensures that other parts of the script can 
    be accessed without running this.
    """

    # Configuration variables
    INPUT_FILE = 'era_interim_monthly_197901_201512_upscaled_annual.nc'
    OUTPUT_FILE = 'out.nc'
    PLOT_FILE = 'plot.png'
    VAR_NAME = 't2m'
    LAT_MIN = 5
    LAT_MAX = 50
    LON_MIN = 10
    LON_MAX = 100
    TIME_MIN = 0
    TIME_MAX = 10

    # Load dataset
    print("Loading the dataset")
    t2m_data = load_dataset(INPUT_FILE,VAR_NAME)

    # Compute temporal series of spatial mean and spatial standard deviation
    print("Computing the statistics")
    temporal_spatial_mean = calculate_spatial_mean(t2m_data,TIME_MIN,TIME_MAX,
                                                   LAT_MIN,LAT_MAX,LON_MIN,LON_MAX)
    temporal_spatial_variance = calculate_spatial_variance(t2m_data,TIME_MIN,TIME_MAX,
                                                           LAT_MIN,LAT_MAX,LON_MIN,LON_MAX,
                                                           temporal_spatial_mean)

    #Visualize the data
    print("Visualizing the data")
    visualize_data(temporal_spatial_mean,temporal_spatial_variance,PLOT_FILE)

    #Output the data to netcdf
    print("Saving the computed statistics to netcdf")
    output_data_to_netcdf(OUTPUT_FILE,temporal_spatial_mean,temporal_spatial_variance)

    return

if __name__ == "__main__":

    main()

"""
This script provides some basic spatial stats to compute on
some input data and spatial and temporal coordinates.
"""
import argparse
import json
import netCDF4 as nc
import numpy as np
import matplotlib
import os
import sys
matplotlib.use('Agg') # Force Matplotlib to not use any X-Windows backend
import matplotlib.pyplot as plt


def calculate_spatial_mean(data, config):

    # Define the final variable as a list
    temporal_spatial_mean = []

    # Validate indices against data shape
    if config['TIME_MAX'] > data.shape[0] or config['LAT_MAX'] > data.shape[1] or config['LON_MAX'] > data.shape[2]:
        print("Error: Configuration indices are out of data bounds.")
        sys.exit(1)

    # Calculate temporally varying spatial mean
    for t in range(data.shape[0]): 
        if ((t < config['TIME_MIN']) | (t >= config['TIME_MAX'])):
            continue
        
        pixel_count = 0
        total_data = 0

        for y in range(data.shape[1]):
            if ((y < config['LAT_MIN']) | (y >= config['LAT_MAX'])):
                continue

            for x in range(data.shape[2]):
                if ((x < config['LON_MIN']) | (x >= config['LON_MAX'])):
                    continue
                
                pixel_count = pixel_count + 1
                total_data = total_data + data[t][y][x]

        temporal_spatial_mean.append(total_data / pixel_count)

    return np.array(temporal_spatial_mean)

def calculate_spatial_variance(data, config, temporal_spatial_mean):

    # Define the final variable as a list
    temporal_spatial_variance = []

    # Calculate temporally varying spatial mean
    for t in range(data.shape[0]): 
        if ((t < config['TIME_MIN']) | (t >= config['TIME_MAX'])):
            continue
        
        pixel_count = 0
        diff_squared_sum = 0

        for y in range(data.shape[1]):
            if ((y < config['LAT_MIN']) | (y >= config['LAT_MAX'])):
                continue

            for x in range(data.shape[2]):
                if ((x < config['LON_MIN']) | (x >= config['LON_MAX'])):
                    continue
                
                pixel_count = pixel_count + 1
                diff =  data[t][y][x] - temporal_spatial_mean[t - config['TIME_MIN']]
                diff_squared_sum = diff_squared_sum + (diff)

        temporal_spatial_variance.append(diff_squared_sum / pixel_count)

    return np.array(temporal_spatial_variance)

def load_dataset(config):

    # Check if file exists
    if not os.path.exists(config['INPUT_FILE']):
        print(f"Error: The file {config['INPUT_FILE']} does not exist.")
        sys.exit(1)

    try:
        # Load dataset from netcdf file
        file_pointer_input = nc.Dataset(config['INPUT_FILE'],'r') 
        
        # Check if variable exists in file
        if config['VAR_NAME'] not in file_pointer_input.variables:
            print(f"Error: Variable '{config['VAR_NAME']}' not found in {config['INPUT_FILE']}.")
            file_pointer_input.close()
            sys.exit(1)

        t2m_data = file_pointer_input.variables[config['VAR_NAME']][:]
        file_pointer_input.close()
        return t2m_data

    except Exception as e:
        print(f"An unexpected error occurred while loading the dataset: {e}")
        sys.exit(1)

def visualize_data(temporal_spatial_mean, temporal_spatial_variance, config):

    try:
        # Plot and save the time series
        plt.plot(temporal_spatial_mean, label="Mean")
        plt.plot(temporal_spatial_variance, label="Variance")
        plt.legend()
        plt.savefig(config['PLOT_FILE'])  # Saves directly to disk
        plt.close()
    except Exception as e:
        print(f"Error during visualization: {e}")

    return

def output_data_to_netcdf(output_file, temporal_spatial_mean, temporal_spatial_variance):

    try:
        # Output the data to a netcdf file
        file_pointer_output = nc.Dataset(output_file,'w')
        file_pointer_output.createDimension('t',temporal_spatial_mean.shape[0])

        var_v1 = file_pointer_output.createVariable('temporal_spatial_mean','f4',('t',))
        var_v1[:] = temporal_spatial_mean

        var_v2 = file_pointer_output.createVariable('temporal_spatial_variance','f4',('t',))
        var_v2[:] = temporal_spatial_variance

        file_pointer_output.close()
    except Exception as e:
        print(f"Error while saving NetCDF: {e}")

    return

def main():

    """
    The director of the orchestra. When this function is called, it runs the defined
    sequence of functions. However, it also ensures that other parts of the script can 
    be accessed without running this.
    """

    # Gather arguments from the terminal and convert to dictionary
    config = vars(get_args())

    # Override with json info if present
    if config.get('JSON_FILE'):
        try:
            with open(config['JSON_FILE'], 'r') as f:
                json_config = json.load(f)
                config.update(json_config)
        except Exception as e:
            print(f"Error loading JSON config: {e}")
            sys.exit(1)

    # Load dataset
    print("Loading the dataset")
    t2m_data = load_dataset(config)

    # Compute temporal series of spatial mean and spatial standard deviation
    print("Computing the statistics")
    temporal_spatial_mean = calculate_spatial_mean(t2m_data, config)
    temporal_spatial_variance = calculate_spatial_variance(t2m_data, config, temporal_spatial_mean)

    #Visualize the data
    print("Visualizing the data")
    visualize_data(temporal_spatial_mean, temporal_spatial_variance, config)

    #Output the data to netcdf
    print("Saving the computed statistics to netcdf")
    output_data_to_netcdf(config['OUTPUT_FILE'], temporal_spatial_mean, temporal_spatial_variance)

    return

def get_args():
    """Defines and collects command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process spatial statistics from netCDF4 data."
    )
    
    # Input/Output paths
    parser.add_argument('--INPUT_FILE', type=str, default='era_interim_monthly_197901_201512_upscaled_annual.nc', 
                        help='Path to the input NetCDF file')
    parser.add_argument('--OUTPUT_FILE', type=str, default='out.nc', 
                        help='Name of the output NetCDF file')
    parser.add_argument('--PLOT_FILE', type=str, default='plot.png', 
                        help='Name of the output diagnostic plot')
    parser.add_argument('--VAR_NAME', type=str, default='t2m', 
                        help='The variable name to extract from the NetCDF')
    
    # Bounding Box Arguments
    parser.add_argument('--LAT_MIN', type=int, default=5)
    parser.add_argument('--LAT_MAX', type=int, default=50)
    parser.add_argument('--LON_MIN', type=int, default=10)
    parser.add_argument('--LON_MAX', type=int, default=100)
    parser.add_argument('--TIME_MIN', type=int, default=0)
    parser.add_argument('--TIME_MAX', type=int, default=10)

    # Optional JSON config file path
    parser.add_argument('--JSON_FILE', type=str, default=None)
    
    return parser.parse_args()

if __name__ == "__main__":
    main()

"""
This script provides an Object-Oriented approach to computing spatial stats
on NetCDF input data across spatial and temporal coordinates.
"""
import argparse
import json
import os
import sys
import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SpatialAnalyzer:
    def __init__(self, config):
        """
        Initializes the analyzer with configuration and loads the dataset.
        In OOP, __init__ is the 'constructor' that sets up the object's state.
        """
        self.config = config
        self.data = self._load_dataset()
        self.means = None
        self.variances = None

    def _load_dataset(self):
        """Internal helper to load and validate data."""
        if not os.path.exists(self.config['INPUT_FILE']):
            print(f"Error: The file {self.config['INPUT_FILE']} does not exist.")
            sys.exit(1)

        try:
            file_pointer = nc.Dataset(self.config['INPUT_FILE'], 'r')
            if self.config['VAR_NAME'] not in file_pointer.variables:
                print(f"Error: Variable '{self.config['VAR_NAME']}' not found.")
                file_pointer.close()
                sys.exit(1)

            data = file_pointer.variables[self.config['VAR_NAME']][:]
            file_pointer.close()
            return data
        except Exception as e:
            print(f"An unexpected error occurred while loading: {e}")
            sys.exit(1)

    def run_analysis(self):
        """Orchestrates the computation of statistics."""
        print("Computing the statistics")
        t_start, t_end = self.config['TIME_MIN'], self.config['TIME_MAX']

        # Slicing the subset once to be used by both mean and variance
        subset = self.data[t_start:t_end,
                           self.config['LAT_MIN']:self.config['LAT_MAX'],
                           self.config['LON_MIN']:self.config['LON_MAX']]

        self.means = np.mean(subset, axis=(1, 2))
        self.variances = np.var(subset, axis=(1, 2))

    def visualize(self):
        """Generates the diagnostic plot."""
        if self.means is None:
            print("Error: No analysis results to visualize.")
            return

        print("Visualizing the data")
        try:
            plt.plot(self.means, label="Mean")
            plt.plot(self.variances, label="Variance")
            plt.legend()
            plt.savefig(self.config['PLOT_FILE'])
            plt.close()
        except Exception as e:
            print(f"Error during visualization: {e}")

    def save_netcdf(self):
        """Exports results to a NetCDF file."""
        print("Saving the computed statistics to netcdf")
        try:
            file_out = nc.Dataset(self.config['OUTPUT_FILE'], 'w')
            file_out.createDimension('t', self.means.shape[0])

            var_v1 = file_out.createVariable('temporal_spatial_mean', 'f4', ('t',))
            var_v1[:] = self.means

            var_v2 = file_out.createVariable('temporal_spatial_variance', 'f4', ('t',))
            var_v2[:] = self.variances

            file_out.close()
        except Exception as e:
            print(f"Error while saving NetCDF: {e}")

def get_args():
    """Defines and collects command line arguments."""
    parser = argparse.ArgumentParser(description="Process spatial statistics from netCDF4 data.")
    parser.add_argument('--INPUT_FILE', type=str, 
                        default='era_interim_monthly_197901_201512_upscaled_annual.nc')
    parser.add_argument('--OUTPUT_FILE', type=str, default='out.nc')
    parser.add_argument('--PLOT_FILE', type=str, default='plot.png')
    parser.add_argument('--VAR_NAME', type=str, default='t2m')
    parser.add_argument('--LAT_MIN', type=int, default=5)
    parser.add_argument('--LAT_MAX', type=int, default=50)
    parser.add_argument('--LON_MIN', type=int, default=10)
    parser.add_argument('--LON_MAX', type=int, default=100)
    parser.add_argument('--TIME_MIN', type=int, default=0)
    parser.add_argument('--TIME_MAX', type=int, default=10)
    parser.add_argument('--JSON_FILE', type=str, default=None)
    return parser.parse_args()

def main():
    # 1. Setup Configuration
    config = vars(get_args())
    if config.get('JSON_FILE'):
        try:
            with open(config['JSON_FILE'], 'r') as f:
                config.update(json.load(f))
        except Exception as e:
            print(f"Error loading JSON config: {e}")
            sys.exit(1)

    # 2. OOP Execution
    # Create the object (this loads the data)
    analyzer = SpatialAnalyzer(config)
    
    # Run the methods
    analyzer.run_analysis()
    analyzer.visualize()
    analyzer.save_netcdf()

    return

if __name__ == "__main__":
    main()

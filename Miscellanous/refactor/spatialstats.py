"""
Spatial Statistics Analyzer
===========================

This module provides an Object-Oriented interface for processing weather and climate data
stored in NetCDF format. It handles data loading, statistical computation
(spatial mean and variance), visualization, and data export.

classes
-------
SpatialAnalyzer
    The main driver class for the load, transform, save, and plot pipeline.
"""

import argparse
import json
import netCDF4 as nc
import numpy as np
import matplotlib
import os
import sys

# use 'Agg' backend to allow plotting on headless servers (HPC clusters)
# without requiring an X11 window system.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

class SpatialAnalyzer:
    """
    A class to handle the loading, processing, and visualization of spatial
    climate data.

    Attributes
    ----------
    config : dict
        Configuration dictionary containing file paths and slice indices.
    data : numpy.ndarray
        The loaded 3D climate and weather dataset (Time x Latitude x Longitude).
    means : numpy.ndarray or None
        The computed spatial means over time.
    variances : numpy.ndarray or None
        The computed spatial variances over time.
    """

    def __init__(self, config):
        """
        Initialize the analyzer with configuration parameters and load data.

        Parameters
        ----------
        config : dict
            A dictionary containing the following keys:
            - 'INPUT_FILE': Path to source NetCDF.
            - 'VAR_NAME': Variable name inside the NetCDF (e.g., 't2m').
            - 'TIME_MIN', 'TIME_MAX': Indices for time slicing.
            - 'LAT_MIN', 'LAT_MAX': Indices for latitude slicing.
            - 'LON_MIN', 'LON_MAX': Indices for longitude slicing.
        """
        self.config = config
        # Load data immediately upon instantiation
        self.data = self._load_dataset()
        self.means = None
        self.variances = None

    def _load_dataset(self):
        """
        Internal helper to load and validate the NetCDF dataset.

        Returns
        -------
        numpy.ndarray
            The raw data array extracted from the NetCDF file.

        Raises
        ------
        SystemExit
            If the file does not exist or the variable is missing.
        """
        # 1. Validation: Ensure file exists before trying to open
        if not os.path.exists(self.config['INPUT_FILE']):
            print(f"Error: The file {self.config['INPUT_FILE']} does not exist.")
            sys.exit(1)

        try:
            # Open in read-only mode to prevent accidental corruption
            file_pointer = nc.Dataset(self.config['INPUT_FILE'], 'r') 
            
            # 2. Validation: Ensure target variable exists
            if self.config['VAR_NAME'] not in file_pointer.variables:
                print(f"Error: Variable '{self.config['VAR_NAME']}' not found.")
                file_pointer.close()
                sys.exit(1)

            # Load the actual data into memory
            data = file_pointer.variables[self.config['VAR_NAME']][:]
            file_pointer.close()
            return data

        except Exception as e:
            print(f"An unexpected error occurred while loading: {e}")
            sys.exit(1)

    def run_analysis(self):
        """
        Compute spatial statistics (mean and variance) over the specified dimensions.

        This method uses NumPy vectorization to avoid slow Python loops.
        It reduces the 3D array (Time, Lat, Lon) into 1D arrays (Time)
        by aggregating over the spatial axes (1 and 2).
        """
        print("Computing the statistics")
        t_start, t_end = self.config['TIME_MIN'], self.config['TIME_MAX']
        
        # Optimization: Slice the subset ONCE to avoid repeated indexing overhead
        # This reduces memory access time significantly compared to slicing inside a loop.
        subset = self.data[t_start:t_end, 
                           self.config['LAT_MIN']:self.config['LAT_MAX'], 
                           self.config['LON_MIN']:self.config['LON_MAX']]

        # Vectorized calculation: Collapsing axes 1 (Lat) and 2 (Lon)
        self.means = np.mean(subset, axis=(1, 2))
        self.variances = np.var(subset, axis=(1, 2))

    def visualize(self):
        """
        Generate a diagnostic plot of the computed statistics.

        Saves a PNG file comparing the trend of Mean vs Variance over time.
        Uses the file path specified in config['PLOT_FILE'].
        """
        if self.means is None:
            print("Error: No analysis results to visualize. Run run_analysis() first.")
            return

        print("Visualizing the data")
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.means, label="Spatial Mean")
            plt.plot(self.variances, label="Spatial Variance")
            plt.title(f"Statistics for {self.config['VAR_NAME']}")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()
            
            # Save to disk instead of showing, compatible with headless clusters
            plt.savefig(self.config['PLOT_FILE'])
            plt.close()
            print(f"Plot saved to {self.config['PLOT_FILE']}")
        except Exception as e:
            print(f"Error during visualization: {e}")

    def save_netcdf(self):
        """
        Export the computed statistics to a new NetCDF file.

        The output file will contain two variables:
        - 'temporal_spatial_mean'
        - 'temporal_spatial_variance'
        Both are indexed by a new dimension 't'.
        """
        print(f"Saving statistics to {self.config['OUTPUT_FILE']}")
        try:
            file_out = nc.Dataset(self.config['OUTPUT_FILE'], 'w')
            
            # Create a dimension for Time (unlimited or fixed size based on data)
            file_out.createDimension('t', self.means.shape[0])

            # Create and populate variables
            var_v1 = file_out.createVariable('temporal_spatial_mean', 'f4', ('t',))
            var_v1[:] = self.means

            var_v2 = file_out.createVariable('temporal_spatial_variance', 'f4', ('t',))
            var_v2[:] = self.variances

            file_out.close()
        except Exception as e:
            print(f"Error while saving NetCDF: {e}")

def get_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        The populated namespace of arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process spatial statistics from NetCDF climate data."
    )
    # I/O Arguments
    parser.add_argument('--INPUT_FILE', type=str, 
                        default='era_interim_monthly_197901_201512_upscaled_annual.nc',
                        help="Path to the input NetCDF file.")
    parser.add_argument('--OUTPUT_FILE', type=str, default='out.nc',
                        help="Path to save the output NetCDF file.")
    parser.add_argument('--PLOT_FILE', type=str, default='plot.png',
                        help="Path to save the diagnostic plot.")
    parser.add_argument('--VAR_NAME', type=str, default='t2m',
                        help="Name of the variable to analyze (e.g., 't2m', 'precip').")
    
    # Slicing Arguments
    parser.add_argument('--LAT_MIN', type=int, default=5, help="Starting latitude index.")
    parser.add_argument('--LAT_MAX', type=int, default=50, help="Ending latitude index.")
    parser.add_argument('--LON_MIN', type=int, default=10, help="Starting longitude index.")
    parser.add_argument('--LON_MAX', type=int, default=100, help="Ending longitude index.")
    parser.add_argument('--TIME_MIN', type=int, default=0, help="Starting time index.")
    parser.add_argument('--TIME_MAX', type=int, default=10, help="Ending time index.")
    
    # Config File Argument
    parser.add_argument('--JSON_FILE', type=str, default=None,
                        help="Path to a JSON config file. Overrides CLI defaults.")
    
    return parser.parse_args()

def main():
    """
    Main execution entry point.
    
    Workflow:
    1. Parse CLI arguments.
    2. If a JSON file is provided, update/override CLI args with JSON values.
    3. Initialize the SpatialAnalyzer.
    4. Run analysis, visualization, and export.
    """
    # 1. Setup Configuration
    config = vars(get_args())

    # Priority: JSON config > CLI defaults
    if config.get('JSON_FILE'):
        if os.path.exists(config['JSON_FILE']):
            try:
                with open(config['JSON_FILE'], 'r') as f:
                    config.update(json.load(f))
                print(f"Configuration loaded from {config['JSON_FILE']}")
            except Exception as e:
                print(f"Error loading JSON config: {e}")
                sys.exit(1)
        else:
            print(f"Warning: JSON file {config['JSON_FILE']} not found. Using defaults.")

    # 2. OOP Execution
    try:
        analyzer = SpatialAnalyzer(config)
        analyzer.run_analysis()
        analyzer.visualize()
        analyzer.save_netcdf()
        print("Processing complete.")
    except Exception as e:
        print(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

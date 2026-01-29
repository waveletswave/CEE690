# Spatial Statistics Analyzer

A Python tool for efficient processing of climate data stored in NetCDF format. This script calculates spatial means and variances over user-defined temporal and spatial subsets using vectorized NumPy operations.

## Features

* **Vectorized Performance:** Uses NumPy optimization to process large datasets without slow Python loops.
* **HPC Ready:** Uses the `Agg` matplotlib backend, allowing plot generation on headless servers.
* **Flexible Configuration:** Supports both Command Line Arguments and JSON configuration files.
* **Standardized Output:** Produces both a visualization (PNG) and a data file (NetCDF).

## Requirements

* Python 3.8+
* `netCDF4`
* `numpy`
* `matplotlib`

Install dependencies via pip:
```bash
pip install netCDF4 numpy matplotlib

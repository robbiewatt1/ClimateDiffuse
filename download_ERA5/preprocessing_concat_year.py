import os 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import random
import argparse

# Get arguments from argparser
parser = argparse.ArgumentParser()
## Arguments: year
parser.add_argument('--year', metavar='year', type=int)
parser.add_argument('--remove_files', metavar='remove_files', action=argparse.BooleanOptionalAction)

datadir = "./data/"

## Provide year and month as input to this file using args
args = parser.parse_args()
year = args.year
remove_files = args.remove_files

# Open first month
month="01"
filename = f"samples_{year}{month}.nc"
path_to_file = f"{datadir}{filename}"
ds = xr.open_dataset(path_to_file, engine="netcdf4")


for m in range(2,13):
    month = f"{m:02d}"
    filename = f"samples_{year}{month}.nc"
    path_to_file = f"{datadir}{filename}"
    ds2 = xr.open_dataset(path_to_file, engine="netcdf4")

    # Concatenate along time axis
    ds = xr.concat((ds, ds2), dim="time")

# Save
save_file = f"samples_{year}.nc"
ds.to_netcdf(f"{datadir}{save_file}")

if remove_files:
    print("Removing intermediate files")
    for m in range(1,13):
        month = f"{m:02d}"
        filename = f"samples_{year}{month}.nc"
        path_to_file = f"{datadir}{filename}"
        os.remove(f"{datadir}{filename}")


import os 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import random
import argparse

datadir = "./data/"

# Get arguments from argparser
parser = argparse.ArgumentParser()
## Arguments: year
parser.add_argument('--year', metavar='year', type=int)
parser.add_argument('--month', metavar='month', type=int)
parser.add_argument('--last_day', metavar='last_day', type=int)
parser.add_argument('--remove_files', metavar='remove_files', action=argparse.BooleanOptionalAction)

## Provide year and month as input to this file using args
args = parser.parse_args()
year = args.year
month = args.month
last_day = args.last_day
remove_files = args.remove_files

first_day=1

# Variable names of surface files
varnames= {"VAR_2T":"128_167_2t", 
           "VAR_10U":"128_165_10u", 
           "VAR_10V":"128_166_10v"}

var_keys = list(varnames.keys())
var_key = var_keys[0]

# Set random seed for reproducibility, but different for each year/month
seed = year*12 + month
print(seed)
random.seed(seed)

## First variable for setting up
varname = varnames[var_key]
filename = f"e5.oper.an.sfc.{varname}.ll025sc.{year}{month:02d}{first_day:02d}00_{year}{month:02d}{last_day}23.nc"

# Open file
path_to_file = f"{datadir}{filename}"
ds = xr.open_dataset(path_to_file, engine="netcdf4")

# Select time inds randomly 
time_inds = np.arange(len(ds.time), dtype=int)
random.shuffle(time_inds)
## Select 30 time inds from this month
time_inds = time_inds[0:30]

# Pre-processed dataset
ds_proc = ds.isel(time=time_inds)

## Open next vars and add them to the dataset.
for var_key in var_keys[1:]:
    varname = varnames[var_key]
    filename = f"e5.oper.an.sfc.{varname}.ll025sc.{year}{month:02d}{first_day:02d}00_{year}{month:02d}{last_day}23.nc"

    # Open file
    path_to_file = f"{datadir}{filename}"
    ds = xr.open_dataset(path_to_file, engine="netcdf4")

    # Pre-processed dataset and add to existing
    ds_proc2 = ds.isel(time=time_inds)
    ds_proc = xr.merge((ds_proc, ds_proc2))


save_file = f"samples_{year}{month:02d}.nc"
ds_proc.to_netcdf(f"{datadir}{save_file}")

if remove_files:
    print("Removing intermediate files")
    for var_key in var_keys:
        varname = varnames[var_key]
        filename = f"e5.oper.an.sfc.{varname}.ll025sc.{year}{month:02d}{first_day:02d}00_{year}{month:02d}{last_day}23.nc"
        os.remove(f"{datadir}{filename}")

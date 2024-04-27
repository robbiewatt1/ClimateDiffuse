import sys
import matplotlib.pyplot as plt
import cartopy
import torch
from torch.utils.data.dataloader import DataLoader

sys.path.append('../src/')
from DatasetUS import *
from Inference import *

## Saves US dataset over test years for quicker analysis
## Test years: 2018-2022
year_start = 2021
year_end = 2021

## Dirs
data_dir="/home/Everyone/ERA5/data/"
model_dir="/home/Everyone/Model_Inference/"
plot_dir="../plots/"
save_dir="../output/Truth/"

# Code identical to dataset, but we keep things in xarray format
filenames = [f"samples_{year}.nc" for year in range(year_start, year_end)]

filename0 = filenames[0]
path_to_file = data_dir + filename0
ds = xr.open_dataset(path_to_file, engine="netcdf4")


varnames = ["temp", "u-comp wind", "v-comp wind"]
n_var = len(varnames)

# Select domain with size 256 x 128 (W x H)
ds_US = ds.sel(latitude=slice(54.5, 22.6),  # latitude is ordered N to S
               longitude=slice(233.6, 297.5))  # longitude ordered E to W
lon = ds_US.longitude  # len 256
lat = ds_US.latitude  # len 128

# Concatenate other files
for filename in filenames[1:]:
    path_to_file = data_dir + filename
    ds = xr.open_dataset(path_to_file, engine="netcdf4")
    # Select domain
    ds_US = xr.concat((ds_US,
                       ds.sel(latitude=slice(54.5, 22.6),  # latitude is ordered N to S
                              longitude=slice(233.6, 297.5))),  # longitude ordered E to W
                       dim="time")


print(ds_US)

## Save to netcdf4
save_filename = f"{save_dir}/samples_{year_start}-{year_end}.nc"
ds_US.to_netcdf(save_filename)
print(f"Saved as {save_filename}")

import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from compute_spectrum import compute_spectrum2d

# Models
models = ["Truth", "Diffusion", "UNet", "LinearInterpolation"]

year_start = 2018
year_end = 2023

# Variables - defines three separate subplots
varnames = ["VAR_2T", "VAR_10U", "VAR_10V"]
vmin = [250, -10, -10]
vmax = [300, 10, 10]
vmax_stds = [1, 1, 1]
cmaps = ["rainbow", "BrBG_r", "BrBG_r"]

model = "Diffusion"
rngs = range(0, 30)
n_samples = len(rngs)

plot_dir = "../output/plots/diffusion_pred/"
# First, get the truth
filename = f"../output/Truth/samples_{year_start}-{year_end}.nc"
ds = xr.open_dataset(filename, engine="netcdf4")
truth = xr.concat([ds[varname] for varname in varnames], dim="var")
lat = ds.latitude
lon = ds.longitude
time = ds.time

ntime, nlat, nlon = len(time), len(lat), len(lon)
print(truth.shape)
# Get coarse version / Linear interp
filename = f"../output/LinearInterpolation/samples_{year_start}-{year_end}.nc"
ds = xr.open_dataset(filename, engine="netcdf4")
coarse = xr.concat([ds[varname] for varname in varnames], dim="var")

# Get UNet
filename = f"../output/UNet/samples_{year_start}-{year_end}.nc"
ds = xr.open_dataset(filename, engine="netcdf4")
unet = xr.concat([ds[varname] for varname in varnames], dim="var")

# Get Diffusion
# Get UNet
rng=0
filename = f"../output/Diffusion/samples{rng}_{year_start}-{year_end}.nc"
ds = xr.open_dataset(filename, engine="netcdf4")
diffusion = xr.concat([ds[varname] for varname in varnames], dim="var")

for rng in range(20):
    filename = f"../output/Diffusion/samples{rng}_{year_start}-{year_end}.nc"
    ds = xr.open_dataset(filename, engine="netcdf4")
    diffusion = xr.concat([ds[varname] for varname in varnames], dim="var")
    if rng==0:
        diffusion_all = diffusion
    else:
        diffusion_all = xr.concat((diffusion_all, diffusion), dim="ens")
diffusion_mean = diffusion_all.mean(dim="ens")
diffusion_std = diffusion_all.std(dim="ens")


# Plot
plot_varnames = ["Temperature", "Zonal wind", "Meridional wind"]
plot_var_labels = ["K", "m/s", "m/s"]
plt.rcParams.update({'font.size': 18})

# Plot all 30 preds for a selection of timesteps
for t, timestep in enumerate(time[::60]):
    plt.clf()
    fig, axs = plt.subplots(1,3, figsize=(16, 2.4), # 16, 10.2
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            gridspec_kw={'wspace': 0.1,
                                         'hspace': 0.1})
    for i, varname in enumerate(varnames):
        # Plot truth for first plot
        ax = axs[i]
        plt.sca(ax)
        ax.coastlines()
        ax.add_feature(cartopy.feature.LAKES, edgecolor='black', facecolor='none')
        pcm = plt.pcolormesh(lon, lat, diffusion_std[i, t],
                       vmin=0., vmax=vmax_stds[i],
                       shading='nearest',
                       cmap="YlOrRd",)
        if i==0:
            plt.text(lon[0]-2, lat[len(lat) // 2], f"Diffusion Std", transform=ccrs.PlateCarree(),
                rotation='vertical', ha='right', va='center', zorder=10)

        cax = axs[i].inset_axes([0., -0.25, 1, 0.1])
        plt.colorbar(pcm, cax = cax, orientation="horizontal", label=f"{plot_var_labels[i]}")

    # add labels
    axs_flat = axs.flatten()
    labels = ["m)", "n)", "o)"]

    for i in range(len(axs_flat)):
        plt.text(x=-0.08, y=1.02, s=labels[i],
                 fontsize=16, transform=axs_flat[i].transAxes)

    plt.tight_layout()
    save_filename = f"{plot_dir}/diffusion_std_{timestep.values}.png"
    plt.savefig(save_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved as {save_filename}")






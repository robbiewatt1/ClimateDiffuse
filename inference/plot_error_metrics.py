import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from CRPS import crps

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from compute_spectrum import compute_spectrum2d

# Models
models = ["Truth", "Diffusion", "UNet", "LinearInterpolation"]

# Variables - defines three separate subplots
varnames = ["VAR_2T", "VAR_10U", "VAR_10V"]
vmax = [1, 1, 1]

# For diffusion
rngs = range(0, 30)
n_ens = len(rngs)

plot_dir = "../output/plots/"
year_start = 2018
year_end = 2023


data = {}

# Loop over data
for m, model in enumerate(models):
    print(model)
    # Loop over ensembles for diffusion (probabilistic)
    if model == "Diffusion":
        ntime, nlat, nlon = data["Truth"]["VAR_2T"].shape
        diffusion = np.zeros((3, n_ens, ntime, nlat, nlon))

        # Loop over data
        for r, rng in enumerate(rngs):
            filename = f"../output/{model}/samples{rng}_{year_start}-{year_end}.nc"
            ds = xr.open_dataset(filename, engine="netcdf4")

            for i, varname in enumerate(varnames):
                diffusion[i, r] = ds[varname].to_numpy()

        data["DiffusionEns"] = {}
        data["DiffusionMean"] = {}
        for i, varname in enumerate(varnames):
            data["DiffusionEns"][varname] = diffusion[i]
            data["DiffusionMean"][varname] = diffusion[i].mean(axis=0)
    else:
        filename = f"../output/{model}/samples_{year_start}-{year_end}.nc"
        ds = xr.open_dataset(filename, engine="netcdf4")

        data[model] = {}
        for i, varname in enumerate(varnames):
            data[model][varname] = ds[varname].to_numpy()


lon = ds.longitude
lat = ds.latitude
nlat, nlon = len(lat), len(lon)

# Get areas
area_weights = np.cos(np.deg2rad(lat.to_numpy()))
area_weights = np.repeat(area_weights[:, None], nlon, axis=1)


## Error metrics:
MAE_diffusion = np.zeros((3, 128, 256))
MAE_UNet = np.zeros((3, 128, 256))
MAE_linearinterp = np.zeros((3, 128, 256))

RMSE_diffusion = np.zeros((3, 128, 256))
RMSE_UNet = np.zeros((3, 128, 256))
RMSE_linearinterp = np.zeros((3, 128, 256))

CRPS_diffusion = np.zeros((3, 128, 256))

print("Calc errors")
for i, varname in enumerate(varnames):
    print(varname)
    MAE_diffusion[i] = np.mean(np.abs( data["DiffusionMean"][varname] - data["Truth"][varname] ), axis=0)
    MAE_UNet[i] = np.mean(np.abs( data["UNet"][varname] - data["Truth"][varname] ), axis=0)
    MAE_linearinterp[i] = np.mean(np.abs( data["LinearInterpolation"][varname] - data["Truth"][varname] ), axis=0)
    RMSE_diffusion[i] = np.sqrt(np.mean( ( data["DiffusionMean"][varname] - data["Truth"][varname] )**2, axis=0))
    RMSE_UNet[i] = np.sqrt(np.mean(( data["UNet"][varname] - data["Truth"][varname] )**2, axis=0))
    RMSE_linearinterp[i] = np.sqrt(np.mean(( data["LinearInterpolation"][varname] - data["Truth"][varname] )**2,
                                           axis=0))

    diffusion_i = data["DiffusionEns"][varname]
    diffusion_i = diffusion_i.reshape((n_ens, ntime, nlat*nlon)) # flatten x-y axis
    CRPS_diffusion_flat = crps(data["Truth"][varname].reshape((ntime, nlat*nlon)), diffusion_i )
    CRPS_diffusion[i] = CRPS_diffusion_flat.reshape((nlat, nlon))

print(MAE_diffusion.mean(), CRPS_diffusion.mean())

# Set up plots
print("Plot")
# plot MAE for temp, u, v
plot_varnames = ["Temperature", "Zonal wind", "Meridional wind"]
plot_var_labels = ["K", "m/s", "m/s"]

plt.clf()
plt.rcParams.update({'font.size': 18})

fig, axs = plt.subplots(3, 3, figsize=(16, 9),
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        gridspec_kw={'wspace': 0.1,
                                     'hspace': 0.08})

for i in range(3):
    ax = axs[0, i]
    plt.sca(ax)
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black', facecolor='none')
    pcm = plt.pcolormesh(lon, lat, MAE_UNet[i],
                         vmin=0, vmax=vmax[i],
                         cmap="YlOrRd",
                         shading='nearest')
    plt.title(plot_varnames[i])
    if i==0:
        plt.text(lon[0]-2, lat[len(lat)//2], f"U-Net MAE", transform=ccrs.PlateCarree(),
                 rotation='vertical', ha='right', va='center',  zorder=10)

    ax = axs[1, i]
    plt.sca(ax)
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black', facecolor='none')
    pcm = plt.pcolormesh(lon, lat, MAE_diffusion[i],
                   vmin=0, vmax=vmax[i],
                   cmap = "YlOrRd",
                   shading='nearest')
    if i==0:
        plt.text(lon[0]-2, lat[len(lat)//2], f"Diffusion MAE",  transform=ccrs.PlateCarree(),
                 rotation='vertical', ha='right', va='center',  zorder=10)

    ax = axs[2, i]
    plt.sca(ax)
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black', facecolor='none')
    pcm = plt.pcolormesh(lon, lat, CRPS_diffusion[i],
                         vmin=0, vmax=vmax[i],
                         cmap="YlOrRd",
                         shading='nearest')
    if i==0:
        plt.text(lon[0]-2, lat[len(lat)//2], f"Diffusion CRPS", transform=ccrs.PlateCarree(),
                 rotation='vertical', ha='right', va='center',
                 zorder=10)

    cax = axs[2, i].inset_axes([0., -0.25, 1, 0.1])
    plt.colorbar(pcm, cax=cax, orientation="horizontal", label=plot_var_labels[i])

# add labels
axs_flat = axs.flatten()
labels = ["a)", "b)", "c)",
          "d)", "e)", "f)",
          "g)", "h)", " i)"]
for i in range(len(axs_flat)):
    plt.text(x=-0.07, y=1.03, s=labels[i],
             fontsize=16, transform=axs_flat[i].transAxes)

plt.tight_layout()
save_filename = f"{plot_dir}/error_maps.png"
plt.savefig(save_filename, bbox_inches="tight")
print(f"Saved as {save_filename}")

# Area weighted means
print(MAE_diffusion.shape,area_weights.shape)
# Repeat area weights
area_weights = np.repeat(area_weights[None, :, :], 3, axis=0)
print(MAE_diffusion.shape,area_weights.shape)

MAE_diffusion = np.average(MAE_diffusion, axis=(1,2), weights=area_weights)
MAE_UNet = np.average(MAE_UNet, axis=(1,2), weights=area_weights)
MAE_linearinterp = np.average(MAE_linearinterp, axis=(1,2), weights=area_weights)

RMSE_diffusion = np.sqrt(np.average(RMSE_diffusion**2, axis=(1,2), weights=area_weights))
RMSE_UNet = np.sqrt(np.average(RMSE_UNet**2, axis=(1,2), weights=area_weights))
RMSE_linearinterp = np.sqrt(np.average(RMSE_linearinterp**2, axis=(1,2), weights=area_weights))

CRPS_diffusion = np.average(CRPS_diffusion, axis=(1,2), weights=area_weights)

print(f" Diffusion mean abs: {MAE_diffusion}")
print(f" UNet mean abs: {MAE_UNet}")
print(f" Linear Interp mean abs: {MAE_linearinterp}")

print(f" Diffusion RMSE : {RMSE_diffusion}")
print(f" UNet RMSE : {RMSE_UNet}")
print(f" Linear Interp RMSE : {RMSE_linearinterp}")

print(f" Diffusion CRPS: {CRPS_diffusion}")


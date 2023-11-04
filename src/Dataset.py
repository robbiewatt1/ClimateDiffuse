import torch
import torchvision

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

class UpscaleDataset(torch.utils.data.Dataset):
    """
    Dataset class of images with a low resolution and a high resolution counterpart
    for the US continent.
    """

    def __init__(self, data_dir, 
                 in_shape=(16, 32), out_shape=(128, 256), 
                 year_start = 1950, year_end = 2001,
                 normalize_mean = torch.Tensor([ 2.8504e+02,  4.4644e-01, -1.1768e-01]),
                 normalize_std = torch.Tensor([12.6422,  3.3157,  3.6051] ),
                 ds_mean = None):
        """
        :param path: path to the dataset
        :param in_shape: shape of the low resolution images
        :param out_shape: shape of the high resolution images
        """
        
        print("Opening files")
        self.filenames = [f"samples_{year}.nc" for year in range(year_start, year_end)]

        # Open first file for saving dimension info
        filename0 = self.filenames[0]
        path_to_file = data_dir + filename0
        ds = xr.open_dataset(path_to_file, engine="netcdf4")
        
        # Dimensions: lon, lat (global domain)
        self.lon_glob = ds.longitude
        self.lat_glob = ds.latitude
        self.varnames = ["temp", "u-comp wind", "v-comp wind"]
        self.C = 3  # Number of channels
        
        # Select domain with size 256 x 128 (W x H)
        ds_US = ds.sel(latitude=slice(54.5, 22.6),      # latitude is ordered N to S
                       longitude=slice(233.6, 297.5))   # longitude ordered E to W
        self.lon = ds_US.longitude      # len 256
        self.lat = ds_US.latitude       # len 128
        self.W = len(self.lon)          # Width 
        self.H = len(self.lat)          # Height
        
        # Concatenate other files
        for filename in self.filenames[1:]:
            path_to_file = data_dir + filename
            ds = xr.open_dataset(path_to_file, engine="netcdf4")
            # Select domain with size 256 x 128 (W x H)
            ds_US = xr.concat((ds_US,
                               ds.sel(latitude=slice(54.5, 22.6),      # latitude is ordered N to S
                                      longitude=slice(233.6, 297.5))),   # longitude ordered E to W
                               dim="time")

        print("All files accessed. Creating tensors")

        # Convert xarray dataarrays into torch Tensor
        t = ds_US.VAR_2T
        u = ds_US.VAR_10U
        v = ds_US.VAR_10V
        
        # Convert xarray dataarrays into torch Tensor (loads into memory)
        t = torch.from_numpy(t.to_numpy())
        u = torch.from_numpy(u.to_numpy())
        v = torch.from_numpy(v.to_numpy())

        # Stack into (ntime, 3, 128, 256)
        Y = torch.stack((t,u,v), dim=1)
        
        # Transforms
        # Normalize
        normalize_transform = torchvision.transforms.Normalize(normalize_mean, normalize_std)
        Y = normalize_transform(Y)

        # Coarsen
        coarsen_transform = torchvision.transforms.Resize(in_shape,
             interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
             antialias=True)
        X = coarsen_transform(Y)

        if ds_mean is None:
            X_mean = torchvision.transforms.functional.resize(
                torch.mean(X, dim=0), (Y.shape[-2], Y.shape[-1]),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True)
            self.ds_mean = torch.mean(Y, dim=0) - X_mean
        else:
            self.ds_mean = ds_mean

        # Save
        self.Y = Y.float()
        self.X = X.float()

        # Dimensions from orig to coarse
        lat_coarse_inds = np.arange(0, len(self.lat), 8, dtype=int)
        lon_coarse_inds = np.arange(0, len(self.lon), 8, dtype=int)

        self.lon_coarse = self.lon.isel(longitude=lon_coarse_inds)     # len 32
        self.lat_coarse = self.lat.isel(latitude=lat_coarse_inds)      # len 16
        
        print("Dataset initialized.")


    def __len__(self):
        """
        :return: length of the dataset
        """
        return self.X.shape[0]

    def __getitem__(self, index):
        """
        :param index: index of the dataset
        :return: input data and time data
        """
        return self.X[index], self.Y[index] 

    def plot_coarse(self, image_coarse, ax):
        plt.sca(ax)
        ax.coastlines()
        plt.contourf(self.lon_coarse, self.lat_coarse, image_coarse,
               levels = np.arange(-2., 2.1, 0.2), extend="both")

    def plot_fine(self, image_fine, ax):
        plt.sca(ax)
        ax.coastlines()
        plt.contourf(self.lon, self.lat, image_fine,
                     levels = np.arange(-2., 2.1, 0.2), extend="both")

    def plot_all_channels(self, X, Y):
        """Plots T, u, V for single image (no batch dimension)"""
        fig, axs = plt.subplots(3, 2, figsize=(8, 2*self.C),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        for i in range(3):
            self.plot_coarse(X[i], axs[i, 0])
            plt.title(self.varnames[i]+" coarse-res")
            self.plot_fine(Y[i], axs[i, 1])
            plt.title(self.varnames[i]+" fine-res")

        plt.tight_layout()
        return fig, axs

    def plot_batch(self, X, Y, Y_pred, N=3):
        """Plots u,v,T for N samples out of batch, separate
        column for coarse, predicted fine and truth fine"""
        fig, axs = plt.subplots(self.C * N, 3, figsize=(8,N*5),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        for j in range(N):
            # Plot batch
            for i in range(self.C):
                # Plot channel
                self.plot_coarse(X[j, i], axs[(j*N)+i, 0])
                #plt.title(self.varnames[i]+" coarse-res")

                self.plot_fine(Y_pred[j, i], axs[(j*N)+i, 1])
                #plt.title(self.varnames[i]+" pred fine-res")
                self.plot_fine(Y[j, i], axs[(j*N)+i, 2])
                #plt.title(self.varnames[i]+" truth fine-res")

        plt.tight_layout()
        return fig, axs

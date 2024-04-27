import torch
import torchvision

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


class UpscaleDataset(torch.utils.data.Dataset):
    """
    Dataset class of images with a low resolution and a high resolution counterpart
    over the US continent.
    """

    def __init__(self, data_dir,
                 in_shape=(16, 32), out_shape=(128, 256),
                 year_start=1950, year_end=2001,
                 normalize_rawdata_mean=torch.Tensor([2.8504e+02,  4.4536e-01, -1.1892e-01]),
                 normalize_rawdata_std=torch.Tensor([12.7438,  3.4649,  3.742]),
                 normalize_residual_mean=torch.Tensor([-9.4627e-05, -1.3833e-03, -1.5548e-03]),
                 normalize_residual_std=torch.Tensor([1.6042, 1.0221, 1.0384]),
                 constant_variables=None,
                 constant_variables_filename="ERA5_const_sfc_variables.nc"
                 ):
        """
        :param data_dir: path to the dataset directory
        :param in_shape: shape of the low resolution images
        :param out_shape: shape of the high resolution images
        :param year_start: starting year of file named samples_{year_start}.nc
        :param year_end: ending year of file named samples_{year_end}.nc
        :param normalize_mean: channel-wise mean values estimated over all samples
        for normalizing file
        :param normalize_std: channel-wise standard deviation values estimated
        over all samples for normalizing file
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
        self.n_var = len(self.varnames)


        # Select domain with size 256 x 128 (W x H)
        ds_US = ds.sel(latitude=slice(54.5, 22.6),  # latitude is ordered N to S
                       longitude=slice(233.6, 297.5))  # longitude ordered E to W
        self.lon = ds_US.longitude  # len 256
        self.lat = ds_US.latitude  # len 128
        self.nlon = self.W = len(self.lon)  # Width
        self.nlat = self.H = len(self.lat)  # Height

        # Concatenate other files
        for filename in self.filenames[1:]:
            path_to_file = data_dir + filename
            ds = xr.open_dataset(path_to_file, engine="netcdf4")
            # Select domain with size 256 x 128 (W x H)
            ds_US = xr.concat((ds_US,
                               ds.sel(latitude=slice(54.5, 22.6),  # latitude is ordered N to S
                                      longitude=slice(233.6, 297.5))),  # longitude ordered E to W
                              dim="time")

        print("All files accessed. Creating tensors")
        self.ntime = len(ds_US.time)

        # Convert xarray dataarrays into torch Tensor (loads into memory)
        t = torch.from_numpy(ds_US.VAR_2T.to_numpy()).float()
        u = torch.from_numpy(ds_US.VAR_10U.to_numpy()).float()
        v = torch.from_numpy(ds_US.VAR_10V.to_numpy()).float()

        # Stack into (ntime, 3, 128, 256), creating the fine resolution image.
        fine = torch.stack((t, u, v), dim=1)

        # Transforms
        # Coarsen
        coarsen_transform = torchvision.transforms.Resize(in_shape,
                                                          interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                          antialias=True)
        interp_transform = torchvision.transforms.Resize(out_shape,
                                                         interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                         antialias=True)
        # Coarsen fine into coarse image, but interp to keep it on same grid
        # This will be our input into NN
        coarse = interp_transform(coarsen_transform(fine))
        # Calculate residual = fine - coarse. this will be our target
        residual = fine - coarse

        # Save unnormalized coarse and fine images for plotting
        self.coarse = coarse
        self.fine = fine

        # Normalize : use raw data means for coarse image
        normalize_rawdata_transform = torchvision.transforms.Normalize(normalize_rawdata_mean, normalize_rawdata_std)
        coarse_norm = normalize_rawdata_transform(coarse)

        # use residual means for the difference between them
        normalize_residual_transform = torchvision.transforms.Normalize(normalize_residual_mean, normalize_residual_std)
        residual_norm = normalize_residual_transform(residual)

        print(normalize_residual_std.shape)
        self.inverse_normalize_residual = lambda residual_norm: ((residual_norm *
                                                                  normalize_residual_std[:, np.newaxis, np.newaxis]) +
                                                                 normalize_residual_mean[:, np.newaxis, np.newaxis])

        # Save
        self.targets = residual_norm     # targets  = normalized residual
        self.inputs = coarse_norm        # inputs   = normalized coarse

        # Define limits for plotting (plus/minus 2 sigma
        self.vmin = normalize_rawdata_mean - 2 * normalize_rawdata_std
        self.vmax = normalize_rawdata_mean + 2 * normalize_rawdata_std

        print(self.vmin, self.vmax)



        # Additional channels for constant variables
        self.constant_variables = constant_variables
        if constant_variables is not None:
            print("Opening constant variables file (e.g. land-sea mask, topography)")
            # Open file
            ds_const = xr.open_dataset(data_dir + constant_variables_filename,
                                       engine="netcdf4")
            ds_const = ds_const.sel(latitude=slice(54.5, 22.6),  # latitude is ordered N to S
                                    longitude=slice(233.6, 297.5))

            # Get torch tensors and concatenate
            self.const_var = torch.zeros((self.ntime,
                                          len(constant_variables),
                                          self.nlat,
                                          self.nlon),
                                         dtype=torch.float)

            for i, const_varname in enumerate(constant_variables):
                const_var = ds_const[const_varname]
                # normalize?
                if const_varname != "lsm":
                    print(f"Normalize {const_varname}")
                    weighted_var = const_var.weighted(np.cos(np.radians(ds_const.latitude)))
                    mean_var = weighted_var.mean()  # 2270.3596
                    std_var = weighted_var.std()  # 6149.4727
                    print(f"Mean:{mean_var}, Std{std_var}")
                    const_var = (const_var - mean_var) / std_var
                self.const_var[i] = torch.from_numpy(const_var.to_numpy()).float()
            self.inputs = torch.concatenate((self.inputs, self.const_var), dim=1)

        # Dimensions from orig to coarse
        lat_coarse_inds = np.arange(0, len(self.lat), 8, dtype=int)
        lon_coarse_inds = np.arange(0, len(self.lon), 8, dtype=int)

        self.lon_coarse = self.lon.isel(longitude=lon_coarse_inds)  # len 32
        self.lat_coarse = self.lat.isel(latitude=lat_coarse_inds)  # len 16

        # Time embeddings
        self.time = ds_US.time.dt        # in datetime format
        self.year = self.time.year
        self.month = self.time.month
        self.day = self.time.day
        self.hour = self.time.hour
        # day of year (1 to 360)
        self.doy = ((self.month - 1.) * 30 + (self.day - 1.))

        # Normalize and convert to numpy (load into mem)
        self.year_norm = (self.year.to_numpy() - 1940.)/100
        self.doy_norm = self.doy.to_numpy()/360.
        self.hour_norm = self.hour.to_numpy()/24.

        # Torch arrays and float
        self.year_norm = torch.from_numpy(self.year_norm).float()
        self.doy_norm = torch.from_numpy(self.doy_norm).float()
        self.hour_norm = torch.from_numpy(self.hour_norm).float()

        print("Dataset initialized.")

    def __len__(self):
        """
        :return: length of the dataset
        """
        return self.inputs.shape[0]

    def __getitem__(self, index):
        """
        :param index: index of the dataset
        :return: input data and time data
        """
        return {"inputs": self.inputs[index],
                "targets": self.targets[index],
                "fine": self.fine[index],
                "coarse": self.coarse[index],
                "year": self.year_norm[index],
                "doy": self.doy_norm[index],
                "hour": self.hour_norm[index]}

    def residual_to_fine_image(self, residual, coarse_image):
        return coarse_image + self.inverse_normalize_residual(residual)

    def plot_fine(self, image_fine, ax, vmin=-2, vmax=2):
        plt.sca(ax)
        ax.coastlines()
        plt.pcolormesh(self.lon, self.lat, image_fine,
                       vmin=vmin, vmax=vmax, shading='nearest')

    def plot_all_channels(self, X, Y):
        """Plots T, u, V for single image (no batch dimension)"""
        fig, axs = plt.subplots(self.n_var, 2, figsize=(8, 2 * self.n_var),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        for i in range(self.n_var):
            self.plot_fine(X[i], axs[i, 0])
            plt.title(self.varnames[i] + " coarse-res")
            self.plot_fine(Y[i], axs[i, 1])
            plt.title(self.varnames[i] + " fine-res")

        plt.tight_layout()
        return fig, axs

    def plot_batch(self, coarse_image, fine_image, fine_image_pred, N=3):
        """Plots u,v,T for N samples out of batch, separate
        column for coarse, predicted fine and truth fine"""
        fig, axs = plt.subplots(self.n_var * N, 3, figsize=(8, N * 5),
                                subplot_kw={'projection': ccrs.PlateCarree()})

        for j in range(N):
            # Plot batch
            for i in range(self.n_var):
                # Plot channel
                self.plot_fine(coarse_image[j, i], axs[(j * N) + i, 0],
                               vmin=self.vmin[i], vmax=self.vmax[i])
                self.plot_fine(fine_image_pred[j, i], axs[(j * N) + i, 1],
                               vmin=self.vmin[i], vmax=self.vmax[i])
                self.plot_fine(fine_image[j, i], axs[(j * N) + i, 2],
                               vmin=self.vmin[i], vmax=self.vmax[i])

        plt.tight_layout()
        return fig, axs

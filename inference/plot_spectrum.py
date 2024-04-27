import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from compute_spectrum import compute_spectrum2d

plt.rcParams.update({'font.size': 18})


# Models - lines shown in different colors
models = ["Truth", "Diffusion", "UNet", "LinearInterpolation"]
colors = ["black", "red", "blue", "yellowgreen"]

plot_dir = "../output/plots/"
year_start = 2018
year_end = 2023

# For diffusion
rngs = range(0, 30)
n_ens = len(rngs)

# Variables - defines three separate subplots
varnames = ["VAR_2T", "VAR_10U", "VAR_10V"]
plot_varnames = ["Temperature", "Zonal wind", "Meridional wind"]

# Set up plot
plt.clf()
fig, axs = plt.subplots(1, 3, figsize=(9, 3),
                        sharex=True, sharey=True)

spectrum_all = {}

# Loop over data
for m, model in enumerate(models):
    print(model)
    if model != "Diffusion":
        spectrum_all[model] = {}

        filename = f"../output/{model}/samples_{year_start}-{year_end}.nc"
        ds = xr.open_dataset(filename, engine="netcdf4")

        for i, varname in enumerate(varnames):
            data = ds[varname].to_numpy()

            # Compute spectrum
            kvals, Abins = compute_spectrum2d(data)
            spectrum_all[model][varname] = Abins

            plt.sca(axs[i])
            axs[i].loglog(kvals, Abins,
                          color=colors[m],
                          label=models[m],
                          alpha=0.5)
            plt.title(plot_varnames[i])
            plt.xlabel("$k$")
            plt.ylabel("$P(k)$")
    elif model == "Diffusion":

        spectrum_all["Diffusion_mean"] = {}
        spectrum_all["Diffusion_std"] = {}

        # Loop over data
        Abins_diffusion = np.zeros((3, n_ens, len(kvals)))
        for r, rng in enumerate(rngs):
            filename = f"../output/{model}/samples{rng}_{year_start}-{year_end}.nc"
            ds = xr.open_dataset(filename, engine="netcdf4")

            for i, varname in enumerate(varnames):
                data = ds[varname].to_numpy()
                kvals, Abins = compute_spectrum2d(data)
                Abins_diffusion[i, r] = Abins
        print("Calculated for all ensembles in Diffusion")

        # Plot spectrum
        for i, varname in enumerate(varnames):
            spectrum_all["Diffusion_mean"][varname] = Abins_diffusion[i].mean(axis=0)
            spectrum_all["Diffusion_std"][varname] = Abins_diffusion[i].std(axis=0)

            Abins_mean = spectrum_all["Diffusion_mean"][varname]
            Abins_std = spectrum_all["Diffusion_std"][varname]

            print(Abins_mean)
            print(Abins_std)

            plt.sca(axs[i])
            axs[i].loglog(kvals, Abins_mean,
                          color=colors[m],
                          label=models[m],
                          alpha=0.5)
            axs[i].fill_between(kvals,
                                Abins_mean - Abins_std,
                                Abins_mean + Abins_std,
                                color= colors[m], alpha=0.3)


#plt.legend()
fig_filename = f"{plot_dir}/spectrum.png"
plt.tight_layout()
plt.savefig(fig_filename, bbox_inches="tight")
print(f"Saved as {fig_filename}")

# Plot the differences
# Set up plot
plt.clf()
fig, axs = plt.subplots(2, 3, figsize=(16, 8),
                        sharex=True, sharey="row")
for m, model in enumerate(models):
    print(model, colors[m])
    if model != "Diffusion":
        for i, varname in enumerate(varnames):
            # Already computed spectrum and saved into dictionary
            Abins = spectrum_all[model][varname]

            plt.sca(axs[0, i])
            axs[0, i].loglog(kvals, Abins,
                          color=colors[m],
                          label=models[m],
                          alpha=0.5)
            plt.title(plot_varnames[i])
            plt.ylabel("$P(k)$")

            if model != "Truth":
                diff = np.abs(spectrum_all["Truth"][varname] - Abins)
                plt.sca(axs[1, i])
                axs[1, i].loglog(kvals, diff,
                              color=colors[m],
                              label=models[m],
                              alpha=0.5)
                plt.ylabel("$P(k)$")
                plt.xlabel("$k$")

    elif model == "Diffusion":
        # Plot spectrum
        for i, varname in enumerate(varnames):
            Abins_mean = spectrum_all["Diffusion_mean"][varname]
            diff = np.abs(spectrum_all["Truth"][varname] - Abins_mean)
            Abins_std = spectrum_all["Diffusion_std"][varname]

            plt.sca(axs[0, i])
            axs[0, i].loglog(kvals, Abins_mean,
                             color=colors[m],
                             label=models[m],
                             alpha=0.5)

            plt.sca(axs[1, i])
            axs[1, i].loglog(kvals, diff,
                          color=colors[m],
                          label=models[m],
                          alpha=0.5)
            #axs[1, i].fill_between(kvals,
            #                    diff - Abins_std,
            #                    diff + Abins_std,
            #                    color= colors[m], alpha=0.3)
# add labels
axs_flat = axs.flatten()
labels = ["a)", "b)", "c)",
          "d)", "e)", "f)"]
for i in range(len(axs_flat)):
    plt.text(x=-0.15, y=1.02, s=labels[i],
             fontsize=16, transform=axs_flat[i].transAxes)

# add legend at the bottom
leg_ax = axs[0, 1]
# Put a legend below current axis
leg_ax.legend(loc='lower center',
              bbox_to_anchor=(0.5, -1.7),
              ncol=4)

fig_filename = f"{plot_dir}/spectrum_incl_differences.png"
plt.tight_layout()
plt.savefig(fig_filename, bbox_inches="tight")
print(f"Saved as {fig_filename}")



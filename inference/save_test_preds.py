import sys
import xarray as xr
import torch
from torch.utils.data.dataloader import DataLoader

sys.path.append('../src')
from Inference import *

# Test years: 2018-2022
year_start = 2018
year_end = 2023

# Choose diffusion or U-Net to run
modelname = "UNet"    # "Diffusion" or "UNet" or "LinearInterpolation"

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Dirs
data_dir="/home/Everyone/ERA5/data/"
model_dir="/home/Everyone/Model_Inference/"
save_dir = f"../output/{modelname}/"

# Load model, sample function
if modelname == "Diffusion":
    model = EDMPrecond((256, 128), 8,
                            3).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/Model_chpt/diffusion.pt"))
    sample_function = sample_model_EDS
    num_steps = 100

    rngs = range(0, 30)
elif modelname == "UNet":
    sample_function = lambda test_batch, model, device, dataset_test, num_steps: sample_unet(test_batch, model, device, dataset_test)
    # Load model
    model = UNet((256, 128), 5, 3, label_dim=2, use_diffuse=False).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/Model_chpt/unet2.pt"))
    num_steps = None
    rngs = [""]

elif modelname == "LinearInterpolation":
    def sample_function(test_batch, model, device, dataset_test, num_steps=None):
        coarse, fine = test_batch["coarse"], test_batch["fine"]
        return coarse, fine, coarse
    model = None
    num_steps = None
    rngs = [""]
else:
    raise Exception(f"Choose modelname either Diffusion or UNet. You chose {modelname}")


print(f"Running model {modelname} with sample function {sample_function}.")

# Load dataset
dataset_test = UpscaleDataset(data_dir,
                              year_start=year_start,
                              year_end=year_end,
                              constant_variables=["lsm", "z"])

nlat = dataset_test.nlat
nlon = dataset_test.nlon
ntime = dataset_test.ntime
print(ntime, nlat, nlon)



for rng in rngs:
    if modelname == "Diffusion":
        np.random.seed(seed=rng)
    # Set up dataloader. Make sure shuffle=False so we go through timesteps in order.
    BATCH_SIZE = 32
    dataloader = DataLoader(dataset_test,
                            batch_size=BATCH_SIZE,
                            shuffle=False)


    # Set up xarray for saving: we will base this on the truth samples saved so all arrays have same
    # format, with same dimensions for time, lat and lon
    truth_filename = f"../output/Truth/samples_{year_start}-{year_end}.nc"
    truth_ds = xr.open_dataset(truth_filename, engine="netcdf4")
    print(truth_ds)

    # Create new arrays for saving predictions
    var_2T = xr.zeros_like(truth_ds.VAR_2T)
    var_10U = xr.zeros_like(truth_ds.VAR_10U)
    var_10V = xr.zeros_like(truth_ds.VAR_10V)

    t = 0   # time index
    for test_batch in dataloader:
        # Run model
        coarse, fine, predicted = sample_function(test_batch, model,
                                                  device, dataset_test,
                                                  num_steps=num_steps)
        all_pred_variables = predicted.detach().numpy()


        # This batch (may not be exactly = BATCH_SIZE for all batches)
        n_batch = all_pred_variables.shape[0]
        # Fill xarrays with predictions from t to t_end=t+n_batch
        t_end = t + n_batch
        var_2T[t:t_end] = all_pred_variables[:, 0]
        var_10U[t:t_end] = all_pred_variables[:, 1]
        var_10V[t:t_end] = all_pred_variables[:, 2]

        # Reset time index t for next iteration
        t = t_end


    # Create dataset
    ds_US = var_2T.to_dataset(name="VAR_2T")
    ds_US["VAR_10U"] = var_10U
    ds_US["VAR_10V"] = var_10V

    print(ds_US)


    # Save to netcdf4
    save_filename = f"{save_dir}/samples{rng}_{year_start}-{year_end}.nc"
    ds_US.to_netcdf(save_filename)
    print(f"Saved as {save_filename}")


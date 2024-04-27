import torch
from Network import UNet, EDMPrecond
import numpy as np
import matplotlib.pyplot as plt
from DatasetUS import UpscaleDataset


@torch.no_grad()
def sample_unet(input_batch, model, device, dataset):

    images_input = input_batch["inputs"].to(device)
    coarse, fine = input_batch["coarse"], input_batch["fine"]
    condition_params = torch.stack(
        (input_batch["doy"].to(device),
         input_batch["hour"].to(device)), dim=1)
    residual = model(images_input, class_labels=condition_params)
    predicted = dataset.residual_to_fine_image(residual.detach().cpu(), coarse)
    return coarse, fine, predicted


@torch.no_grad()
def sample_model_EDS(input_batch, model, device, dataset, num_steps=40,
                     sigma_min=0.002, sigma_max=80, rho=7, S_churn=40,
                     S_min=0, S_max=float('inf'), S_noise=1):

    images_input = input_batch["inputs"].to(device)
    coarse, fine = input_batch["coarse"], input_batch["fine"]
    condition_params = torch.stack(
        (input_batch["doy"].to(device),
         input_batch["hour"].to(device)), dim=1)

    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)

    init_noise = torch.randn((images_input.shape[0], 3, images_input.shape[2],
                              images_input.shape[3]),
                             dtype=torch.float64, device=device)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64,
                                device=init_noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
               * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([model.round_sigma(t_steps),
                         torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = init_noise.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = model.round_sigma(t_cur + gamma * t_cur)
        x_hat = (x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise *
                 torch.randn_like(x_cur))

        # Euler step.
        denoised = model(x_hat, t_hat, images_input, condition_params).to(
            torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = model(x_next, t_next, images_input,
                             condition_params).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    predicted = dataset.residual_to_fine_image(
        x_next.detach().cpu(), coarse)


    return coarse, fine, predicted


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diff_model = EDMPrecond((256, 128), 8,
                            3).to(device)
    diff_model.load_state_dict(torch.load("./Model_chpt/diffusion.pt"))

    unet_model = UNet((256, 128), 5, 3,
                      label_dim=2, use_diffuse=False).to(device)
    unet_model.load_state_dict(torch.load("./Model_chpt/unet.pt"))

    datadir = "/home/Everyone/ERA5/data/"
    dataset_test = UpscaleDataset(datadir, year_start=2017, year_end=2022,
                                  constant_variables=["lsm", "z"])

    # Try diffusion model
    coarse, fine, predicted = sample_model_EDS(dataset_test[0:4], diff_model,
                                               device, dataset_test)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].pcolormesh(coarse[0, 0])
    ax[0].set_title("Coarse")
    ax[1].pcolormesh(fine[0, 0])
    ax[1].set_title("Fine")
    ax[2].pcolormesh(predicted[0, 0])
    ax[2].set_title("Predicted")

    # Try unet model
    coarse, fine, predicted = sample_unet(dataset_test[0:4], unet_model,
                                             device, dataset_test)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].pcolormesh(coarse[0, 0])
    ax[0].set_title("Coarse")
    ax[1].pcolormesh(fine[0, 0])
    ax[1].set_title("Fine")
    ax[2].pcolormesh(predicted[0, 0])
    ax[2].set_title("Predicted")

    plt.show()
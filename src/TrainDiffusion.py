import torch
import Network
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from DatasetUS import UpscaleDataset
from torch.utils.tensorboard import SummaryWriter


# Loss class taken from EDS_Diffusion/loss.py
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, conditional_img=None, labels=None,
                 augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data)**2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, conditional_img, labels,
                   augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


def training_step(model, loss_fn, optimiser, data_loader, scaler, step,
                  accum=4, writer=None, device="cuda"):
    """
    Function for a single training step.
    :param model: Instance of the Unet class
    :param loss_fn: Loss function
    :param optimiser: Optimiser to use
    :param data_loader: Data loader
    :param scaler: Scaler for mixed precision training
    :param step: Current step
    :param accum: Number of steps to accumulate gradients over
    :param writer: Tensorboard writer
    :param device: Device to use
    :return: Loss value
    """

    model.train()
    with tqdm(total=len(data_loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {step}")

        epoch_losses = []
        step_loss = 0
        for i, batch in enumerate(data_loader):
            tq.update(1)

            image_input = batch["inputs"].to(device)
            image_output = batch["targets"].to(device)
            day = batch["doy"].to(device)
            hour = batch["hour"].to(device)
            condition_params = torch.stack((day, hour), dim=1)

            # forward unet
            with torch.cuda.amp.autocast():
                loss = loss_fn(net=model, images=image_output,
                               conditional_img=image_input,
                               labels=condition_params)
                loss = torch.mean(loss)

            # backpropagation
            scaler.scale(loss).backward()
            step_loss += loss.item()

            if (i + 1) % accum == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)

                if writer is not None:
                    writer.add_scalar("Loss/train", step_loss / accum,
                                      step * len(data_loader) + i)
                step_loss = 0

            epoch_losses.append(loss.item())
            tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        tq.set_postfix_str(s=f"Loss: {mean_loss:.4f}")
    return mean_loss


@torch.no_grad()
def sample_model(model, dataloader, num_steps=40, sigma_min=0.002,
                 sigma_max=80, rho=7, S_churn=40, S_min=0,
                 S_max=float('inf'), S_noise=1, device="cuda"):

    batch = next(iter(dataloader))
    images_input = batch["inputs"].to(device)
    coarse, fine = batch["coarse"], batch["fine"]

    condition_params = torch.stack(
        (batch["doy"].to(device),
         batch["hour"].to(device)), dim=1)

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

    predicted = dataloader.dataset.residual_to_fine_image(
        x_next.detach().cpu(), coarse)

    fig, ax = dataloader.dataset.plot_batch(coarse, fine, predicted)

    plt.subplots_adjust(wspace=0, hspace=0)
    base_error = torch.mean(torch.abs(fine - coarse))
    pred_error = torch.mean(torch.abs(fine - predicted))

    return (fig, ax), (base_error.item(), pred_error.item())


def main():
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 10000
    accum = 8

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = Network.EDMPrecond((256, 128), 8,
                                 3, label_dim=2)
    network.to(device)

    # define the datasets
    datadir = "/home/Everyone/ERA5/data/"
    dataset_train = UpscaleDataset(datadir, year_start=1950, year_end=2017,
                                   constant_variables=["lsm", "z"])

    dataset_test = UpscaleDataset(datadir, year_start=2017, year_end=2018,
                                  constant_variables=["lsm", "z"])

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)

    scaler = torch.cuda.amp.GradScaler()

    # define the optimiser
    optimiser = torch.optim.AdamW(network.parameters(), lr=learning_rate)

    # Define the tensorboard writer
    writer = SummaryWriter("./runs")

    # define loss function
    loss_fn = EDMLoss()

    # train the model
    losses = []
    for step in range(0, num_epochs):
        epoch_loss = training_step(network, loss_fn, optimiser,
                                   dataloader_train, scaler, step,
                                   accum, writer)
        losses.append(epoch_loss)

        if (step + 0) % 5 == 0:
            (fig, ax), (base_error, pred_error) = sample_model(
                network, dataloader_test)
            fig.savefig(f"./results/{step}.png", dpi=300)
            plt.close(fig)

            writer.add_scalar("Error/base", base_error, step)
            writer.add_scalar("Error/pred", pred_error, step)

        # save the model
        if losses[-1] == min(losses):
            torch.save(network.state_dict(), f"./Model/{step}.pt")


if __name__ == "__main__":
    main()
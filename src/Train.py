import torch
import torchvision
from Network import UNet
from Dataset import UpscaleDataset
from Diffusion import SimpleDiffusion, forward_diffusion
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.set_float32_matmul_precision('high')


def train_step(unet_model, diff_model, data_loader, optimiser, schedule,
               loss_fn, scaler, step, writer=None):
    """
    Function for a single training step.
    :param unet_model: instance of the Unet class
    :param diff_model: instance of the Diffusion class
    :param data_loader: data loader
    :param optimiser: optimiser to use
    :param schedule: learning rate schedule
    :param loss_fn: loss function to use
    :param scaler: scaler for mixed precision training
    :param step: current step
    :param writer: tensorboard writer
    :return: loss value
    """

    unet_model.train()

    with tqdm(total=len(data_loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {step}")

        epoch_losses = []
        for i, (image_low, image_high) in enumerate(data_loader):
            tq.update(1)

            # using bilinear interpolation to upscale the low resolution
            # to the high resolution
            image_low = torchvision.transforms.functional.resize(
                image_low, (image_high.shape[-2], image_high.shape[-1]),
                interpolation=2, antialias=True)

            image_low = image_low + data_loader.dataset.ds_mean

            # move to device
            image_low = image_low.to(diff_model.device)
            image_high = image_high.to(diff_model.device)

            # Remove the interpolated image from the high resolution image
            image_high = (image_high - image_low)

            # get time samples
            time = torch.randint(1, diff_model.num_diffusion_timesteps,
                                 (image_low.shape[0],),
                                 device=diff_model.device)

            # forward diffusion
            x, true_noise = forward_diffusion(diff_model, image_high, time)

            # concatenate the low resolution image with the noise along the
            # channel dimension
            x = torch.cat((image_low, x), dim=1)

            # forward unet
            with torch.cuda.amp.autocast():
                predict_noise = unet_model(x, time)
                loss = loss_fn(true_noise, predict_noise)

            # backpropagation
            scaler.scale(loss).backward()

            if (i + 1) % 4 == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
                schedule.step()

            epoch_losses.append(loss.item())
            tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")

            if writer is not None:
                writer.add_scalar("Loss/train", loss.item(),
                                  step * len(data_loader) + i)

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        tq.set_postfix_str(s=f"Loss: {mean_loss:.4f}")

    return mean_loss


@torch.no_grad()
def sample_model(unet_model, diff_model, dataloader):
    """
    Function for sampling the model.
    :param unet_model: instance of the Unet class
    :param diff_model: instance of the Diffusion class
    :param dataloader: data loader
    """

    unet_model.eval()

    # Get n_images from the dataloader
    samples = next(iter(dataloader))
    images_low = samples[0].to(diff_model.device)
    images_high = samples[1].to(diff_model.device)

    images = torch.randn((images_low.shape[0], *diff_model.img_shape),
                         device=diff_model.device)

    # upscale the low resolution images to the high resolution
    images_low_interp = torchvision.transforms.functional.resize(
        images_low, (images_high.shape[-2], images_high.shape[-1]),
        interpolation=2, antialias=True)

    images_low_interp = images_low_interp + dataloader.dataset.ds_mean.to(
        diff_model.device)

    # Loop over time steps
    for t in reversed(range(1, diff_model.num_diffusion_timesteps)):
        time = torch.ones(images_low.shape[0], dtype=torch.long,
                          device=diff_model.device) * t

        z = torch.randn_like(images) if t > 1 else torch.zeros_like(images)
        network_in = torch.cat((images_low_interp, images), dim=1)
        predict_noise = unet_model.forward(network_in, time)

        beta = diff_model.beta[t]
        alpha_1 = 1. / torch.sqrt(diff_model.alpha[t])
        alpha_2 = torch.sqrt(1 - diff_model.alpha_cumulative[t])

        images = (alpha_1 * (images - (beta / alpha_2) * predict_noise) +
                  beta**0.5 * z)

    images = images + images_low_interp

    # Plot images in a grid
    fig, ax = dataloader.dataset.plot_batch(images_low.detach().cpu().numpy(),
                                            images_high.detach().cpu().numpy(),
                                            images.detach().cpu().numpy(), N=3)
    plt.subplots_adjust(wspace=0, hspace=0)

    base_error = torch.mean(
        (images_high - images_low_interp - dataloader.dataset.ds_mean.to(
            diff_model.device))**2)**0.5
    pred_error = torch.mean((images_high - images)**2)**0.5

    return (fig, ax), (base_error.item(), pred_error.item())


def main():
    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # define the ml model
    unet_model = UNet(6, 3)
    #unet_model.load_state_dict(torch.load("./Models/1.pt"))
    unet_model.to(device)

    # define the diffusion model
    diff_model = SimpleDiffusion(device=device)

    # define the datasets
    datadir = "/home/Everyone/ERA5/data/"
    dataset_train = UpscaleDataset(datadir, year_start=1950, year_end=2021)
    dataset_test = UpscaleDataset(datadir, year_start=2021, year_end=2022,
                                  ds_mean=dataset_train.ds_mean)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=8,
                                                   shuffle=True, num_workers=4)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8,
                                                  shuffle=True, num_workers=4)

    scaler = torch.cuda.amp.GradScaler()

    # define the loss function
    loss_fn = torch.nn.MSELoss()

    # define the optimiser
    optimiser = torch.optim.AdamW(unet_model.parameters(), lr=3e-5)

    schedule = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.9999)

    # Define the tensorboard writer
    writer = SummaryWriter("./runs")

    unet_model = torch.compile(unet_model, mode="reduce-overhead")

    # train the model
    losses = []
    for step in range(10000):
        epoch_loss = train_step(
            unet_model, diff_model, dataloader_train, optimiser, schedule,
            loss_fn,
            scaler, step, writer)
        losses.append(epoch_loss)

        print(schedule.optimizer.param_groups[0]['lr'])

        if (step + 0) % 5 == 0:
            (fig, ax), (base_error, pred_error) = sample_model(
                unet_model, diff_model, dataloader_test)
            fig.savefig(f"./results/{step}.png", dpi=300)
            plt.close(fig)

            writer.add_scalar("Error/base", base_error, step)
            writer.add_scalar("Error/pred", pred_error, step)

        # save the model
        if losses[-1] == min(losses):
            torch.save(unet_model.state_dict(), f"./Models/{step}.pt")


if __name__ == "__main__":
    main()

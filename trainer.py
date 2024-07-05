import torch
import torch.nn as nn
from torch.optim import Adam
from model import MyDDPM, MyUNet
from tqdm import tqdm
import os
from utils import show_images, generate_new_images, show_forward


def train_model(loader, n_epochs, lr, checkpoint_dir, dataset_name=None, device=None, n_steps=1000):
    min_beta, max_beta = 10 ** -4, 0.02
    ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
    # 显示前向加噪过程
    # show_forward(ddpm, loader, device)
    optim = Adam(ddpm.parameters(), lr=lr)
    # 设置模型保存路径
    model_name = f"ddpm_{dataset_name}.pt"
    store_path = os.path.join(checkpoint_dir, model_name)
    training_loop(ddpm, loader, n_epochs, optim, device, store_path=store_path, display=True)
    return ddpm


def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path=None):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in range(n_epochs):
        inner_pbar = tqdm(total=len(loader), desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500", leave=False)
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            # Loading data
            x0 = batch[0].to(device)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

            inner_pbar.update(1)
            inner_pbar.set_postfix_str(f'Loss: {loss.item():.3f}')

        inner_pbar.close()

        # Display images generated at this epoch
        if display and (epoch + 1) % 10 == 0:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"\nLoss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        tqdm.write(log_string)
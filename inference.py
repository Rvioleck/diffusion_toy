import os
import torch
from model import MyDDPM, MyUNet
from utils import show_images, generate_new_images


def generate_and_show_inference_images(model_path, log_path, dataset_name, device, n_steps):
    """Loads a saved model and generates new images."""
    min_beta, max_beta = 10 ** -4, 0.02
    ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
    ddpm.load_state_dict(torch.load(model_path, map_location=device))
    gif_name = os.path.join(log_path, f"{dataset_name}_images.gif")
    generated = generate_new_images(ddpm, gif_name=gif_name, device=device)
    show_images(generated, "Images generated after training")

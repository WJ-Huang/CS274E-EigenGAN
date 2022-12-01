import os

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from config import *
from model import Generator

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NUM_SAMPLE = 64
    CKPT = 'D:/UCI/1-Q1/Deep Generative Model/Final Project/generator_step_50000.ckpt'
    LOGDIR = "./results"

    ckpt = torch.load(CKPT, map_location="cpu")

    g_ema = Generator(
        size=SIZE,
        num_basis=NUM_BASIS,
        noise_dim=NOISE_DIM,
        base_channels=BASE_CHANNELS,
        max_channels=MAX_CHANNELS,
    ).to(device).eval()
    g_ema.load_state_dict(ckpt)

    with torch.no_grad():
        save_image(
            g_ema.sample(NUM_SAMPLE), 
            "outputs.png", nrow=8, normalize=True, value_range=(-1, 1))
        traverse_range = 4.0
        intermediate_points = 9
        traverse_samples = 8
        es, zs = g_ema.sampleLatentVariables(traverse_samples)
        _, n_layers, n_dim = zs.shape # (8, 6, 6)

        offsets = np.linspace(-traverse_range, traverse_range, intermediate_points)
        for i_layer in range(n_layers):
            for i_dim in range(n_dim):
                print(f"layer {i_layer} - dim {i_dim}")
                imgs = []
                for offset in offsets:
                    _zs = zs.clone()
                    _zs[:, i_layer, i_dim] = offset
                    with torch.no_grad():
                        img = g_ema((es, _zs)).cpu()
                        img = torch.cat([_img for _img in img], dim=1)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=2)

                imgs = (imgs.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
                Image.fromarray(imgs).save(
                    os.path.join(LOGDIR, f"traverse_L{i_layer}_D{i_dim}.png")
                )
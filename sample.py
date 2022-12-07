import torch
import torchvision
import model
import numpy as np
import copy

from config import *

TRAVERSE = True

def deg_ortho(U, device):
    with torch.no_grad():
        M = copy.deepcopy(U)
        norms = torch.norm(U, dim=1)
        M = (M.T / norms).T
    return torch.sum((M.matmul(M.T) - torch.eye(M.shape[0], device=device))**2)

def gen_and_save_sample(generator, n_samples, nrow, path):
    with torch.no_grad():
        samples = generator.sample(n_samples)
    torchvision.utils.save_image(samples, path, nrow=nrow, normalize=True,value_range=(-1, 1))

def traverse_and_save_samples(generator, num_samples, range_len, samples_per_range):
    epsilon_samples, z_samples = generator.sampleLatentVariables(num_samples)
    _, n_layers, n_dim = z_samples.shape

    js = np.linspace(-range_len, range_len, samples_per_range)
    for layer in range(n_layers):
        U = generator.blocks[layer].subspacelayer.U
        print(f"||MMT-I||_F^2={deg_ortho(U, generator.getDevice())*1000}")
        for dim in range(n_dim):
            print(f"Layer {layer}, dimension {dim}, Lij={generator.blocks[layer].subspacelayer.L[dim]}")
            imgs = []
            for j in js:
                zs = copy.deepcopy(z_samples)
                zs[:, layer, dim] = j
                with torch.no_grad():
                    img = generator((epsilon_samples, zs)).cpu()
                    imgs.append(img)
            imgs = torch.stack(imgs, dim=0).swapaxes(0, 1)
            imgs = imgs.reshape(num_samples * samples_per_range, 3, SIZE, SIZE)
            torchvision.utils.save_image(imgs, f"samples/traverse_L{layer}_D{dim}.jpg", normalize=True, value_range=(-1, 1), nrow=samples_per_range)

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    generator = model.Generator(
        size=SIZE,
        num_basis=NUM_BASIS,
        noise_dim=NOISE_DIM,
        base_channels=BASE_CHANNELS,
        max_channels=MAX_CHANNELS
    ).to(device)

    generator.load_state_dict(torch.load("model_checkpoints/g_ema_step_1000000.ckpt"))

    if not TRAVERSE:
        gen_and_save_sample(generator, 16, 4, "samples/sample.jpg")
    else:
        traverse_and_save_samples(generator, 8, 4.5, 9)

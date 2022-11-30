import torch
import torchvision
import model

from config import *

def gen_and_save_sample(generator, n_samples, nrow, path):
    with torch.no_grad():
        samples = generator.sample(n_samples)
    torchvision.utils.save_image(samples, path, nrow=nrow, normalize=True,value_range=(-1, 1))

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

    generator.load_state_dict(torch.load("model_checkpoints/generator_step_15000.ckpt"))
    gen_and_save_sample(generator, 16, 4, "samples/sample.jpg")
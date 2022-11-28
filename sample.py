import torch
import torchvision
import train
import model
import matplotlib.pyplot as plt

def gen_and_save_sample(generator, n_samples, n_row, path):
    with torch.no_grad():
        samples = generator.sample(n_samples)
    torchvision.utils.save_image(samples, path, n_row)

if __name__ == "__main__":
    SIZE = train.SIZE
    NUM_BASIS = train.NUM_BASIS
    NOISE_DIM = train.NOISE_DIM
    BASE_CHANNELS = train.BASE_CHANNELS
    MAX_CHANNELS = train.MAX_CHANNELS

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    generator = model.Generator(
        size=SIZE,
        num_basis=NUM_BASIS,
        noise_dim=NOISE_DIM,
        base_channels=BASE_CHANNELS,
        max_channels=MAX_CHANNELS
    ).to(device)

    generator.load_state_dict(torch.load("model_checkpoints/generator_step_7000.pth"))
    gen_and_save_sample(generator, 16, 4, "samples/sample.jpg")
import torch
import model
import loss
import copy
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import Dataset, infinite_loader
from sample import gen_and_save_sample
from config import *


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
    

    discriminator = model.Discriminator(
        size=SIZE, 
        base_channels=BASE_CHANNELS,
        max_channels=MAX_CHANNELS,
    ).to(device)
    
    if START_FROM > 0:
        generator.load_state_dict(torch.load(f"model_checkpoints/generator_step_{START_FROM}.ckpt"))
        discriminator.load_state_dict(torch.load(f"model_checkpoints/discriminator_step_{START_FROM}.ckpt"))
        g_ema.load_state_dict(torch.load(f"model_checkpoints/g_ema_step_{START_FROM}.ckpt"))

    # optimizers
    g_optim = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.99),
    )
    d_optim = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.99),
    )

    # losses
    training_loss = open(TRAINING_LOSS_FILE, "a")

    # data
    transform = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.CenterCrop(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = Dataset(DATASET_PATH, transform)
    loader = infinite_loader(
        DataLoader(
            dataset,
            batch_size=BATCH,
            shuffle=True,
            drop_last=True,
            num_workers=NUM_WORKERS
        )
    )

    for step in range(START_FROM+1, STEPS):
        real = next(loader).to(device)
        
        with torch.no_grad():
            fake = generator.sample(BATCH)

        real_pred = discriminator(real)
        fake_pred = discriminator(fake)

        # adverserial loss
        discriminator_loss = loss.discriminator_hinge_loss(real_pred, fake_pred)
        discriminator.zero_grad()
        discriminator_loss.backward()
        d_optim.step()

        # R1 penalty, lazy penalty
        if step % REG_EVERY == 0:
            real.requires_grad = True
            real_pred = discriminator(real)
            r1 = loss.r1_loss(real_pred, real) * R1_PENALTY_COEFFICIENT
            discriminator.zero_grad()
            r1.backward()
            d_optim.step()

        fake = generator.sample(BATCH)
        fake_pred = discriminator(fake)
        generator_loss = loss.generator_hinge_loss(fake_pred) + generator.regularize() * ORTHOGONAL_REGULERIZATION_COEFFICIENT
        generator.zero_grad()
        generator_loss.backward()
        g_optim.step()

        print(f'Step: {step}/{STEPS}, D_loss: {discriminator_loss.item()}, G_loss: {generator_loss.item()}')
        training_loss.write(f"{discriminator_loss.item()}, {generator_loss.item()}\n")

        if step > 0 and step % SAVE_EVERY == 0:
            torch.save(discriminator.state_dict(), f"model_checkpoints/discriminator_step_{step}.ckpt")
            torch.save(generator.state_dict(), f"model_checkpoints/generator_step_{step}.ckpt")
        
        if step > 0 and step % SAMPLE_EVERY == 0:
            gen_and_save_sample(generator, 16, 4, f"samples/samples_{step}.jpg")

    training_loss.close()
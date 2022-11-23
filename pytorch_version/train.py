import torch
import model
import loss
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import Dataset, infinite_loader

DATASET_PATH = "../data/test"
SIZE = 64
BATCH = 16
LEARNING_RATE = 1e-3
STEPS = 10
NUM_BASIS = 6
NOISE_DIM = 512
BASE_CHANNELS = 16
MAX_CHANNELS = 512
R1_PENALTY_COEFFICIENT = 10
ORTHOGONAL_REGULERIZATION_COEFFICIENT = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
calculateDLoss, calculateGLoss = loss.calculateHingeLoss()

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
        drop_last=True
    )
)

for step in range(STEPS):
    print(step)
    real = next(loader).to(device)
    with torch.no_grad():
        fake = generator.sample(BATCH)

    real_pred = discriminator(real)
    fake_pred = discriminator(fake)

    # adverserial loss
    discriminator_loss = calculateDLoss(real_pred, fake_pred)
    discriminator.zero_grad()
    discriminator_loss.backward()
    d_optim.step()

    print("r1")
    # R1 penalty
    real.requires_grad = True
    real_pred = discriminator(real)
    r1 = loss.calculateR1Loss(real_pred, real) * R1_PENALTY_COEFFICIENT
    discriminator.zero_grad()
    r1.backward()
    d_optim.step()

    fake = generator.sample(BATCH)
    fake_pred = discriminator(fake)
    generator_loss = calculateGLoss(fake_pred) + generator.regularizeUOrthogonal() * ORTHOGONAL_REGULERIZATION_COEFFICIENT
    generator.zero_grad()
    generator_loss.backward()
    g_optim.step()

    print(f'Step: \t {step}, D_loss:\t{discriminator_loss.item()}, G_loss:\t{generator_loss.item()}')
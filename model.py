import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SubspaceModel(nn.Module):
    def __init__(self, 
        dim: int,           # d
        num_basis: int      # q
    ) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.empty((num_basis, dim)))    # size(d, q)
        nn.init.orthogonal_(self.U)
        self.L = nn.Parameter(torch.FloatTensor([3 * i for i in range(num_basis, 0, -1)])) # q
        self.mu = nn.Parameter(torch.zeros(dim)) # size(d, 1)

    def forward(self, z):
        return torch.mm(self.L * z, self.U) + self.mu

class EigenBlock(nn.Module):
    def __init__(self, 
        width: int, 
        height: int, 
        in_channels: int, 
        out_channels: int, 
        num_basis: int
    ) -> None:
        super().__init__()
        self.subspacelayer = SubspaceModel(
            dim=width * height * in_channels, 
            num_basis=num_basis
        )

        self.subspace_conv1 = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            output_padding=0
        )

        self.subspace_conv2 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        self.feature_conv1 = nn.ConvTranspose2d( # output size 2H
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        self.feature_conv2 = nn.ConvTranspose2d( # output size H
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=0
        )
    
    def forward(self, z, h):
        phi = self.subspacelayer(z).reshape(h.shape)
        out = self.feature_conv1(F.leaky_relu(self.subspace_conv1(phi) + h, 0.2, inplace=True))
        out = self.feature_conv2(F.leaky_relu(self.subspace_conv2(phi) + out, 0.2, inplace=True))
        return out

class Generator(nn.Module):
    def __init__(self,
        size: int = 64,
        num_basis: int = 6,
        noise_dim: int = 64, # generator input (\epsilon) dim
        base_channels: int = 16,
        max_channels: int = 64
    ) -> None:
        super().__init__()

        self.noise_dim = noise_dim
        self.num_basis = num_basis
        self.num_blocks = int(math.log(size, 2) - 2) # size = 4 * (2**num_blocks)

        def getChannelsNumber(block_idx: int):
            return min(max_channels, base_channels * (2**(self.num_blocks - block_idx)))
        
        self.fc_layer = nn.Linear(self.noise_dim, 4 * 4 * getChannelsNumber(0))

        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.blocks.append(
                EigenBlock(
                    width=4 * 2**i,
                    height=4 * 2**i,
                    in_channels=getChannelsNumber(i),
                    out_channels=getChannelsNumber(i + 1),
                    num_basis=self.num_basis
                )
            )
        
        self.output_layer = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                base_channels,
                3,
                kernel_size=7,
                stride=1,
                padding=3
            ),
            nn.Tanh()
        )

    def getDevice(self):
        return self.fc_layer.weight.device

    def sampleLatentVariables(self, batch: int):
        device = self.getDevice()
        epsilon_samples = torch.randn(batch, self.noise_dim, device= device)
        z_samples = torch.randn(batch, self.num_blocks, self.num_basis, device=device) # sample z of each block together
        return epsilon_samples, z_samples
    
    def forward(self, latent_variables):
        epsilon_samples, z_samples = latent_variables

        output = self.fc_layer(epsilon_samples).reshape(len(epsilon_samples), -1, 4, 4)
        for block, z_sample in zip(self.blocks, torch.permute(z_samples, (1, 0, 2))):
            output = block(z_sample, output)
        
        return self.output_layer(output)
    
    def sample(self, batch: int):
        return self.forward(self.sampleLatentVariables(batch))

    def regularize(self):
        reg = []
        for layer in self.modules():
            if isinstance(layer, SubspaceModel):
                reg.append(torch.mean((layer.U.matmul(layer.U.T) - torch.eye(layer.U.shape[0], device=self.getDevice()))**2))
        return sum(reg) / len(reg)

class Discriminator(nn.Module):
    def __init__(self,
        size: int = 64,
        base_channels: int = 16,
        max_channels: int = 64    
    ) -> None:
        super().__init__()

        blocks = [
            nn.Conv2d(
                3,
                base_channels,
                kernel_size=7,
                stride=1,
                padding=3
            ),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        num_channels = base_channels
        for _ in range(int(math.log(size, 2) - 2)):
            next_num_channels = min(max_channels, num_channels * 2)
            blocks += [
                nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_channels, next_num_channels, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            num_channels = next_num_channels
        blocks.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.LeakyReLU(0.2, inplace=True))

        self.blocks = nn.Sequential(*blocks)
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * num_channels, num_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_channels, 1)
        )

    def forward(self, input):
        output = self.blocks(input)
        return self.output_layer(output)
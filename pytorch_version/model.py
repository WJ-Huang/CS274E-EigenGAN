# %%
import numpy as np
import math
import torch
import torch.nn as nn
from functools import partial

# %%
class SubspaceModel(nn.Module):
    def __init__(self, 
        dim: int,           # d
        num_basis: int      # q
    ) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.empty((dim, num_basis)))    # size(d, q)
        nn.init.orthogonal_(self.U)
        self.L = nn.Parameter(torch.FloatTensor([3 * i for i in range(num_basis, 0, -1)])) #
        self.mu = nn.Parameter(torch.zeros(dim)) # size(d, 1)

    def forward(self, z):
        return self.U.mm(self.L * z) + self.mu

class ConvLayer(nn.Module):
    def __init__(self,
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1, 
        padding_mode: str = "zeros", 
        # groups: int = 1, 
        # # bias: bool = True, 
        transposed: bool = False, 
        # normalization: str = None, 
        activation: bool = True, 
        pre_activate: bool = False
    ) -> None:
        if transposed:
            conv = partial(nn.ConvTranspose2d, output_padding=stride - 1)
            padding_mode = "zeros"
        else:
            conv = nn.Conv2d
        
        layers = [conv(
                    in_channels, 
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    padding_mode=padding_mode
                    )]
        if activation:
            if pre_activate:
                layers.insert(0, nn.LeakyReLU())
            else:
                layers.append(nn.LeakyReLU())
        super().__init__(*layers)

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
        self.subspace_conv1 = ConvLayer( # output size H
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            transposed=True,
            activation=False
        )
        self.subspace_conv2 = ConvLayer( # output size 2H
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            transposed=True,
            activation=False
        )
        self.feature_conv1 = ConvLayer( # output size 2H
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            transposed=True,
            pre_activate=True
        )
        self.feature_conv2 = ConvLayer( # output size H
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            transposed=True,
            pre_activate=True
        )
    
    def forward(self, z, h):
        phi = self.subspacelayer(z).view(h.shape)
        h = self.feature_conv1(self.subspace_conv1(phi) + h)
        h = self.feature_conv2(self.subspace_conv2(phi) + h)
        return h

# %%
class Generator(nn.Module):
    def __init__(self,
        size: int = 256,
        num_basis: int = 6,
        noise_dim: int = 512, # generator input (\epsilon) dim
        base_channels: int = 16,
        max_channels: int = 512
    ) -> None:
        super().__init__()

        assert (size & (size-1) == 0) and size != 0, "image size should be a power of 2."

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
            ConvLayer(
                base_channels,
                3,
                kernel_size=7,
                stride=1,
                padding=3, # keep the size same
                pre_activate=True
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

        output = self.fc_layer(epsilon_samples).view(len(epsilon_samples), -1, 4, 4)
        for block, z_sample in zip(self.blocks, torch.permute(z_samples, (1, 0, 2))):
            output = block(z_sample, output)
        
        return self.output_layer(output)

    def regularizeUOrthogonal(self):
        reg = []
        for layer in self.modules():
            if isinstance(layer, SubspaceModel):
                UUT = layer.U.matmul(layer.U.transpose())
                reg.append(
                    torch.mean((UUT - torch.eye(UUT.shape([0]), device=UUT.device))**2)
                    )
        return sum(reg) / len(reg)


class Discriminator(nn.Module):
    def __init__(self,
        size: int = 256,
        base_channels: int = 16,
        max_channels: int = 512    
    ) -> None:
        super().__init__()

        blocks = [
            ConvLayer(
                3,
                base_channels,
                kernel_size=7,
                stride=1,
                padding=3 # keep the same size
            )
        ]

        num_channels = base_channels
        for _ in range(int(math.log(size, 2) - 2)):
            next_num_channels = min(max_channels, num_channels * 2)
            blocks += [
                ConvLayer(num_channels, num_channels, kernel_size=3, stride=1), # same size
                ConvLayer(num_channels, next_num_channels, kernel_size=3, stride=2) # downsample to H/2
                ]
            num_channels = next_num_channels
        blocks.append(ConvLayer(num_channels, num_channels, kernel_size=3, stride=1))

        self.blocks = nn.Sequential(*blocks)
        self.output_layer = nn.Sequential(
            nn.Flatten(), #
            nn.Linear(4 * 4 * num_channels, num_channels),
            nn.LeakyReLU(),
            nn.Linear(num_channels, 1)
        )

    def forward(self, input):
        output = self.blocks(input)
        return self.output_layer(output)





import torch
from torch import nn, optim


def init_weights(model):
    classname = model.__class__.__name__
    if 'Conv' in classname:
        model.weight.data.normal_(0.0, 0.02)
    elif 'BatchNorm' in classname:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


class DCGANGenerator(nn.Module):
    def __init__(self, z_size, num_features, num_channels):
        super(DCGANGenerator, self).__init__()
        self.stack = nn.Sequential(
            # Z -> (F*8)x4x4
            nn.ConvTranspose2d(z_size, num_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(True),
            # (F*8)x4x4 -> (F*4)x8x8
            nn.ConvTranspose2d(num_features * 8, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(True),
            # (F*4)x4x4 -> (F*2)x16x16
            nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(True),
            # (F*2)x4x4 -> Fx32x32
            nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),
            # Fx32x32 -> Cx64x64
            nn.ConvTranspose2d(num_features, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            )

    def forward(self, input):
        return self.stack(input)

class DCGANDiscriminator(nn.Module):
    def __init__(self, num_features, num_channels, leak):
        super(DCGANDiscriminator, self).__init__()
        self.stack = nn.Sequential(
            # Cx64x64 -> Fx32x32
            nn.Conv2d(num_channels, num_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(leak, inplace=True),
            # Fx32x32 -> (F*2)x16x16
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(leak, inplace=True),
            # (F*2)x16x16 -> (F*4)x8x8
            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(leak, inplace=True),
            # (F*4)x8x8 -> (F*8)x4x4
            nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(leak, inplace=True),
            # (F*8)x4x4 -> 1
            nn.Conv2d(num_features * 8, 1, 4, 1, 0, bias=False),
            )
        
    def forward(self, input):
        output = self.stack(input)
        return output.view(-1, 1).squeeze(1)


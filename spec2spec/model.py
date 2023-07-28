import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(
        self, depth=17, n_channels=64, image_channels=1, kernel_size=3, padding=1
    ):
        super(DnCNN, self).__init__()

        layers = []

        # Head
        layers.append(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            )
        )
        layers.append(nn.ReLU(inplace=True))

        # Body
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=True,
                )
            )
            layers.append(
                nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.9, affine=True)
            )
            layers.append(nn.ReLU(inplace=True))

        # Tail
        layers.append(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=image_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            )
        )

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)  # x is the noisy image, out is the noise
        return x - out  # predicted clean image

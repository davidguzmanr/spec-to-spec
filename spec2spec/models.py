import torch
import torch.nn as nn
from torch.nn import functional as F


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


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

    def _forward(self, x):
        # x has shape (batch_size, 1, 80, mel_length)
        x = x.contiguous().squeeze()

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().unsqueeze(1)
        return x

    def forward(self, x):
        # This is the posnet output, which is the "clean" image. I think it shouldn't make difference to
        # add/substract as long as I keep track of the noisy and clean images
        return x + self._forward(x)
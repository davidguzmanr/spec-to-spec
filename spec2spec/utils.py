import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

sns.set_style("darkgrid")

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = ResBlock

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h.upsample_initial_channel // (2**i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        # print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


UNIVERSAL_CONFIG = {
    "resblock": "1",
    "num_gpus": 0,
    "batch_size": 16,
    "learning_rate": 0.0002,
    "adam_b1": 0.8,
    "adam_b2": 0.99,
    "lr_decay": 0.999,
    "seed": 1234,
    "upsample_rates": [8, 8, 2, 2],
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "segment_size": 8192,
    "num_mels": 80,
    "num_freq": 1025,
    "n_fft": 1024,
    "hop_size": 256,
    "win_size": 1024,
    "sampling_rate": 22050,
    "fmin": 0,
    "fmax": 8000,
    "fmax_for_loss": None,
    "num_workers": 4,
    "dist_config": {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "world_size": 1,
    },
}


def get_vocoder(path, device):
    vocoder = Generator(AttrDict(UNIVERSAL_CONFIG))
    ckpt = torch.load(path, map_location=device)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, max_wav_value=32768.0, lengths=None):
    # mels (1, 80, 111) normal
    # mels small (1, 80, 5)
    with torch.no_grad():
        wavs = vocoder(mels.transpose(1, 2)).squeeze(1)
    wavs = wavs.cpu().numpy()
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs


def plot_pairs(
    noisy_batch: torch.Tensor,
    ground_truth_batch: torch.Tensor,
    prediction_batch: torch.Tensor,
) -> plt.Figure:
    """
    Plot pairs of spectrograms and their corresponding residuals.

    Args:
        noisy_batch (torch.Tensor): Batch of noisy spectrograms. Shape: (batch_size, 1, 80, width).
        ground_truth_batch (torch.Tensor): Batch of ground truth (clean) spectrograms. Shape: (batch_size, 1, 80, width).
        prediction_batch (torch.Tensor): Batch of denoised (predicted) spectrograms. Shape: (batch_size, 1, 80, width).

    Returns:
        plt.Figure: The matplotlib Figure containing the plotted spectrograms and residuals.
    """

    # Grab the one that doesn't have zero padding (extra zeros mess up the distribution)
    index = [
        i
        for (i, noisy) in enumerate(noisy_batch)
        if noisy.squeeze().detach().cpu().numpy()[:, -1].sum() != 0
    ][0]

    noisy = noisy_batch[index].squeeze().detach().cpu().numpy()
    ground_truth = ground_truth_batch[index].squeeze().detach().cpu().numpy()
    prediction = prediction_batch[index].squeeze().detach().cpu().numpy()
    residual_before = (ground_truth - noisy).reshape(-1)
    residual_after = (ground_truth - prediction).reshape(-1)

    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        ncols=1, nrows=6, figsize=(10, 10)
    )

    pcm = ax0.imshow(noisy, cmap="viridis")
    ax0.grid(False)
    fig.colorbar(pcm, ax=ax0)
    ax0.set_title("Noisy spectrogram", fontsize=10, weight="bold")

    pcm = ax1.imshow(ground_truth, cmap="viridis")
    ax1.grid(False)
    fig.colorbar(pcm, ax=ax1)
    ax1.set_title("Ground truth spectrogram", fontsize=10, weight="bold")

    pcm = ax2.imshow(prediction, cmap="viridis")
    ax2.grid(False)
    fig.colorbar(pcm, ax=ax2)
    ax2.set_title("Denoised (predicted) spectrogram", fontsize=10, weight="bold")

    pcm = ax3.imshow(residual_before.reshape(80, -1), cmap="viridis")
    ax3.grid(False)
    fig.colorbar(pcm, ax=ax3)
    ax3.set_title(f"Residual before", fontsize=10, weight="bold")

    pcm = ax4.imshow(residual_after.reshape(80, -1), cmap="viridis")
    ax4.grid(False)
    fig.colorbar(pcm, ax=ax4)
    ax4.set_title(f"Residual after", fontsize=10, weight="bold")

    data = pd.DataFrame(
        data={
            "x": np.concatenate((residual_before, residual_after)),
            "Label": ["Before"] * len(residual_before)
            + ["After"] * len(residual_after),
        }
    )
    sns.histplot(
        data=data, x="x", hue="Label", kde=True, stat="density", ax=ax5, bins=100
    )
    ax5.set_title(f"Residual distribution", fontsize=10, weight="bold")
    ax5.set_xlim(-4, 4)

    return fig


def get_audios(
    vocoder_path: str,
    noisy_batch: torch.Tensor,
    ground_truth_batch: torch.Tensor,
    prediction_batch: torch.Tensor,
):
    vocoder = get_vocoder(
        vocoder_path,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    # Grab the one that doesn't have zero padding (extra zeros mess up the distribution)
    index = [
        i
        for (i, noisy) in enumerate(noisy_batch)
        if noisy.squeeze().detach().cpu().numpy()[:, -1].sum() != 0
    ][0]

    noisy = noisy_batch[index].detach().transpose(2, 1)
    ground_truth = ground_truth_batch[index].detach().transpose(2, 1)
    prediction = prediction_batch[index].detach().transpose(2, 1)

    noisy_wav = vocoder_infer(noisy, vocoder)[0]
    ground_truth_wav = vocoder_infer(ground_truth, vocoder)[0]
    prediction_wav = vocoder_infer(prediction, vocoder)[0]

    return (
        torch.tensor(noisy_wav),
        torch.tensor(ground_truth_wav),
        torch.tensor(prediction_wav),
    )

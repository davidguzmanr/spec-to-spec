from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from dataset import DenoisingDataModule
from models import DnCNN, PostNet, DnCNNConfig, PostNetConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LightningCLI
from torch.nn import functional as F
from utils import get_audios, plot_pairs


class SpecDenoiser(LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        loss: str = 'mse',
        vocoder_path: Optional[str] = None,
        network: str = 'DnCNN',
        network_kwargs: Optional[Dict] = None,
    ) -> None:
        super(SpecDenoiser, self).__init__()
        self.save_hyperparameters()

        network_kwargs = (
            {} if self.hparams.network_kwargs is None else self.hparams.network_kwargs
        )

        if self.hparams.network == 'DnCNN':
            config = DnCNNConfig(**network_kwargs)
            self.model = DnCNN(config)
        elif self.hparams.network == 'PostNet':
            config = PostNetConfig(**network_kwargs)
            self.model = PostNet(config)

        if self.hparams.loss == 'mse':
            self.loss = nn.MSELoss(reduction='mean')
        elif self.hparams.loss == 'l1':
            self.loss = nn.L1Loss(reduction='mean')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        clean, noisy = batch
        out = self.forward(noisy)  # predicted clean
        loss = self.loss(out, clean)

        self.log("train/loss", loss)

        if (batch_idx == 0) or (batch_idx == self.trainer.num_training_batches - 1):
            self.logger.experiment.add_figure(
                f"train/batch-{batch_idx}",
                plot_pairs(noisy, clean, out),
                self.global_step,
            )

            if self.hparams.vocoder_path:
                noisy_wav, clean_wav, out_wav = get_audios(
                    self.hparams.vocoder_path, noisy, clean, out
                )

                self.logger.experiment.add_audio(
                    f"train/noisy-{batch_idx}",
                    noisy_wav,
                    self.global_step,
                    sample_rate=22050,
                )

                self.logger.experiment.add_audio(
                    f"train/clean-{batch_idx}",
                    clean_wav,
                    self.global_step,
                    sample_rate=22050,
                )

                self.logger.experiment.add_audio(
                    f"train/denoised-{batch_idx}",
                    out_wav,
                    self.global_step,
                    sample_rate=22050,
                )

        return loss

    def validation_step(self, batch, batch_idx):
        clean, noisy = batch
        out = self.forward(noisy)  # predicted clean
        loss = self.loss(out, clean)

        self.log("val/loss", loss)

        if (batch_idx == 0) or (batch_idx == self.trainer.num_val_batches[0] - 1):
            self.logger.experiment.add_figure(
                f"val/batch-{batch_idx}",
                plot_pairs(noisy, clean, out),
                self.global_step,
            )

            if self.hparams.vocoder_path:
                noisy_wav, clean_wav, out_wav = get_audios(
                    self.hparams.vocoder_path, noisy, clean, out
                )

                self.logger.experiment.add_audio(
                    f"val/noisy-{batch_idx}",
                    noisy_wav,
                    self.global_step,
                    sample_rate=22050,
                )

                self.logger.experiment.add_audio(
                    f"val/clean-{batch_idx}",
                    clean_wav,
                    self.global_step,
                    sample_rate=22050,
                )

                self.logger.experiment.add_audio(
                    f"val/denoised-{batch_idx}",
                    out_wav,
                    self.global_step,
                    sample_rate=22050,
                )

        return loss

    def test_step(self, batch, batch_idx):
        clean, noisy = batch
        out = self.forward(noisy)  # predicted clean
        loss = self.loss(out, clean)

        self.log("test/loss", loss)

        if (batch_idx == 0) or (batch_idx == self.trainer.num_test_batches[0] - 1):
            self.logger.experiment.add_figure(
                f"test/batch-{batch_idx}",
                plot_pairs(noisy, clean, out),
                self.global_step,
            )

            if self.hparams.vocoder_path:
                noisy_wav, clean_wav, out_wav = get_audios(
                    self.hparams.vocoder_path, noisy, clean, out
                )

                self.logger.experiment.add_audio(
                    f"test/noisy-{batch_idx}",
                    noisy_wav,
                    self.global_step,
                    sample_rate=22050,
                )

                self.logger.experiment.add_audio(
                    f"test/clean-{batch_idx}",
                    clean_wav,
                    self.global_step,
                    sample_rate=22050,
                )

                self.logger.experiment.add_audio(
                    f"test/denoised-{batch_idx}",
                    out_wav,
                    self.global_step,
                    sample_rate=22050,
                )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        # return [optimizer], [scheduler]

        return optimizer


def cli_main():
    cli = LightningCLI(
        SpecDenoiser, DenoisingDataModule, seed_everything_default=42, run=False
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main()

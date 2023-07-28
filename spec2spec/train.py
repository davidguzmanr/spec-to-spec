import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy

from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LightningCLI

from dataset import DenoisingDataset, DenoisingDataModule
from model import DnCNN

from typing import Any, Optional

from utils import plot_pairs, get_vocoder, get_audios


class SpecDenoiser(LightningModule):
    def __init__(self, lr: float = 1e-4, vocoder_path: Optional[str] = None) -> None:
        super(SpecDenoiser, self).__init__()
        self.save_hyperparameters()

        self.model = DnCNN()

        if vocoder_path:
            self.vocoder = get_vocoder(
                vocoder_path,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        clean, noisy = batch
        out = self.forward(noisy)  # predicted clean
        loss = F.mse_loss(out, clean, reduction="sum")

        self.log("train/loss", loss, on_step=True, on_epoch=False)

        if batch_idx == 0:
            self.logger.experiment.add_figure(
                f"train/specs", plot_pairs(noisy, clean, out), self.global_step
            )
            if self.hparams.vocoder_path:
                noisy_wav, clean_wav, out_wav = get_audios(
                    self.vocoder, noisy, clean, out
                )

                self.logger.experiment.add_audio(
                    "train/noisy", noisy_wav, self.global_step, sample_rate=22050
                )

                self.logger.experiment.add_audio(
                    "train/clean", clean_wav, self.global_step, sample_rate=22050
                )

                self.logger.experiment.add_audio(
                    "train/denoised", out_wav, self.global_step, sample_rate=22050
                )

        return loss

    def validation_step(self, batch, batch_idx):
        clean, noisy = batch
        out = self.forward(noisy)  # predicted clean
        loss = F.mse_loss(out, clean, reduction="sum")

        self.log("val_loss", loss, on_step=True, on_epoch=False)

        if batch_idx == 0:
            self.logger.experiment.add_figure(
                f"val/specs", plot_pairs(noisy, clean, out), self.global_step
            )
            if self.hparams.vocoder_path:
                noisy_wav, clean_wav, out_wav = get_audios(
                    self.vocoder, noisy, clean, out
                )

                self.logger.experiment.add_audio(
                    "val/noisy", noisy_wav, self.global_step, sample_rate=22050
                )

                self.logger.experiment.add_audio(
                    "val/clean", clean_wav, self.global_step, sample_rate=22050
                )

                self.logger.experiment.add_audio(
                    "val/denoised", out_wav, self.global_step, sample_rate=22050
                )

        return loss

    def test_step(self, batch, batch_idx):
        clean, noisy = batch
        out = self.forward(noisy)  # predicted clean
        loss = F.mse_loss(out, clean, reduction="sum")

        self.log("test_loss", loss, on_step=True, on_epoch=False)

        if batch_idx == 0:
            self.logger.experiment.add_figure(
                f"test/specs", plot_pairs(noisy, clean, out), self.global_step
            )
            # TODO: add audios from the predictions
            if self.hparams.vocoder_path:
                noisy_wav, clean_wav, out_wav = get_audios(
                    self.vocoder, noisy, clean, out
                )

                self.logger.experiment.add_audio(
                    "test/noisy", noisy_wav, self.global_step, sample_rate=22050
                )

                self.logger.experiment.add_audio(
                    "test/clean", clean_wav, self.global_step, sample_rate=22050
                )

                self.logger.experiment.add_audio(
                    "test/denoised", out_wav, self.global_step, sample_rate=22050
                )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
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

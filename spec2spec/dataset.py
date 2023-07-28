import torch
from torch.utils.data import Dataset, DataLoader, random_split

from pytorch_lightning import LightningDataModule

from typing import Union, Tuple
from pathlib import Path
from os import listdir


class DenoisingDataset(Dataset):
    def __init__(self, data_dir: Union[Path, str]) -> None:
        super(DenoisingDataset, self).__init__()

        self.data_dir = Path(data_dir)
        self.clean_path = self.data_dir / "clean"
        self.noisy_path = self.data_dir / "noisy"

        self.clean_files = sorted(
            [self.clean_path / file for file in listdir(self.clean_path)]
        )
        self.noisy_files = sorted(
            [self.noisy_path / file for file in listdir(self.noisy_path)]
        )

    def __len__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return len(self.clean_files)

    def __getitem__(self, index):
        clean = torch.load(self.clean_files[index]).unsqueeze(0)
        noisy = torch.load(self.noisy_files[index]).unsqueeze(0)

        # TODO: when there is a mismatch in the shapes, should we trim or pad?

        return clean, noisy

    @staticmethod
    def custom_collate(batch):
        # Find the maximum length in the batch
        max_length = max(clean.shape[-1] for clean, _ in batch)

        # Pad each tensor in the batch to the maximum length
        padded_clean = []
        padded_noisy = []
        for clean, noisy in batch:
            padded_clean.append(
                torch.nn.functional.pad(clean, (0, max_length - clean.shape[-1]))
            )
            padded_noisy.append(
                torch.nn.functional.pad(noisy, (0, max_length - noisy.shape[-1]))
            )

        # Stack the padded tensors to create the batch
        padded_clean = torch.stack(padded_clean)
        padded_noisy = torch.stack(padded_noisy)

        return padded_clean, padded_noisy


class DenoisingDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[Path, str] = "data/",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super(DenoisingDataModule, self).__init__()

        self.save_hyperparameters()

    def setup(self, stage: str):
        full_dataset = DenoisingDataset(self.hparams.data_dir)

        self.train, self.val, self.test = random_split(
            dataset=full_dataset,
            lengths=[0.9, 0.05, 0.05],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.train.dataset.custom_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.val.dataset.custom_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.test.dataset.custom_collate,
        )

from os import listdir
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class DenoisingDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[Path, str],
        filelist: Union[Path, str],
        sort: bool = True,
    ) -> None:
        """
        Custom PyTorch Dataset for denoising data.

        Args:
            data_dir (Union[Path, str]): Directory path containing 'clean' and 'noisy' subdirectories.
            filelist (Union[Path, str]): Path to the CSV file containing a list of filenames (without extensions).
            sort (bool, optional): Whether to sort the files based on their lengths. Defaults to True. This is
            useful due to the amount of padding that it is added in each batch.
        """
        super(DenoisingDataset, self).__init__()

        self.data_dir = Path(data_dir)
        self.filelist = Path(filelist)
        self.sort = sort

        self.clean_path = self.data_dir / "clean"
        self.noisy_path = self.data_dir / "noisy"
        self.filelist_df = pd.read_csv(filelist, sep='|', quotechar='\0')

        self.clean_files = sorted(
            [self.clean_path / f'{file}.pt' for file in self.filelist_df['basename']]
        )
        self.noisy_files = sorted(
            [self.noisy_path / f'{file}.pt' for file in self.filelist_df['basename']]
        )

        if sort:
            self.frames = pd.DataFrame(
                data={'clean': self.clean_files, 'noisy': self.noisy_files}
            )
            # Sort based on the ground truth frames
            self.frames['frames'] = self.frames['clean'].apply(
                lambda x: torch.load(x).size(-1)
            )
            self.frames = self.frames.sort_values(by='frames')

            self.clean_files = self.frames['clean'].tolist()
            self.noisy_files = self.frames['noisy'].tolist()

    def __len__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return len(self.clean_files)

    def __getitem__(self, index):
        clean = torch.load(self.clean_files[index]).unsqueeze(0)
        noisy = torch.load(self.noisy_files[index]).unsqueeze(0)

        # TODO: when there is a mismatch in the shapes, should we trim or pad?

        return clean, noisy

    @staticmethod
    def custom_collate(batch):
        """
        Custom collate function to handle padding within a batch.

        Args:
            batch (List[Tuple[torch.Tensor, torch.Tensor]]): A list of tuples containing clean and noisy tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing padded clean and noisy tensors.
        """
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
        train_filelist: Union[Path, str] = 'data/train.txt',
        val_filelist: Union[Path, str] = 'data/val.txt',
        test_filelist: Union[Path, str] = 'data/test.txt',
        sort: bool = True,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        """
        DataModule for denoising data used in PyTorch Lightning.

        Args:
            data_dir (Union[Path, str], optional): Directory path containing 'clean' and 'noisy' subdirectories.
            train_filelist (Union[Path, str], optional): Path to the CSV file containing a list of filenames for training set.
            val_filelist (Union[Path, str], optional): Path to the CSV file containing a list of filenames for validation set.
            test_filelist (Union[Path, str], optional): Path to the CSV file containing a list of filenames for test set.
            sort (bool, optional): Whether to sort the files based on their lengths. Defaults to True.
            batch_size (int, optional): Batch size for DataLoader. Defaults to 64.
            num_workers (int, optional): Number of workers for DataLoader. Defaults to 4.
        """
        super(DenoisingDataModule, self).__init__()

        self.save_hyperparameters()

    def setup(self, stage: str):
        self.train_dataset = DenoisingDataset(
            data_dir=self.hparams.data_dir,
            filelist=self.hparams.train_filelist,
            sort=self.hparams.sort,
        )

        self.val_dataset = DenoisingDataset(
            data_dir=self.hparams.data_dir,
            filelist=self.hparams.val_filelist,
            sort=self.hparams.sort,
        )

        self.test_dataset = DenoisingDataset(
            data_dir=self.hparams.data_dir,
            filelist=self.hparams.test_filelist,
            sort=self.hparams.sort,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.train_dataset.custom_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.val_dataset.custom_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.test_dataset.custom_collate,
        )

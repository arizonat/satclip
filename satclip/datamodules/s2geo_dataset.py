import os
from typing import Any, Callable, Dict, Optional

import pandas as pd
import rasterio
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset
import matplotlib.pyplot as plt
import numpy as np
import torch

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .transforms import get_pretrained_s2_train_transform, get_s2_train_transform, get_precomputed_train_transform

CHECK_MIN_FILESIZE = 10000 # 10kb

class S2GeoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/u/leca5365/Data/satclip-1M",
        index_fn: str = "index.csv",
        batch_size: int = 64,
        num_workers: int = 6,
        crop_size: int = 256,
        val_random_split_fraction: float = 0.1,
        transform: str = 'pretrained',
        mode: str = "both",
        embeddings_fn: str = "embeddings.parquet"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        if transform=='pretrained':
            self.train_transform = get_pretrained_s2_train_transform(resize_crop_size=crop_size)
        elif transform=='default':
            self.train_transform = get_s2_train_transform()
        else:
            self.train_transform = transform

        if self.mode == "precomputed":
            self.train_transform = get_precomputed_train_transform()
            
        self.val_random_split_fraction = val_random_split_fraction

        self.index_fn = index_fn
        self.embeddings_fn = embeddings_fn
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_dir):
            print("""
            No dataset found. To download, please follow instructions on: https://github.com/microsoft/satclip
            """)

    def setup(self, stage="fit"):
        dataset = S2Geo(root=self.data_dir, index_fn=self.index_fn, embeddings_fn=self.embeddings_fn, transform=self.train_transform, mode=self.mode)

        N_val = int(len(dataset) * self.val_random_split_fraction)
        N_train = len(dataset) - N_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [N_train, N_val])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            #persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        raise NotImplementedError

class S2Geo(NonGeoDataset):
    """S2-100K dataset.

    This dataset contains 100,000 256x256 patches of 12 band Sentinel imagery sampled randomly
    from Sentinel 2 scenes on the Microsoft Planetary Computer that have <20% cloud cover,
    intersect land, and were captured between 2021-01-01 and 2023-05-17 (there are 2,359,972
    such scenes).
    """

    validation_filenames = [
        "index-balanced-1M.csv",
        "images/",
        #"images/patch_0.tif",
        #"images/patch_99999.tif",
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        mode: Optional[str] = "both",
        embeddings_fn: str = "embeddings.parquet",
        index_fn: str = "index.csv",
    ) -> None:
        """Initialize a new S2-100K dataset instance.
        Args:
            root: root directory of S2-100K pre-sampled dataset
            transform: torch transform to apply to a sample
            mode: which data to return (options are "both" or "points"), useful for embedding locations without loading images 
        """
        assert mode in ["both", "points", "precomputed"]
        self.embeddings_fn = embeddings_fn
        self.index_fn = index_fn
        self.root = root
        self.transform = transform
        self.mode = mode
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        df = pd.read_csv(os.path.join(self.root, self.index_fn))
        self.filenames = []
        self.points = []
        self.embeddings = []

        # Parse the embeddings parquet files
        if self.mode == "precomputed":
            embeddings_path = os.path.join(self.root, self.embeddings_fn)

            if not os.path.exists(embeddings_path):
                raise RuntimeError(f"Embeddings file {embeddings_path} not found.")
            embeddings_df = pd.read_parquet(embeddings_path)
            embeddings_dict = dict(zip(embeddings_df['image_path'], embeddings_df['embedding']))
            print(f"Loaded {len(embeddings_dict)} embeddings from {embeddings_path}")

        n_skipped_files = 0
        for i in range(df.shape[0]):
            filename = os.path.join(self.root, "images", df.iloc[i]["fn"])

            if os.path.getsize(filename) < CHECK_MIN_FILESIZE:
                n_skipped_files += 1
                continue

            self.filenames.append(df.iloc[i]["fn"])
            self.points.append(
                (df.iloc[i]["lon"], df.iloc[i]["lat"])
            )

            if self.mode == "precomputed":
                self.embeddings.append(embeddings_dict[df.iloc[i]["fn"]])

        if self.mode == "precomputed":
            if len(self.embeddings) != len(self.filenames):
                raise RuntimeError("Some embeddings are missing for the given filenames.")

        print(f"skipped {n_skipped_files}/{len(df)} images because they were smaller "
              f"than {CHECK_MIN_FILESIZE} bytes... they probably contained nodata pixels")

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            dictionary with "image" and "point" keys where point is in (lon, lat) format
        """
        point = torch.tensor(self.points[index])
        sample = {"point": point}

        if self.mode == "both":
            with rasterio.open(self.filenames[index]) as f:
                data = f.read().astype(np.float32)
            #img = torch.tensor(data)
            sample["image"] = data

        elif self.mode == "precomputed":
            sample["embedding"] = self.embeddings[index]
            
        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as e:
                print(f"Error applying transform to sample {index}: {e}, {sample}")
                raise e
        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.
        Returns:
            length of dataset
        """
        return len(self.filenames)

    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.
        Returns:
            True if the dataset directories and split files are found, else False
        """
        print(f"Checking integrity of {self.root}...")
        for filename in self.validation_filenames:
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                print(filepath +' missing' )
                return False
        return True

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.
        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
        ncols = 1

        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        ax.imshow(image[:, :, [3,2,1]] / 4000)
        ax.axis("off")

        if show_titles:
            ax.set_title(f"({sample['point'][0]:0.4f}, {sample['point'][1]:0.4f})")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

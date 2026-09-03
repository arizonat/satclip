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

from .transforms import get_precomputed_train_transform, get_pretrained_s2_train_transform, get_s2_train_transform, get_s2_train_transform_temporal, get_pretrained_s2_train_transform_temporal

import datetime
import calendar

CHECK_MIN_FILESIZE = 10000 # 10kb

class S2GeoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/geoclip_s2",
        embeddings_fn: str = "embeddings.parquet",
        batch_size: int = 64,
        num_workers: int = 6,
        crop_size: int = 256,
        val_random_split_fraction: float = 0.1,
        transform: str = 'pretrained',
        mode: str = "both",
        index_fn: str = "index.csv"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.embeddings_fn = embeddings_fn
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
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_dir):
            print(f"""
            No dataset found at {self.data_dir}. To download, please follow instructions on: https://github.com/microsoft/satclip
            """)

    def setup(self, stage="fit"):
        dataset = S2Geo(root=self.data_dir,
                        embeddings_fn=self.embeddings_fn,
                        index_fn=self.index_fn, 
                        transform=self.train_transform, 
                        mode=self.mode)

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

class S2GeoTemporalDataModule(S2GeoDataModule):
    def __init__(
        self,
        data_dir: str = "/data/geoclip_s2",
        embeddings_fn: str = "embeddings.parquet",
        batch_size: int = 64,
        num_workers: int = 6,
        crop_size: int = 256,
        val_random_split_fraction: float = 0.1,
        transform: str = 'pretrained',
        mode: str = "both",
        temporal_positional_encoding: Optional[str] = "normalized_day_of_year",
        index_fn: str = "index.csv"
    ):
        super().__init__()
        self.embeddings_fn = embeddings_fn
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode

        if transform=='pretrained':
            self.train_transform = get_pretrained_s2_train_transform_temporal(resize_crop_size=crop_size)
        elif transform=='default':
            self.train_transform = get_s2_train_transform_temporal()
        else:
            self.train_transform = transform

        if self.mode == "precomputed":
            self.train_transform = get_precomputed_train_transform()
            
        self.val_random_split_fraction = val_random_split_fraction
        self.temporal_positional_encoding = temporal_positional_encoding
        self.index_fn = index_fn
        self.save_hyperparameters()

    def setup(self, stage="fit"):
        dataset = S2GeoTemporal(root=self.data_dir, 
                                index_fn=self.index_fn,
                                embeddings_fn=self.embeddings_fn,
                                transform=self.train_transform, 
                                mode=self.mode, 
                                temporal_positional_encoding=self.temporal_positional_encoding)

        N_val = int(len(dataset) * self.val_random_split_fraction)
        N_train = len(dataset) - N_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [N_train, N_val])

class S2Geo(NonGeoDataset):
    """S2 dataset.
    """

    validation_filenames = [
        "index.csv",
        "images/",
    ]

    def __init__(
        self,
        root: str,
        index_fn: str = "index.csv",
        embeddings_fn: str = "embeddings.parquet",
        transform: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        mode: Optional[str] = "both",
    ) -> None:
        """Initialize a new dataset instance.
        Args:
            root: root directory of pre-sampled dataset
            transform: torch transform to apply to a sample
            mode: which data to return (options are "both" or "points"), useful for embedding locations without loading images 
        """
        assert mode in ["both", "points", "precomputed"]
        self.root = root
        self.transform = transform
        self.mode = mode
        self.index_fn = index_fn
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        df = pd.read_csv(os.path.join(self.root, self.index_fn))
        self.filenames = []
        self.points = []
        self.embeddings = []

        if self.mode == "precomputed":
            embeddings_path = os.path.join(self.root, embeddings_fn)
            if not os.path.exists(embeddings_path):
                raise RuntimeError(f"Embeddings file {embeddings_path} not found.")
            embeddings_df = pd.read_parquet(embeddings_path)
            embeddings_dict = dict(zip(embeddings_df['image_path'], embeddings_df['embedding']))

        n_skipped_files = 0
        for i in range(df.shape[0]):
            filename = os.path.join(self.root, "images", df.iloc[i]["fn"])
            
            if os.path.getsize(filename) < CHECK_MIN_FILESIZE:
                n_skipped_files += 1
                continue

            self.filenames.append(filename)
            self.points.append(
                (df.iloc[i]["lon"], df.iloc[i]["lat"])
            )

            if self.mode == "precomputed":
                self.embeddings.append(embeddings_dict.get(filename))

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
            sample = self.transform(sample)
            
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
    
class S2GeoTemporal(S2Geo):
    """Placeholder for future S2-Temporal dataset implementation."""
    def __init__(
        self,
        root: str,
        index_fn: str = "index.csv",
        embeddings_fn: str = "embeddings.parquet",
        transform: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        mode: Optional[str] = "both",
        temporal_positional_encoding: Optional[str] = "normalized_day_of_year"
    ) -> None:
        """Initialize a new S2-temporal dataset instance.
        Args:
            root: root directory of S2-temporal pre-sampled dataset
            transform: torch transform to apply to a sample
            mode: which data to return (options are "both" or "points"), useful for embedding locations without loading images 
            if mode is precomputed, the dataset will return precomputed embeddings instead of images
            temporal_positional_encoding: which temporal positional encoding to use (options are "normalized_day_of
        """
        assert mode in ["both", "points", "precomputed"]
        self.root = root
        self.transform = transform
        self.mode = mode
        self.temporal_positional_encoding = temporal_positional_encoding
        self.index_fn = index_fn

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        df = pd.read_csv(os.path.join(self.root, self.index_fn))
        self.filenames = []
        self.points = []
        self.embeddings = []

        if self.mode == "precomputed":
            embeddings_path = os.path.join(self.root, embeddings_fn)
            if not os.path.exists(embeddings_path):
                raise RuntimeError(f"Embeddings file {embeddings_path} not found.")
            embeddings_df = pd.read_parquet(embeddings_path)
            embeddings_dict = dict(zip(embeddings_df['image_path'], embeddings_df['embedding']))

        n_skipped_files = 0
        print("Parsing with temporal positional encoding:", self.temporal_positional_encoding)
        for i in range(df.shape[0]):
            filename = os.path.join(self.root, "images", df.iloc[i]["fn"])

            if os.path.getsize(filename) < CHECK_MIN_FILESIZE:
                n_skipped_files += 1
                continue

            self.filenames.append(filename)

            if self.mode == "precomputed":
                embedding = embeddings_dict.get(filename)
                if embedding is None:
                    raise RuntimeError(f"Embedding for {filename} not found.")
                # embedding = np.load(os.path.join(self.root, "embeddings", df.iloc[i]["fn"].replace(".tif", ".npy")))
                self.embeddings.append(embedding)

            # Parse S2 timestamp to POSIX time, note that S2 timestamps are in UTC and up to the second anyway
            t = self.parse_time(df.iloc[i]["ts"], temporal_positional_encoding = self.temporal_positional_encoding)

            if type(t) == tuple:
                pt = (df.iloc[i]["lon"], df.iloc[i]["lat"]) + t
            else:
                pt = (df.iloc[i]["lon"], df.iloc[i]["lat"], t)

            self.points.append(pt)

        # Normalization of temporal features to [0,1] is done here for encodings that require it
        # TODO: This normalization should be done in a more elegant way, but for now this works
        if self.temporal_positional_encoding == "normalized_posix_timestamp":
            # Get min and max timestamps
            timestamps = [p[2] for p in self.points]
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            print(f"Normalizing timestamps from [{min_ts}, {max_ts}] to [0,1]")

            # Normalize timestamps to [0,1]
            self.points = [(p[0], p[1], (p[2]-min_ts)/(max_ts-min_ts)) for p in self.points]

        elif self.temporal_positional_encoding == "norm_doy_norm_year":
            # Get min and max years
            years = [p[3] for p in self.points]
            min_year = min(years)
            max_year = max(years)
            print(f"Normalizing years from [{min_year}, {max_year}] to [0,1]")

            # Normalize years to [0,1]
            self.points = [(p[0], p[1], p[2], (p[3]-min_year)/(max_year-min_year)) for p in self.points]

        elif self.temporal_positional_encoding == "toy_norm_year":
            # Get min and max years
            years = [p[3] for p in self.points]
            min_year = min(years)
            max_year = max(years)
            print(f"Normalizing years from [{min_year}, {max_year}] to [0,1]")

            # Normalize years to [0,1]
            self.points = [(p[0], p[1], p[2], (p[3]-min_year)/(max_year-min_year)) for p in self.points]

        elif self.temporal_positional_encoding == "norm_year":
            # Get min and max years
            years = [p[2] for p in self.points]
            min_year = min(years)
            max_year = max(years)
            print(f"Normalizing years from [{min_year}, {max_year}] to [0,1]")

            # Normalize years to [0,1]
            self.points = [(p[0], p[1], (p[2]-min_year)/(max_year-min_year)) for p in self.points]

        print(f"skipped {n_skipped_files}/{len(df)} images because they were smaller "
            f"than {CHECK_MIN_FILESIZE} bytes... they probably contained nodata pixels")

    def _time_of_year(self, dt: datetime.datetime) -> float:
        """Compute the time of year as a float in [0,1].
        Args:
            dt: datetime object
        Returns:
            time of year as float in [0,1]
        """
        year_start = datetime.datetime(dt.year, 1, 1, tzinfo=dt.tzinfo).timestamp()
        seconds_in_year = 365 * 24 * 60 * 60 if not calendar.isleap(dt.year) else 366 * 24 * 60 * 60
        return (dt.timestamp() - year_start) / seconds_in_year
    
    def parse_time(self, time_str: str,
                    temporal_positional_encoding: str="normalized_day_of_year") -> float:
        """Parse a datetime string to POSIX timestamp.
        Args:
            time_str: datetime string in ISO 8601 format
        Returns:
            time encoding as float
        """
        # dt = datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f+00:00')
        dt = datetime.datetime.fromisoformat(time_str)

        if temporal_positional_encoding == "sin_norm_day_of_year":
            day_of_year = float(dt.timetuple().tm_yday-1)
            # Normalized day of year [-1,1]
            return np.sin(2 * np.pi * day_of_year / 364.0)
        
        elif temporal_positional_encoding == "normalized_day_of_year":
            # Normalized day of year [0,1]
            return float(dt.timetuple().tm_yday-1) / 364.0
        
        elif temporal_positional_encoding == "day_of_year":
            # 0-indexed day of year, note: leap years not considered
            return float(dt.timetuple().tm_yday-1)

        elif temporal_positional_encoding == "norm_doy_norm_year":
            return (float(dt.timetuple().tm_yday-1) / 364.0, dt.year)
        
        elif temporal_positional_encoding == "posix_timestamp":
            # POSIX timestamp from UTC, as float (seconds since epoch)
            return dt.timestamp()

        elif temporal_positional_encoding == "normalized_posix_timestamp":
            # POSIX timestamp from UTC, as float (seconds since epoch), normalization handled outside this function
            return dt.timestamp()

        elif temporal_positional_encoding == "toroidal":
            # Based on GTLoc paper, we encode time as a 2D toroidal representation (month, day) normalized to [0,1]
            t_tuple = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
            doms = calendar.monthrange(dt.year, dt.month)
            norm_month = (1/12) * ((t_tuple[1]-1) + (t_tuple[2]-1)/doms[1])
            norm_hour = (1/24) * (t_tuple[3] + (t_tuple[4]/60) + (t_tuple[5]/3600))
            return (norm_month, norm_hour)

        elif temporal_positional_encoding == "toy":
            # ToY (Time of Year) encoding, normalized to [0,1]
            return self._time_of_year(dt)

        elif temporal_positional_encoding == "toy_norm_year":
            # ToY (Time of Year) year encoding, normalized to [0,1]
            # Returns a tuple of (normalized_time_of_year, year), normalization handled outside this function
            return (self._time_of_year(dt), dt.year)
        
        elif temporal_positional_encoding == "norm_year":
            # Normalize year to [0,1] outside this function, returns year as float
            return dt.year
        
        else:
            return dt.timestamp()
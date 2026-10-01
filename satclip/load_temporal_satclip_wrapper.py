"""
Remade this file with just TemporalSatCLIP Wrapper.

Don't need ClimplicitWrapper for coordbench
"""

import calendar
import torch
from main import SatCLIPLightningModule
from satclip import TemporalSatCLIP
from torch import nn
import numpy as np

class TemporalSatCLIPWrapper(nn.Module):
    def __init__(self, model_name: str = "tsatclip/linear", ckpt_path: str = None, device: str = "cuda", posix_min_time: int = None, posix_max_time: int = None, embedding_dim: int = 256):
        # Options for model_name: "tsatclip/linear", "tsatclip/doy"
        super().__init__()
        if model_name == "tsatclip/toroidal":
            # Older checkpoints did not properly have tpe_type set to toroidal, so we need to load the model and then set the tpe_type to toroidal
            print(f"Loading TemporalSatCLIP model from checkpoint {ckpt_path} with tpe_type set to toroidal")
            self.lightning_model = SatCLIPLightningModule.load_from_checkpoint(ckpt_path, tpe_type="toroidal")
        elif ckpt_path is not None:
            self.lightning_model = SatCLIPLightningModule.load_from_checkpoint(ckpt_path)
        else:
            self.model = TemporalSatCLIP(model_name=model_name)

        self.model_name = model_name
        self.lightning_model.eval()
        self.spatiotemporal_enc = self.lightning_model.model.location
        self.visual_enc = self.lightning_model.model.visual
        self.posix_min_time = posix_min_time
        self.posix_max_time = posix_max_time

        self.embedding_dim = embedding_dim

    def forward(self, x):
        # Wrapper expects (N, [lat, lon, time]) and returns embeddings
        # The TemporalSatCLIP model expects (N, [lon, lat, posix_time]) and returns embeddings

        x = x.clone()
        posix_time = x[..., 2]

        # Handle time conversion based on model_name
        if self.model_name == "tsatclip/doy":
            # Convert posix time to normalized day of year
            day_of_year = ((posix_time % 31556926) / 86400).long() + 1
            x[..., 2] = (day_of_year.int() - 1) / 364.0  # Normalize to [0, 1]

        elif self.model_name == "tsatclip/linear":
            # Convert posix time to linear time
            linear_time = (posix_time - self.posix_min_time) / (self.posix_max_time - self.posix_min_time)
            x[..., 2] = linear_time.float()

        elif self.model_name == "tsatclip/toroidal":
            # Convert posix time to toroidal representation
            dates = posix_time.cpu().numpy().astype('datetime64[M]')
            years = posix_time.cpu().numpy().astype('datetime64[Y]').astype(int) + 1970
            months = dates.astype(int) % 12 + 1
            days = posix_time.cpu().numpy().astype('datetime64[D]') - dates + 1
            doms = np.array([calendar.monthrange(year, month) for year, month in zip(years, months)])
            norm_month = torch.tensor(((months - 1) + ((days - 1) / doms[:, 1])).astype(float) / 12.0).to(x.device)


            hours = (posix_time % 86400) / 3600
            minutes = (posix_time % 3600) / 60
            seconds = posix_time % 60
            norm_hours = (hours + (minutes/60) + (seconds/3600)) / 24.0

            x = torch.cat([x[...,:-1], norm_month.unsqueeze(-1), norm_hours.unsqueeze(-1)], dim=-1)

        with torch.no_grad():
            # x = x.permute(1, 0, 2)  # Change to (lon, lat, time)
            x[..., :2] = x[..., :2][:, [1, 0]]  # Change to (lon, lat)
            embeddings = self.spatiotemporal_enc(x).detach()
        return embeddings
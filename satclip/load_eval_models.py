import torch
from main import SatCLIPLightningModule
from satclip import TemporalSatCLIP
from torch import nn
import numpy as np

import gtloc

DEFAULT_POSIX_MIN_TIME = 1609487709.024  # Jan 1, 2021
DEFAULT_POSIX_MAX_TIME = 1767609639.024  # Dec 31, 2025
DEFAULT_TS_LINEAR_CKPT_PATH = "/home/leca5365/Documents/satclip/satclip/satclip_temporal_logs/satclip-s2-1M-100k-satclip_loss-normalized_posix_timestamp/satclip-s2-satcliploss-1M-normalized_posix_timestamp/satclip-s2-satcliploss-1M-normalized_posix_timestamp/checkpoints/last.ckpt"
DEFAULT_TS_DOY_CKPT_PATH = "/home/leca5365/Documents/satclip/satclip/satclip_temporal_logs/satclip-s2-satcliploss-1M/satclip-s2-satcliploss-1M/checkpoints/last-v1.ckpt"
DEFAULT_GTLOC_CKPT_PATH = "/home/leca5365/Documents/gtloc/ckpts/gtloc.pt"


class TemporalSatCLIPWrapper(nn.Module):
    def __init__(self, model_name: str = "tsatclip/linear", ckpt_path: str = None, device: str = "cuda", posix_min_time: int = None, posix_max_time: int = None, embedding_dim: int = 256):
        # Options for model_name: "tsatclip/linear", "tsatclip/doy"
        super().__init__()
        if ckpt_path is not None:
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

        with torch.no_grad():
            # x = x.permute(1, 0, 2)  # Change to (lon, lat, time)
            x[..., :2] = x[..., :2][:, [1, 0]]  # Change to (lon, lat)
            embeddings = self.spatiotemporal_enc(x).detach()
        return embeddings

class GTLocWrapper(nn.Module):
    def __init__(self, ckpt_path: str = DEFAULT_GTLOC_CKPT_PATH, device: str = "cuda"):
        super().__init__()
        self.gtl_model = gtloc.GTLoc(
                hidden_dim= 768,
                embedding_dim= 512,
                queue_size=4096,
                time_sigma=[2**0, 2**4, 2**8],
                loc_sigma= [2**0, 2**4, 2**8],
                freeze_backbone=True,
                galleries='data_dist',
                time_dropout=0.1,
        )
        self.state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        self.gtl_model.load_state_dict(self.state_dict, strict=False)

        self.gtl_model = self.gtl_model.to(device)
        self.gtl_loc_encoder = self.gtl_model.location_encoder
        self.gtl_time_encoder = self.gtl_model.time_encoder

        self.embedding_dim = self.gtl_loc_encoder.embedding_dim + self.gtl_time_encoder.embedding_dim

    def forward(self, x):
        # Wrapper expects (lat, lon, time) and returns embeddings
        # The GTLoc model expects (lon, lat, posix_time) and returns embeddings

        # transform posix time to month-day-hour-minute-second format
        x = x.clone()
        posix_time = x[..., 2]
        lat_lon = x[..., :2].float()

        month = ((posix_time % 31556926) / 2629800).long() + 1
        day = ((posix_time % 2629800) / 86400).long() + 1
        hour = ((posix_time % 86400) / 3600).long()
        minute = ((posix_time % 3600) / 60).long()
        second = (posix_time % 60).long()
        time_rep = torch.stack([month, day, hour, minute, second], dim=-1).float()


        with torch.no_grad():
            loc_embeddings = self.gtl_loc_encoder(lat_lon) 
            time_embeddings = self.gtl_time_encoder(time_rep)
            embeddings = torch.cat([loc_embeddings, time_embeddings], dim=1).detach()
        return embeddings

def load_gtloc_model(ckpt_path: str = DEFAULT_GTLOC_CKPT_PATH, device: str = "cuda"):
    return GTLocWrapper(ckpt_path=ckpt_path, 
                        device=device)

def load_temporal_satclip_linear_model(model_name: str = "tsatclip/linear", 
                                       ckpt_path: str = DEFAULT_TS_LINEAR_CKPT_PATH, 
                                       device: str = "cuda", 
                                       posix_min_time: float = DEFAULT_POSIX_MIN_TIME, 
                                       posix_max_time: float = DEFAULT_POSIX_MAX_TIME):

    return TemporalSatCLIPWrapper(model_name=model_name, 
                                  ckpt_path=ckpt_path, 
                                  device=device, 
                                  posix_min_time=posix_min_time, 
                                  posix_max_time=posix_max_time)


def load_temporal_satclip_doy_model(model_name: str = "tsatclip/doy", 
                                   ckpt_path: str = DEFAULT_TS_DOY_CKPT_PATH, 
                                   device: str = "cuda"):
    return TemporalSatCLIPWrapper(model_name=model_name, 
                                  ckpt_path=ckpt_path, 
                                  device=device)

_REGISTERED_MODELS = {
    "tsatclip/linear": load_temporal_satclip_linear_model,
    "tsatclip/doy": load_temporal_satclip_doy_model,
    "gtloc": load_gtloc_model
}
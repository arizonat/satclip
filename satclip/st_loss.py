import torch
import torch.nn.functional as F
import torch.nn as nn

class TemporalSatCLIPLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            cache_labels=False,
            rank=0,
            world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, logits_per_image, logits_per_coord, autocorrelations_per_image=None, output_dict=False):
        device = logits_per_image.device

        if autocorrelations_per_image is None:
            autocorrelations_per_image = torch.ones((logits_per_image.shape[0], logits_per_image.shape[1]), device=logits_per_image.device)

        log_w = autocorrelations_per_image.log()
        weighted_logits_per_image = log_w + logits_per_image
        weighted_logits_per_coord = log_w.t() + logits_per_coord

        loss_i = -(torch.diagonal(weighted_logits_per_image) - torch.logsumexp(weighted_logits_per_image, dim=1)).mean()
        loss_c = -(torch.diagonal(weighted_logits_per_coord) - torch.logsumexp(weighted_logits_per_coord, dim=1)).mean()

        total_loss = (loss_i + loss_c) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

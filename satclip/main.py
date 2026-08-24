from datetime import datetime
from pathlib import Path

import lightning.pytorch
import torch
from datamodules.s2geo_dataset import S2GeoDataModule, S2GeoTemporalDataModule
from lightning.pytorch.cli import LightningCLI
from loss import SatCLIPLoss, SoftSatCLIPLoss
from model import SatCLIP, TemporalSatCLIP

from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision('high')

# print("Setting anomaly detection to True")
# torch.autograd.set_detect_anomaly(True)

class SatCLIPLightningModule(lightning.pytorch.LightningModule):
    def __init__(
        self,
        is_temporal: bool = False,
        loss_type: str = "satclip_loss", # satclip_loss, soft_loss
        embed_dim=512,
        image_resolution=256,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        in_channels=4,
        le_type="grid",
        pe_type="siren",
        te_type="fourier",
        te_k=10,
        frequency_num=16,
        max_radius=260,
        min_radius=1,
        legendre_polys=16,
        harmonics_calculation="analytic",
        sh_embedding_dims=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_hidden_layers=2,
        capacity=256,
        temporal_loss=None,
    ) -> None:
        super().__init__()

        if is_temporal:
            self.model = TemporalSatCLIP(
                embed_dim=embed_dim,
                image_resolution=image_resolution,
                vision_layers=vision_layers,
                vision_width=vision_width,
                vision_patch_size=vision_patch_size,
                in_channels=in_channels,
                le_type=le_type,
                pe_type=pe_type,
                te_type=te_type,
                te_k=te_k,
                temporal_loss=temporal_loss,
                frequency_num=frequency_num,
                max_radius=max_radius,
                min_radius=min_radius,
                legendre_polys=legendre_polys,
                harmonics_calculation=harmonics_calculation,
                sh_embedding_dims=sh_embedding_dims,
                num_hidden_layers=num_hidden_layers,
                capacity=capacity,
                loss_type=loss_type,
            )
            self.is_temporal = True
        else:
            self.model = SatCLIP(
                embed_dim=embed_dim,
                image_resolution=image_resolution,
                vision_layers=vision_layers,
                vision_width=vision_width,
                vision_patch_size=vision_patch_size,
                in_channels=in_channels,
                le_type=le_type,
                pe_type=pe_type,
                frequency_num=frequency_num,
                max_radius=max_radius,
                min_radius=min_radius,
                legendre_polys=legendre_polys,
                harmonics_calculation=harmonics_calculation,
                sh_embedding_dims=sh_embedding_dims,
                num_hidden_layers=num_hidden_layers,
                capacity=capacity,
            )
            self.is_temporal = False

        if loss_type == "soft_loss":
            self.loss_type = "soft_loss"
            self.loss_fun = SoftSatCLIPLoss()
        elif loss_type == "satclip_loss":
            self.loss_type = "satclip_loss"
            self.loss_fun = SatCLIPLoss()

        print(f"using loss function: {loss_type}")
        print(f"using temporal model: {self.is_temporal}")

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def common_step(self, batch, batch_idx):
        images = batch["image"]
        t_points = batch["point"].float()
        
        if self.loss_type == "soft_loss":
            logits_per_image, logits_per_coord, autocorrelations_per_image = self.model(images, t_points)
            loss = self.loss_fun(logits_per_image, logits_per_coord, autocorrelations_per_image)
        else:
            logits_per_image, logits_per_coord = self.model(images, t_points)
            loss = self.loss_fun(logits_per_image, logits_per_coord)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True)
        self.log("epoch_train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {
                    "params": rest_params,
                    "weight_decay": self.weight_decay,
                },  # specify in configs/default.yaml
            ],
            lr=self.learning_rate,  # specify in configs/default.yaml
        )

        return optimizer


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--watchmodel", action="store_true")

def cli_main(default_config_filename="./configs/default.yaml"):
    save_config_fn = default_config_filename.replace(".yaml", "-latest.yaml")
    # modify configs/default.yaml for learning rate etc.

    # Parse the yaml config just for the datamodule to get the is_temporal flag, which determines which datamodule to use
    import yaml
    with open(default_config_filename, "r") as f:
        config = yaml.safe_load(f)
    is_temporal = config["model"]["is_temporal"]

    if is_temporal:
        print("Using temporal datamodule")
        datamodule_class = S2GeoTemporalDataModule
    else:
        print("Using non-temporal datamodule")
        datamodule_class = S2GeoDataModule

    cli = MyLightningCLI(
        model_class=SatCLIPLightningModule,
        datamodule_class=datamodule_class,  # this will be overridden in MyLightningCLI.before_instantiate_classes
        save_config_kwargs=dict(
            config_filename=save_config_fn,
            overwrite=True,
        ),
        trainer_defaults={
            "accumulate_grad_batches": 16,
            "log_every_n_steps": 3,
        },
        parser_kwargs={"default_config_files": [default_config_filename]},
        seed_everything_default=0,
        run=False,
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if cli.config["model"]["is_temporal"]:
        run_name = f"SatCLIP_S2_Temporal_{ts}"
    else:
        run_name = f"SatCLIP_S2_{ts}"

    for logger in cli.trainer.loggers:
        if isinstance(logger, WandbLogger):
            logger.experiment.name = run_name
        logger.log_hyperparams(cli.datamodule.hparams)

    # if cli.trainer.logger is not None:
    #     cli.trainer.logger.experiment.name = run_name

    #     # this seems to be necessary to force logging of datamodule hyperparams
    #     cli.trainer.logger.log_hyperparams(cli.datamodule.hparams)

    # Create folder to log configs
    # NOTE: Lightning does not handle config paths with subfolders
    dirname_cfg = Path(default_config_filename).parent
    dir_log_cfg = Path(cli.trainer.log_dir) / dirname_cfg
    dir_log_cfg.mkdir(parents=True, exist_ok=True)

    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
    )

if __name__ == "__main__":
    config_fn = "./configs/default.yaml"

    #A100 go vroom vroom 🚗💨
    # if torch.cuda.get_device_name(device=0)=='NVIDIA A100 80GB PCIe':
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     print('Superfastmode! 🚀')
    # else:
    #     torch.backends.cuda.matmul.allow_tf32 = False
    cli_main(config_fn)

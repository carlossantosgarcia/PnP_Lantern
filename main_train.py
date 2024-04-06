# A Primal-Dual Data-Driven Method for Computational Optical Imaging with a
# Photonic Lantern
#
# Version 1.0
# May 2023
#
# Authors:
#   - Mathilde Larchevêque (mathilde.larcheveque@student-cs.fr)
#   - Solal O'Sullivan (solal.osullivan@student-cs.fr)
#   - Carlos Santos García (carlos.santos@student-cs.fr)
#   - Martin Van Waerebeke (martin.vw@student-cs.fr)
#
# For details on the method, refer to the article (WIP)

import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchvision import transforms

from train.data import ImageNetDataModule
from train.loss import LossConstructor
from train.model.model import DnCNN, UNet
from train.train_utils import weights_init_kaiming
from utils import date_to_string


def load_model(model_name: str, config: dict) -> torch.nn.Module:
    if model_name.lower() == "dncnn":
        model = DnCNN(
            in_channels=config["model"][model_name]["in_channels"],
            out_channels=config["model"][model_name]["out_channels"],
            num_layers=config["model"][model_name]["num_layers"],
            features=config["model"][model_name]["features"],
            kernel_size=config["model"][model_name]["kernel_size"],
            residual=config["model"][model_name]["residual"],
        )
    elif model_name.lower() == "unet":
        model = UNet(
            in_nc=config["model"][model_name]["in_nc"],
            out_nc=config["model"][model_name]["out_nc"],
            nc=config["model"][model_name]["nc"],
            nb=config["model"][model_name]["nb"],
            act_mode=config["model"][model_name]["act_mode"],
            downsample_mode=config["model"][model_name]["downsample_mode"],
            upsample_mode=config["model"][model_name]["upsample_mode"],
        )

    # Applies Kaiming initialization for weights
    model.apply(weights_init_kaiming)

    return model


class LitModel(pl.LightningModule):
    """
    Denoiser to train using PyTorch Lightning.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = load_model(
            model_name=config["model"]["model_name"], config=config
        )
        self.loss = LossConstructor(config=config["training"]["losses"])
        self.train_psnr = PSNR(data_range=1.0, reduction="elementwise_mean")
        self.val_psnr = PSNR(data_range=1.0, reduction="elementwise_mean")
        self.test_psnr = PSNR(data_range=1.0, reduction="elementwise_mean")
        self.last_logged_epoch = -1
        self.num_images_to_log = 10
        self.inv_transform = transforms.Compose(
            [
                # transforms.Normalize((0.0), (1 / 0.269)),
                transforms.Normalize((0.0), (1 / 255)),
            ]
        )
        self.kwargs_needed = config["training"]["losses"]["jacobian_reg"][
            "active"
        ]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config["training"]["lr"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.config["training"]["scheduler"]["factor"],
                    patience=self.config["training"]["scheduler"]["patience"],
                    verbose=True,
                ),
                "monitor": "val_loss",
                "frequency": 1,  # In number of validation runs
            },
        }

    def on_fit_start(self):
        """
        Runs right after .fit() is called. Sets random seeds.
        """
        wandb.config.update(self.config)
        pl.seed_everything(seed=42, workers=True)

    def on_train_start(self):
        """
        W&B model summary will now show the minimum value reached during training.
        """
        wandb.define_metric("val_loss_epoch", summary="min")

    def training_step(self, batch, batch_idx):
        """
        Training step. Logs train loss.
        """
        x_noisy, x_clean = batch
        out = self.model(x_noisy)
        kwargs = {}
        if self.kwargs_needed:
            kwargs["model"] = self.model
            kwargs["eval"] = False
        loss, logs, stats = self.loss(out, x_clean, **kwargs)
        self.train_psnr(out, x_clean)

        # Logs train loss
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        for name, val in logs:
            self.log(f"train_{name}", val, on_step=True, on_epoch=True)
        self.log("train_psnr", self.train_psnr, on_step=True, on_epoch=True)

        # Logs custom stats for losses
        self.log_stats(stats)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step. Logs validation loss and images.
        """
        x_noisy, x_clean = batch
        out = self.model(x_noisy)
        kwargs = {}
        if self.kwargs_needed:
            kwargs["model"] = self.model
            kwargs["eval"] = True
        loss, logs, stats = self.loss(out, x_clean, **kwargs)
        self.val_psnr(out, x_clean)

        # Logs validation loss
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        for name, val in logs:
            self.log(f"val_{name}", val, on_step=True, on_epoch=True)
        self.log("val_psnr", self.val_psnr, on_step=True, on_epoch=True)

        # Logs custom stats for losses
        self.log_stats(stats)

        # Logging images performed only once per epoch
        curr_epoch = self.current_epoch
        if self.last_logged_epoch < curr_epoch:
            self.log_denoised_images(x_noisy, x_clean, out, title="validation")
            self.last_logged_epoch = curr_epoch
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step.
        """
        x_noisy, x_clean = batch
        out = self.model(x_noisy)
        kwargs = {}
        if self.kwargs_needed:
            kwargs["model"] = self.model
            kwargs["eval"] = True
        loss, logs, stats = self.loss(out, x_clean, **kwargs)
        self.test_psnr(out, x_clean)

        # Logs test loss
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        for name, val in logs:
            self.log(f"test_{name}", val, on_step=True, on_epoch=True)
        self.log("test_psnr", self.test_psnr, on_step=True, on_epoch=True)

        # Logs custom stats for losses
        self.log_stats(stats, name="test")

    def log_stats(self, stats, name=None):
        for loss_name in stats:
            if stats[loss_name] is not None:
                for stat_name in stats[loss_name]:
                    self.log(
                        stat_name if name is None else f"{name}_{stat_name}",
                        stats[loss_name][stat_name],
                        on_step=True,
                        on_epoch=True,
                    )

    def log_denoised_images(self, x_noisy, x_clean, out, title):
        num_samples = min(self.num_images_to_log, len(x_noisy))

        clean_imgs = [
            self.inv_transform(im).permute(1, 2, 0).cpu().numpy()
            for im in x_clean[:num_samples]
        ]

        noisy_imgs = [
            self.inv_transform(im).permute(1, 2, 0).cpu().numpy()
            for im in x_noisy[:num_samples]
        ]

        denoised_imgs = [
            self.inv_transform(im).permute(1, 2, 0).cpu().numpy()
            for im in out[:num_samples]
        ]

        for idx in range(num_samples):
            self.logger.experiment.log(
                {
                    f"Denoising results - {title}": wandb.Image(
                        np.concatenate(
                            [
                                clean_imgs[idx],
                                noisy_imgs[idx],
                                denoised_imgs[idx],
                            ],
                            axis=1,
                        ),
                        caption=f"step {self.global_step} at epoch {self.current_epoch}",
                    ),
                }
            )


def train(
    config_path: str,
    run_name: str = None,
    project_name: str = None,
    return_ckpt_dir: bool = False,
    resume_from_ckpt=None,
    noise_std: float = None,
    jacob_reg: bool = None,
    initial_weights: str = None,
):

    # Seed
    pl.seed_everything(seed=42, workers=True)

    DATE = date_to_string()
    # Loads training configuration
    config = yaml.safe_load(open(config_path, "r"))

    # Modifies config dict for command-line flags
    if jacob_reg is not None:
        config["training"]["losses"]["jacobian_reg"]["active"] = jacob_reg
    if noise_std is not None:
        config["data"]["noise_std"] = noise_std

    if resume_from_ckpt is None:
        model = LitModel(config=config)
        if initial_weights is not None:
            checkpoint = torch.load(
                initial_weights, map_location=torch.device("cuda")
            )
            model.load_state_dict(checkpoint["state_dict"])
    else:
        model = LitModel.load_from_checkpoint(resume_from_ckpt, config=config)

    # Datamodule
    dm = ImageNetDataModule(
        dataset_name=config["data"]["dataset_name"],
        root=config["data"]["root"],
        batch_size=config["training"]["batch_size"],
        noise_std=config["data"]["noise_std"],
    )

    # Early stop callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=config["training"]["early_stopping"]["min_delta"],
        patience=config["training"]["early_stopping"]["patience"],
        verbose=True,
        mode="min",
    )

    checkpoint_dirs = os.path.join(
        "train",
        "training_checkpoints",
        config["model"]["model_name"].lower(),
        run_name if run_name is not None else DATE,
    )
    if not os.path.exists(checkpoint_dirs):
        os.makedirs(checkpoint_dirs)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dirs,
        filename="{epoch}-{val_loss:.5f}-{jacobian_norm_mean:.5f}"
        if model.kwargs_needed
        else "{epoch}-{val_loss:.5f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # Monitors learning rate to check the impact of scheduler
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # W&B logger
    wandb_logger = WandbLogger(
        name=run_name if run_name is not None else DATE,
        project=project_name
        if project_name is not None
        else config["model"]["model_name"],
        id=run_name if run_name is not None else DATE,
    )

    # Main trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        log_every_n_steps=10,
        val_check_interval=1.0,
        max_epochs=-1,
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            early_stop_callback,
            lr_monitor,
            checkpoint_callback,
        ],
        fast_dev_run=False,
        inference_mode=False,
        # num_sanity_val_steps=0,
    )

    # Training loop
    trainer.fit(
        model=model,
        datamodule=dm,
    )

    # Test
    trainer.test(ckpt_path="best", datamodule=dm)

    # Saves weights and config
    wandb.save(f"{trainer.checkpoint_callback.dirpath}/*ckpt*")
    with open(
        os.path.join(trainer.checkpoint_callback.dirpath, "config.yaml"), "w"
    ) as f:
        yaml.dump(config, f, default_flow_style=False)

    if return_ckpt_dir:
        return trainer.checkpoint_callback.best_model_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Launches training with parameters from config.yml file."
    )
    parser.add_argument(
        "--config_path", type=str, help="YAML configuration file"
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, help="Path to checkpoint", default=None
    )
    parser.add_argument(
        "--run_name", type=str, help="Name of the run launched.", default=None
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        help="Noise standard deviation.",
        default=None,
    )
    parser.add_argument("--jacob_reg", dest="jacob_reg", action="store_true")
    parser.add_argument(
        "--no-jacob_reg", dest="jacob_reg", action="store_false"
    )
    parser.set_defaults(jacob_reg=None)
    parser.add_argument(
        "--initial_weights",
        type=str,
        help="Path to pretrained model.",
        default=None,
    )
    args = parser.parse_args()

    ckpt_path = train(
        args.config_path,
        resume_from_ckpt=args.resume_from_ckpt,
        run_name=args.run_name,
        noise_std=args.noise_std,
        jacob_reg=args.jacob_reg,
        return_ckpt_dir=True,
        initial_weights=args.initial_weights,
    )

    # Import here to avoid circular imports
    from eval_models import eval_model

    eval_model(ckpt_path=ckpt_path, run_name=args.run_name)

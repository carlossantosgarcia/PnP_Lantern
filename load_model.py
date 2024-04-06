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

import os

import torch
import yaml

from main_train import LitModel


def load_from_checkpoint(checkpoint: str = None) -> LitModel:
    """Creates and loads the Lightning module from the given checkpoint

    Args:
        checkpoint (str, optional): Path to the checkpoint to load. Defaults to
        None.

    Returns:
        LitModel: Lightning module with the loaded model in float mode.
    """
    if checkpoint is None:
        model = LitModel(
            config=yaml.safe_load(
                open(os.path.join("train", "config/config.yml"), "r")
            )
        )
    else:
        config_path = os.path.join(os.path.dirname(checkpoint), "config.yaml")
        config = yaml.safe_load(open(config_path, "r"))
        model = LitModel.load_from_checkpoint(
            checkpoint,
            config=config,
            map_location=None
            if torch.cuda.is_available()
            else torch.device("cpu"),
        )
        model = model.float()
    return model


def load_model(model_name: str, checkpoint: str = None):

    if model_name == "UNetRes":
        raise NotImplementedError

    if model_name.lower() == "dncnn":

        class WrapperModel(torch.nn.Module):
            """Allows more flexible arguments to use different models"""

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, **kwargs):
                input_img = (
                    torch.from_numpy(kwargs["img"])
                    .float()[(None,) * 2]
                    .to(kwargs["device"])
                )
                output = self.model(input_img)[0][0].cpu()
                return output.detach().numpy()

            def preprocess(self, **kwargs):
                return kwargs["img"], None

            def postprocess(self, **kwargs):
                return kwargs["img"]

        model = load_from_checkpoint(checkpoint=checkpoint)
        net = WrapperModel(model=model)
        net.eval()

    if model_name.lower() == "unet":
        raise NotImplementedError

    if torch.cuda.is_available():
        net = net.cuda()
    return net

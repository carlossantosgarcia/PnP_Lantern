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

import copy
import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim


class Parameters:
    """Parameters struct"""

    def __init__(
        self,
        Nx: int,
        Ny: int,
        x0: np.ndarray,
        Psi_t: callable,
        Psi: callable,
        noise: float,
        normPsi: float,
        normPhi: float,
        lambda_: float,
        u: np.ndarray = None,
        v: np.ndarray = None,
        weights: float = None,
        stopcrit: float = 1e-6,
        stopnorm: float = 1e-6,
        stopbound: float = 1e-6,
        NbIt: int = 100,
        display: int = 10,
        epsilon: float = 1e-6,
        output_dir: str = "",
    ):
        # Dimensions of the image of interest
        self.Nx = Nx
        self.Ny = Ny

        # Initialisation for the algorithm
        self.x0 = x0

        # Initialisation for the dual variables - if available
        self.u = u
        self.v = v

        # Forward and backward operators for the sparsity basis
        self.Psi_t = Psi_t
        self.Psi = Psi

        self.noise = noise
        # Spectral norm of Psi
        self.normPsi = normPsi
        # Spectral norm of Phi
        self.normPhi = normPhi
        # Weights - if available
        self.weights = weights
        # Free parameter acting on the convergence speed
        self.lambda_ = lambda_
        # Stopping criterion checking on the relative change between 2
        # consecutives values of the objective function
        self.stopcrit = stopcrit
        # Stopping criterion checking on the relative change between 2
        # consecutives iterates
        self.stopnorm = stopnorm
        self.stopbound = stopbound  # Tolerance for the l2 constraint
        self.NbIt = NbIt  # Number of iterations
        self.display = display  # Number of iterations to display information
        self.epsilon = epsilon  # Criterion for minimisation problem
        self.output_dir = output_dir


def date_to_string():
    """Creates a string with current date information, e.g.: '14_May_16h23'"""
    now = datetime.now()
    current_day = now.day
    current_month = now.strftime("%h")
    current_hour = now.strftime("%H")
    current_min = now.strftime("%M")
    current_sec = now.strftime("%S")
    return f"{current_day}_{current_month}_{current_hour}h{current_min}min{current_sec}s"


def create_circular_mask(
    h: int, w: int, center=None, radius=None
) -> np.ndarray:
    """Creates a circular mask to compute PSNR and SSIM scores on observed regions"""
    if center is None:
        # Use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:
        # Use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def save_vector_to_img(
    x: np.ndarray,
    folder: str,
    file_name: str,
    shape: tuple,
    x_true: np.ndarray = None,
) -> None:
    """Saves a given array as a .npy file"""
    assert ".npy" in file_name
    x_img = np.reshape(x, shape)
    with open(os.path.join(folder, file_name), "wb") as f:
        np.save(f, x_img)
    plt.imshow(x_img)
    title = ""
    if x_true is not None:
        h, w = shape
        bin_mask = 1 * create_circular_mask(h=h, w=w, radius=(h - 12) // 2)
        title += f"\n PSNR: {psnr(bin_mask*x_true.reshape(shape), bin_mask*x.reshape(shape)):.3f}"
        title += f"\n SSIM: {ssim(bin_mask*x_true.reshape(shape), bin_mask*x.reshape(shape), data_range=255):.3f}"
    plt.title(title)
    plt.savefig(os.path.join(folder, file_name.replace(".npy", ".png")))


def psnr(x_target: np.ndarray, x_hat: np.ndarray) -> float:
    """Computes the PSNR between a target and an estimated image

    Args:
        x_target (np.ndarray): target image
        x_hat (np.ndarray): reconstruction

    Returns:
        float: Returns PSNR
    """
    return 20 * np.log(255 / (np.sqrt((np.square(x_target - x_hat)).mean())))


def save_config_to_json(param: Parameters) -> None:
    """Saves parameters to JSON file

    Args:
        param (Parameters): Parameters struct
    """
    params_dict = copy.deepcopy(vars(param))
    del params_dict["noise"]
    del params_dict["x0"]
    del params_dict["Psi"]
    del params_dict["Psi_t"]
    with open(os.path.join(param.output_dir, "params.json"), "w") as f:
        json.dump(params_dict, f)


def format_infos(output: list[float] | str) -> str:
    """Formats the output of the evaluation metrics before printing them."""
    if isinstance(output, list):
        return f"{output[0]:.3f}"
    else:
        return output

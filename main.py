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
import time
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import trange
import torch
from skimage.metrics import structural_similarity as ssim

import algorithm
from load_model import load_model
import utils


def load_data(
        mat_folder: str,
        use_synthetic_shapes: bool = False,
        pattern_index: int = 0,) -> tuple:
    """Util function to load operator files or synthetic data.

    Args:
        mat_folder (str): Path to the folder containing the needed
            objects for the inverse problem resolution such as the
            measurement operator, and problem constants.
        use_synthetic_shapes (bool, optional): If true, uses synthetic shape as
            X and corresponding measure y for the inverse problem. Defaults to
            False.
        pattern_index (int, optional): Index of the pattern to use for the
            synthetic shapes. Defaults to 0.

    Returns:
        tuple: Returns the wanted variables from the file:
            - Nx and Ny are the shapes of images X and measurement y
            - Phi is the linear measurement operator
            - xtrue and ydata are the (X, y) values used for the inverse
                problem
    """
    # Loads the data
    data = {}
    keys = ("Nx", "Ny", "Phi", "image_choice", "nb_pat", "xtrue", "ydata")
    for k in keys:
        data[k] = np.load(
            os.path.join(mat_folder, f"{k}.npy"),
            allow_pickle=True,
        ).item()[k]

    # Gets variables from dict
    Nx, Ny = int(data["Nx"]), int(data["Ny"])
    Phi = data["Phi"]
    image_choice = data["image_choice"]
    nb_pat = int(data["nb_pat"])

    if not use_synthetic_shapes:
        x_true, ydata = data["xtrue"], data["ydata"]
    else:
        # Load synthetic shapes as X and y
        shape_name = f"shape_{pattern_index}.npy"
        x_path = os.path.join(".", "shapes", shape_name)
        y_path = os.path.join(".", "shapes", f"obs_{shape_name}")
        x_true, ydata = np.load(x_path), np.load(y_path)

    return Nx, Ny, Phi, image_choice, nb_pat, x_true, ydata


def launch_main(
    mat_folder: str,
    isnr: float = 30.0,
    output_dir: str | None = None,
    method: Literal['wavelets', 'dncnn'] = "wavelets",
    checkpoint: str = None,
    run_name: str = None,
    nb_images: int = 1,
    save_images_logs: bool = False,
    epsilon: float = 50.0,
    nb_it: int = 1000,
    verbose: bool = True,
) -> dict:
    """Launches main image reconstruction algorithm

    Args:
        mat_folder (str): Path required to load the observation operators.
        isnr (float, optional): If given, adds noise to the observed data.
            Defaults to 30.0.
        output_dir (str, optional): If given, saves logs, final images and
            arrays to this folder. Defaults to None.
        method (str, optional): Method used for the reconstruction algorithm.
            Supported methods are "dncnn" and "wavelets". Defaults to
            "wavelets".
        checkpoint (str, optional): If given, checkpoint to load a given
            pretrained model. Defaults to None.
        run_name (str, optional): Name of the experiment run, useful when
            launching algorithm right after training. Defaults to None.
        nb_images (int, optional): Number of images to test. Defaults to 1.
        save_images_logs (bool, optional): If True, saves logs and reconstructed
            images. Defaults to False.
        epsilon (int, optional): Condition on the L2 norm for convergence of the
            algorithm, which will run until the target and reconstructed images
            are less than epsilon apart from each other (l2 norm-wise). Defaults
            to 50.
        nb_it (int, optional): Iterations for the Primal Dual algorithm.
            Defaults to 1000.
        verbose (bool, optional): If True, prints logs. Defaults to True.

    Returns:
        dict: Returns a dictionary keeping evaluation metrics (PSNR and SSIM
            statistics)
    """
    if not 1 <= nb_images <= 51:
        raise ValueError("Chooose a number of images between 1 and 51.")
    method = method.lower()

    if output_dir is None:
        # Creates default output directory
        output_dir = os.path.join(
            os.path.expanduser("~"),
            "saracoil_outputs",
            method,
            utils.date_to_string(),
        )
    elif run_name is not None:
        # Useful when launching evaluation right after training
        output_dir = os.path.join(
            output_dir, method, run_name, utils.date_to_string()
        )
    elif checkpoint is not None:
        # Useful when launching evaluation from a custom ckpt
        ckpt_name = os.path.basename(checkpoint).replace(".ckpt", "")
        output_dir = os.path.join(
            output_dir, method, ckpt_name, utils.date_to_string()
        )
    else:
        output_dir = os.path.join(output_dir, method, utils.date_to_string())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setting random seed
    np.random.seed(42)

    # Loops over the images
    eval_metrics = {"PSNR": [], "SSIM": [], "time(s)": [], "nb_iter": []}
    # Default image is the cross
    for index in trange(nb_images, desc="Inverse problems"):

        # Read measurement operator Phi and data
        Nx, Ny, Phi, _, _, x_true, ydata = load_data(
            mat_folder=mat_folder,
            use_synthetic_shapes=index > 0,
            pattern_index=index-1,
        )
        start = time.time()
        # Step of adding noise
        if isnr is not None:
            sigma_noise = (
                np.linalg.norm(ydata, ord=2)
                / np.size(ydata)
                * 10 ** (-isnr / 20)
            )
            ydata += sigma_noise * np.random.randn(*ydata.shape)

        if method == "wavelets":
            x0 = np.zeros((Nx * Ny, 1))
            # Sparse operator created for wavelets
            Psi, Psit = algorithm.SARA_sparse_operator(
                im=np.random.randn(Nx, Ny), level=2
            )
            normPsi = 1
            noise = None

        elif method == "dncnn":
            net = load_model(method, checkpoint=checkpoint)
            Nx0, Ny0 = Nx, Ny
            x0 = np.zeros((Nx, Ny))
            x0, noise = net.preprocess(
                img=x0,
                noise=True,
                vertical_padding=True,
                new_shape=None,
            )
            Phi, _ = net.preprocess(
                img=Phi,
                noise=True,
                vertical_padding=False,
                new_shape=(Phi.shape[0], x0.size),
            )
            Psi, Psit, normPsi = None, None, None

            # Nx, Ny are the new shape of the image (8kx8k)
            Nx, Ny = x0.shape

        else:
            raise NotImplementedError

        # Computing largest eigenvalues
        normPhi = algorithm.power_method(A=Phi, At=Phi.T, im_size=(Nx * Ny, 1))

        # Parameters for the Primal-Dual solver
        param = utils.Parameters(
            Nx=Nx,
            Ny=Ny,
            x0=x0,
            noise=noise,
            normPhi=normPhi,
            epsilon=epsilon,
            Psi=Psi,
            Psi_t=Psit,
            normPsi=normPsi,
            lambda_=1e-3,
            stopcrit=1e-3,
            stopnorm=1e-3,
            stopbound=1e-2,
            NbIt=nb_it,
            display=10,
            output_dir=output_dir,
        )

        # Save config for reproducibility
        utils.save_config_to_json(param=param)

        if method == "wavelets":
            # First estimate without weights
            Xtemp, param.u, param.v, logs = algorithm.SARA_solver(
                y=ydata,
                Phi=Phi,
                param=param,
                x_true=x_true,
                rw_iter=0,
                index=index,
                save=save_images_logs,
                verbose=verbose,
            )
            nb_iter = logs["nb_iter"]
            end = time.time()

            # Creates logs DataFrame
            df_logs = pd.DataFrame(logs)
            X_final, df_logs = algorithm.reweighting(
                x=Xtemp,
                ydata=ydata,
                Phi=Phi,
                param=param,
                x_true=x_true,
                df_logs=df_logs,
                index=index,
                save=save_images_logs,
                verbose=verbose,
            )

        elif method == "dncnn":
            # Plug-and-Play method
            # First estimate without weights
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

            Xtemp, param.u, logs = algorithm.PnP_solver(
                y=ydata,
                Phi=Phi,
                param=param,
                x_true=x_true,
                net=net,
                device=device,
                index=index,
                save=save_images_logs,
                verbose=verbose,
            )
            nb_iter = logs["nb_iter"]
            end = time.time()
            # Creates logs DataFrame
            df_logs = pd.DataFrame(logs)
            param.u = net.postprocess(
                img=param.u,
                new_shape=(Nx0, Ny0),
                vertical_padding=None,
            )
            X_final = net.postprocess(
                img=Xtemp,
                new_shape=(Nx0, Ny0),
                vertical_padding=True,
            )

        else:
            raise NotImplementedError

        # Computes PSNR and SSIM
        PSNR = utils.psnr(x_true, X_final)
        SSIM = ssim(X_final, x_true, data_range=255)
        eval_metrics["SSIM"].append(SSIM)
        eval_metrics["PSNR"].append(PSNR)
        eval_metrics["time(s)"].append(end-start)
        eval_metrics["nb_iter"].append(nb_iter)
    eval_metrics["output_dir"] = output_dir

    return eval_metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launches SARA-COIL algorithm")
    parser.add_argument(
        "--mat_folder",
        type=str,
        help="Folder containing operators.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to store logs and reconstructed outputs."
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Regularization method used for the Primal-Dual algorithm.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint to load the model.",
        default=None,
    )
    parser.add_argument(
        "--nb_images",
        type=int,
        help="Number of additional images to test. These are synthetic shapes"
        "randomly generated and used for the inverse problem. If ",
        default=1,
    )
    parser.add_argument(
        '--verbose',
        dest='verbose',
        action='store_true',
    )
    parser.add_argument(
        '--no-verbose',
        dest='verbose',
        action='store_false',
    )
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    launch_main(
        mat_folder=args.mat_folder,
        output_dir=args.output_dir,
        method=args.method,
        checkpoint=args.checkpoint,
        nb_images=args.nb_images,
        save_images_logs=True,
        verbose=args.verbose,
    )

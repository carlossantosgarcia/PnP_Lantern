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
import pandas as pd
import yaml

from main import launch_main
import utils


def eval_model(
    ckpt_path: str | None = None,
    run_name: str | None = None,
    isnr: float = 30.,
    nb_images: int = 1,
    nb_it: int = 1000,
    epsilon: float = 50.,
    use_wavelets: bool = True,
    verbose: bool = True,
):
    """Launches evaluation on a given model

    Args:
        ckpt_path (str | None, optional): Path to .ckpt file. If None, runs the
            evaluation on the SARA-COIL algorithm.
        run_name (str | None, optional): Name of the W&B run to evaluate when
            launching evaluation right after training. Defaults to None.
        isnr (float, optional): Noise level to corrupt data for the inverse
            problem. Defaults to 30.
        nb_images (int, optional): Number of images to run evaluation on.
            Defaults to 50.
        nb_it (int, optional): Number of iterations of the Primal-Dual
            algorithm. Defaults to 1000.
        epsilon (float, optional): Controls data fidelity constraint. Defaults to
            50.
        use_wavelets (bool, optional): If true, runs SARA-COIL algorithm with
            wavelets. Defaults to False.
        verbose (bool, optional): If true, prints intermediate logs. Defaults
            to True.
    """
    if ckpt_path is None and not use_wavelets:
        raise ValueError("ckpt_path must be provided if not using wavelets")
    if ckpt_path is not None and use_wavelets:
        raise ValueError("ckpt_path must be None if using wavelets")
    if use_wavelets and run_name is not None:
        raise ValueError("run_name must be None if using wavelets since no W&B"
                         "run is available.")
    method = "dncnn" if ckpt_path is not None else "wavelets"

    if not use_wavelets:
        # Loads config yaml from training config
        config_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
        config = yaml.safe_load(open(config_path, "r"))
        model_name = config["model"]["model_name"]
    else:
        model_name = "wavelets"

    eval_metrics = launch_main(
        mat_folder="data/",
        isnr=isnr,
        output_dir="model_eval",
        method=method,
        checkpoint=ckpt_path,
        run_name=run_name,
        nb_images=nb_images,
        save_images_logs=True,
        nb_it=nb_it,
        epsilon=epsilon,
        verbose=verbose,
    )

    models_eval = {
        "models": model_name,
        "PSNR_max": [max(eval_metrics["PSNR"])],
        "PSNR_min": [min(eval_metrics["PSNR"])],
        "PSNR_avg": [np.mean(eval_metrics["PSNR"])],
        "PSNR_std": [np.std(eval_metrics["PSNR"])],
        "PSNR_med": [np.median(eval_metrics["PSNR"])],
        "SSIM_max": [max(eval_metrics["SSIM"])],
        "SSIM_min": [min(eval_metrics["SSIM"])],
        "SSIM_avg": [np.mean(eval_metrics["SSIM"])],
        "SSIM_std": [np.std(eval_metrics["SSIM"])],
        "SSIM_med": [np.median(eval_metrics["SSIM"])],
        "time(s)_avg": [np.mean(eval_metrics["time(s)"])],
        "time(s)_std": [np.std(eval_metrics["time(s)"])],
        "nb_iter_avg": [np.mean(eval_metrics["nb_iter"])],
    }

    df_stats = pd.DataFrame(models_eval)
    df_stats.to_csv(
        os.path.join(eval_metrics["output_dir"], "eval.csv"), index=False
    )
    if ckpt_path is not None:
        df_stats.to_csv(
            os.path.join(os.path.dirname(ckpt_path), "eval.csv"), index=False
        )
    print("Final metrics: ")
    for k, v in models_eval.items():
        print(f"{k} = {utils.format_infos(v)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launches evaluation on image reconstruction task"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Checkpoint to the model to test.",
        default=None,
    )
    parser.add_argument(
        "--nb_images",
        type=int,
        help="Number of images to test.",
        default=1,
    )
    parser.add_argument(
        '--use_wavelets',
        dest='use_wavelets',
        action='store_true',
    )
    parser.add_argument(
        '--no-use_wavelets',
        dest='use_wavelets',
        action='store_false',
    )
    parser.set_defaults(use_wavelets=False)
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

    eval_model(
        ckpt_path=args.ckpt_path,
        nb_images=args.nb_images,
        use_wavelets=args.use_wavelets,
        use_synthetic_shapes=args.use_synthetic_shapes,
        verbose=args.verbose,
    )

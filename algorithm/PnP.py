import os

import numpy as np
import torch
import pandas as pd

from algorithm.PD_ghost_SARA import Parameters
from utils import save_vector_to_img


def PnP_solver(
    y: np.ndarray,
    Phi: np.ndarray,
    param: Parameters,
    x_true: np.ndarray,
    net: torch.nn.Module | None = None,
    device: torch.device | None = None,
    index: int | None = None,
    save: bool = False,
    verbose: bool = True,):

    """-------------------------------------------------------------------------
    COIL-SARA
    Primal dual algorithm for compressive optical imaging with a photonic
    lantern
    -------------------------------------------------------------------------
    Problem of interest: y = Phi * xtrue
    with - xtrue : original unknown image
        - Phi   : measurement matrix
                    concatenation of the projection patterns
        - y     : observations
    -------------------------------------------------------------------------
    Minimisation problem:
    Minimise || W Psit(x) ||_1 s.t.  x positive
                                and  || y - Phi*x || <= epsilon
    -------------------------------------------------------------------------
    *************************************************************************
    *************************************************************************
    version 0.0
    November 2018
    Author: Audrey Repetti
    Contact: arepetti@hw.ac.uk
    For details on the method, refer to the article
    Compressive optical imaging with a photonic lantern
    For further details on primal-dual algorithm, refer to
    L. Condat, ?A primal-dual splitting method for convex optimization
    involving Lipschitzian, proximable and linear composite terms,? J.
    Optimization Theory and Applications, vol. 158, no. 2, pp. 460-479, 2013
    *************************************************************************
    *************************************************************************"""

    # INITIALISATION
    if len(y.shape) != 2:
        y = y.reshape((-1, 1))

    sigma = 0.99 / np.sqrt(param.normPhi)
    tau = 0.99 / (sigma * param.normPhi)

    assert sigma * tau * param.normPhi < 1, "Error in the choice of step-sizes !"

    if param.u is not None:
        u = param.u
        if len(u.shape) != 2:
            u = u.reshape((-1, 1))
    else:
        u = 0 * y

    x = param.x0
    x_vect = np.reshape(param.x0, (-1, 1))
    Phix = np.matmul(Phi, x_vect)
    l2norm = np.linalg.norm(Phix - y)

    if verbose:
        print("Initialization - Primal dual algorithm")
        print(f"l2 norm           = {l2norm}")
        print(f"     vs. l2 bound = {param.epsilon}")
        print("----------------------------------")

    # Proximity operator
    def sc(Z: np.ndarray, radius: float):
        return Z * min(1, radius / np.linalg.norm(Z))

    # Main iterations
    l2_norms = []
    condnorm_iterates = []
    for it in range(param.NbIt):

        u_old = u
        x_old = x

        # Dual update
        # data fid
        u_ = np.reshape(u, (-1, 1)) + sigma * Phix
        u = u_ - sigma * (sc(u_ / sigma - y, param.epsilon) + y)

        # Primal update
        x_vect = x_vect - tau * np.matmul(Phi.T, (2 * u - u_old))
        x = np.reshape(x_vect, x.shape)

        if net is not None:
            x = (1 / 255) * x
            x = net(img=x, noise=param.noise, device=device)
            x = 255 * x
            x[x < 0] = 0
            x_vect = np.reshape(x, (-1, 1))

        Phix = np.matmul(Phi, x_vect)
        l2norm = np.linalg.norm(Phix - y)
        condnorm = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-6)

        l2_norms.append(l2norm)
        condnorm_iterates.append(condnorm)

        if it % param.display == 0 and verbose:
            print(f"it = {it}")
            print(f"l2 norm                 = {l2norm}")
            print(f"     vs. l2 bound       = {param.epsilon}")
            print(f"     vs. stop l2 bound  = {(1+param.stopbound) * param.epsilon}")
            print(f"cond norm iterates      = {condnorm}")
            print(f"     stop norm iterates = {param.stopnorm}")
            print("----------------------------------")

        if (
            it > 10
            and l2norm < (1 + param.stopbound) * param.epsilon
            and condnorm < param.stopnorm
        ):
            if verbose:
                print(f"stopping criterion reached, it {it}")
                print(f"l2 norm                = {l2norm}")
                print(f"     vs. l2 bound      = {param.epsilon}")
                print(f"     vs. stop l2 bound = {(1+param.stopbound) * param.epsilon}")
                print(f"cond norm iterates      = {condnorm}")
                print(f"     stop norm iterates = {param.stopnorm}")
                print("----------------------------------")
            break

    save_vector_to_img(
        x=net.postprocess(
            **{"img": x, "new_shape": x_true.shape, "vertical_padding": True}
        ),
        folder=param.output_dir,
        file_name=f"x_index_{index}.npy",
        shape=x_true.shape,
        x_true=x_true,
    )

    logs = {
        "l2_norms": l2_norms,
        "cond_iterates": condnorm_iterates,
        'nb_iter': it,
    }

    if save:
        # Saves logs
        df_logs = pd.DataFrame(logs)
        df_logs.to_csv(
            os.path.join(param.output_dir, f"logs_{index}.csv"),
            index=False)
        if verbose:
            print(f"Outputs correctly saved to {param.output_dir}")

    return x, u, logs

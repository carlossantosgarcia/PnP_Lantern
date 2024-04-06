import os

import numpy as np
import pandas as pd

from utils import Parameters, save_vector_to_img


def SARA_solver(
    y: np.ndarray,
    Phi: np.ndarray,
    param: Parameters,
    x_true: np.ndarray,
    rw_iter: int,
    index: int = 0,
    save: bool = False,
    verbose: bool = True,
):

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

    if param.weights is not None:
        weights = param.weights
    else:
        weights = 1

    def Psi_t(x: np.array):
        return param.Psi_t(np.reshape(x, (param.Ny, param.Nx)))

    def Psi(x: np.array):
        return np.reshape(param.Psi(x), (param.Nx * param.Ny, -1))

    sigma = 0.99 / np.sqrt(param.normPsi + param.normPhi)
    tau = 0.99 / (sigma * (param.normPsi + param.normPhi))

    assert (
        sigma * tau * (param.normPsi + param.normPhi) < 1
    ), "Error in the choice of step-sizes !"

    x = param.x0.reshape((-1, 1))

    if param.v is not None:
        v = param.v
        if len(v.shape) != 2:
            u = v.reshape((-1, 1))
    else:
        v = 0 * Psi_t(x)

    if param.u is not None:
        u = param.u
        if len(u.shape) != 2:
            u = u.reshape((-1, 1))
    else:
        u = 0 * y

    Phix = np.matmul(Phi, x)
    Psi_tx = Psi_t(x)
    l2norm = np.linalg.norm(Phix - y, ord=2)
    l1norm = np.linalg.norm(
        weights * Psi_t(x), ord=1
    )

    if verbose:
        print("Initialization - Primal dual algorithm")
        print(f"l2 norm           = {l2norm}")
        print(f"     vs. l2 bound = {param.epsilon}")
        print(f"l1 norm           = {l1norm}")
        print("----------------------------------")

    # Proximity operator
    def sc(Z: np.ndarray, radius: float):
        return Z * min(1, radius / np.linalg.norm(Z, ord=2))

    def soft(z: np.ndarray, T: float):
        return np.sign(z) * np.maximum(np.abs(z) - T, 0)

    # Main iterations
    l2_norms, l1_norms = [], []
    condl1_norms, condnorm_iterates = [], []
    for it in range(param.NbIt):
        v_old = v
        u_old = u
        x_old = x

        # Dual updates
        # regularization
        v_ = v + sigma * Psi_tx
        v = v_ - sigma * soft(v_ / sigma, param.lambda_ * sigma ** (-1) * weights)

        # data fid
        u_ = np.reshape(u, (-1, 1)) + sigma * Phix
        u = u_ - sigma * (sc(u_ / sigma - y, param.epsilon) + y)

        # Primal update
        x = x - tau * (Psi(2 * v - v_old) + np.matmul(Phi.T, (2 * u - u_old)))
        x[x < 0] = 0

        Phix = np.matmul(Phi, x)
        Psi_tx = Psi_t(x)
        l2norm = np.linalg.norm(Phix - y, ord=2)
        l1norm = np.linalg.norm(weights * Psi_t(x), ord=1)
        l1norm_old = l1norm

        l2_norms.append(l2norm)
        l1_norms.append(l1norm)

        condl1 = abs(l1norm - l1norm_old) / l1norm
        condnorm = np.linalg.norm(x - x_old, ord=2) / np.linalg.norm(x, ord=2)

        condl1_norms.append(condl1)
        condnorm_iterates.append(condnorm)

        if it % param.display == 0 and verbose:
            print(f"it = {it}")
            print(f"l2 norm                 = {l2norm}")
            print(f"     vs. l2 bound       = {param.epsilon}")
            print(f"     vs. stop l2 bound  = {(1+param.stopbound) * param.epsilon}")
            print(f"l1 norm                 = {l1norm}")
            print(f"cond l1 norm            = {condl1}")
            print(f"     stop l1 norm       = {param.stopcrit}")
            print(f"cond norm iterates      = {condnorm}")
            print(f"     stop norm iterates = {param.stopnorm}")
            print("----------------------------------")

        if (
            it > 10
            and condl1 < param.stopcrit
            and l2norm < (1 + param.stopbound) * param.epsilon
            and condnorm < param.stopnorm
        ):
            if verbose:
                print(f"stopping criterion reached, it {it}")
                print(f"l2 norm                = {l2norm}")
                print(f"     vs. l2 bound      = {param.epsilon}")
                print(f"     vs. stop l2 bound = {(1+param.stopbound) * param.epsilon}")
                print(f"l1 norm                 = {l1norm}")
                print(f"cond l1 norm            = {condl1}")
                print(f"     stop l1 norm       = {param.stopcrit}")
                print(f"cond norm iterates      = {condnorm}")
                print(f"     stop norm iterates = {param.stopnorm}")
                print("----------------------------------")
            break

    if save:
        save_vector_to_img(
            x=x,
            folder=param.output_dir,
            file_name=f"x_index_{index}_iter_{rw_iter}.npy",
            shape=(param.Nx, param.Ny),
            x_true=x_true,
        )

    logs = {
        "l1_norms": l1_norms,
        "l2_norms": l2_norms,
        "cond_l1": condl1_norms,
        "cond_iterates": condnorm_iterates,
        "rw_iter": [rw_iter] * len(l1_norms),
        'nb_iter': it,
    }

    # Saves logs
    df_logs = pd.DataFrame(logs)
    df_logs.to_csv(os.path.join(param.output_dir, "logs.csv"), index=False)
    if verbose:
        print(f"Outputs correctly saved to {param.output_dir}")
    return x, u, v, logs


def reweighting(
        x,
        ydata,
        Phi,
        param: Parameters,
        x_true,
        df_logs,
        index,
        save,
        verbose=True):
    NbRW = 3
    sigma = 0.1
    Xrec = [x]

    for iterW in range(1, NbRW + 1):
        if verbose:
            print("****************************")
            print(f"Re-weighting iteration {iterW}")
        temp = param.Psi_t(np.reshape(Xrec[-1], (param.Ny, param.Nx)))
        delta = np.std(temp)
        sigma_n = sigma * np.sqrt(np.size(ydata) / (param.Ny * param.Nx))
        delta = max(sigma_n / 10, delta)

        # Weights
        weights = np.abs(temp)
        weights = delta / (delta + weights)
        param.weights = weights

        param.x0 = Xrec[-1]
        Xtemp, param.u, param.v, logs = SARA_solver(
            y=ydata,
            Phi=Phi,
            param=param,
            x_true=x_true,
            rw_iter=iterW,
            index=index,
            save=save,
            verbose=verbose,
        )
        Xrec.append(Xtemp)

        # Concatenates logs DataFrames
        df_logs = pd.concat([df_logs, pd.DataFrame(logs)])

    x_final = np.reshape(Xrec[-1], (param.Nx, param.Ny))
    return x_final, df_logs

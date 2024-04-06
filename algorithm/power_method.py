import numpy as np


def power_method(
    A: np.ndarray,
    At: np.ndarray,
    im_size: tuple,
    max_iter: int = 200,
    eps: float = 1e-6,
) -> float:
    """Performs power method to compute largest eigenvalue of the compound operator AtA

    Args:
        A (np.ndarray): Matrix to use
        At (np.ndarray): Transposed matrix to use
        im_size (tuple[int]): Size of the image
        max_iter (int): Maximum number of iterations
        eps (float): Epsilon value to end the search

    Returns:
        float: Largest eigenvalue of the compound operator AtA
    """
    # Initial guess for the eigenvector
    x0 = np.random.randn(*im_size)
    x0 = x0 / np.linalg.norm(x0, ord=2)

    # Constants
    p, p_new = 1, 1 + 1e-5
    n = 0

    # Relative improvement of the eigenvalue estimate
    cond = float("inf")

    while cond > eps and n < max_iter:
        # New estimate of x, update of p
        x_new = np.matmul(At, np.matmul(A, x0))
        p = p_new

        # New estimate of largest eigenvalue
        p_new = np.linalg.norm(x_new, ord=2) / np.linalg.norm(x0, ord=2)

        # Relative improvement
        cond = abs(p_new - p) / p_new

        x0 = x_new
        n += 1

    return p_new

"""Numerical solver for the 2D diffusion equation."""



import numpy as np

from .operators import compute_laplace_2d_term
from .boundary_conditions import apply_laplace_boundary_2d

from ..config import Laplace2DConfig
from ..setup.grids import compute_dx, compute_dy



def solve_laplace_2d(
    y_array: np.ndarray,
    initial_condition: np.ndarray,
    config: Laplace2DConfig,
) -> np.ndarray:
    """Solve the 2D Laplace equation with an explicit central finite-difference scheme."""

    dx = compute_dx(config)
    dy = compute_dy(config)

    l1norm = 1

    p = initial_condition
    pn = np.empty_like(p)

    history = []

    while l1norm > config.l1_norm_target:

        pn = p.copy()

        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2]) +
                         dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1])) /
                        (2 * (dx**2 + dy**2)))

        p[:, 0] = 0  # p = 0 @ x = 0
        p[:, -1] = y_array  # p = y @ x = 2
        p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
        p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1

        l1norm = (np.sum(np.abs(p[:]) - np.abs(pn[:])) /
                np.sum(np.abs(pn[:])))
        
        history.append(p.copy())
    
    history_array = np.stack(history, axis=0)

    return history_array
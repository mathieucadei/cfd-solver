"""Numerical solver for the 2D diffusion equation."""



import numpy as np

from .operators import compute_laplace_2d_term
from .boundary_conditions import apply_poisson_boundary_2d

from ..config import Laplace2DConfig
from ..setup.grids import compute_dx, compute_dy



def solve_poisson_2d(
    initial_condition: np.ndarray,
    config: Laplace2DConfig,
) -> np.ndarray:
    """Solve the 2D Laplace equation with an explicit central finite-difference scheme."""

    dx = compute_dx(config)
    dy = compute_dy(config)

    p = initial_condition[0]
    b = initial_condition[1]
    pn = np.empty_like(p)

    history = np.zeros((config.max_iterations + 1, config.num_grid_points_y, config.num_grid_points_x))

    history[0] = initial_condition[1]

    for n in range(1, config.max_iterations + 1):

        pn = p.copy()

        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2]) + 
                          dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1])) - 
                          b[1:-1, 1:-1] * dx**2 * dy**2) / \
                        (2 * (dx**2 + dy**2))

        apply_poisson_boundary_2d(p)
        
        history[n] = p
    
    return history

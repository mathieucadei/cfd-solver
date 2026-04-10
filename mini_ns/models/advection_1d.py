import numpy as np
from ..config import Advection1DConfig
from ..grids import compute_1d_grid_points_spacing


def solve_advection_1d(
    u0: np.ndarray,
    config: Advection1DConfig,
) -> np.ndarray:
    """Solves the 1D linear advection equation using the provided initial condition and configuration."""

    dx = compute_1d_grid_points_spacing(config)

    u = u0.copy()
    history = np.zeros((config.max_iterations + 1, config.num_grid_points))

    history[0] = u0

    for n in range(1, config.max_iterations + 1):

        un = u.copy()
        u[1:] = un[1:] - config.wavespeed * config.time_step / dx * (un[1:] - un[:-1])
        history[n] = u
    
    return history
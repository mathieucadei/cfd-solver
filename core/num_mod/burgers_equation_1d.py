import numpy as np
from ..config import BurgersEquation1DConfig
from ..grids import compute_1d_grid_points_spacing
from ..time_step import compute_time_step


def solve_burgers_equation_1d(
    u0: np.ndarray,
    config: BurgersEquation1DConfig,
) -> np.ndarray:
    """Solves the numerical 1D Burgers' equation using the provided initial condition and configuration."""

    dx = compute_1d_grid_points_spacing(config)
    dt = compute_time_step(dx, config)

    u = u0.copy()
    history = np.zeros((config.max_iterations + 1, config.num_grid_points))

    history[0] = u0

    for n in range(1, config.max_iterations + 1):

        un = u.copy()
        u[1:] = un[1:] - un[1:] * config.time_step / dx * (un[1:] - un[:-1]) 
        + config.viscosity * dt / dx**2 * (un[2:]- 2 * un[1:-1] + un[:-2])

        u[0] = un[0] - un[0] * config.time_step / dx * (un[0] - un[-2])  
        + config.viscosity * dt / dx**2 * (un[1] - 2 * un[0] + un[-1])

        u[-1] = un[0]

        history[n] = u
    
    return history
import numpy as np
from ..config import Diffusion1DConfig
from ..setup.grids import compute_dx
from ..setup.time_stepping import compute_dt


def solve_diffusion_1d(
    initial_condition: np.ndarray,
    config: Diffusion1DConfig,
) -> np.ndarray:
    """Solves the numerical 1D diffusion equation using the provided initial condition and configuration."""

    dx = compute_dx(config)
    dt = compute_dt(config)

    u = initial_condition.copy()
    history = np.zeros((config.max_iterations + 1, config.num_grid_points))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()
        u[1:-1] = un[1:-1] + config.viscosity * dt / dx**2 * (un[2:]- 2 * un[1:-1] + un[:-2])

        u[0] = un[0] + config.viscosity * dt / dx**2 * (un[1] - 2 * un[0] + un[-1])
        u[-1] = un[-1] + config.viscosity * dt / dx**2 * (un[0] - 2 * un[-1] + un[-2])

        history[n] = u
    
    return history
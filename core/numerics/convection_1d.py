import numpy as np

from .operators import compute_convection_1d_term

from ..config import Convection1DConfig
from ..setup.grids import compute_dx
from ..setup.time_stepping import compute_convective_dt


def solve_convection_1d(
    initial_condition: np.ndarray,
    config: Convection1DConfig,
) -> np.ndarray:
    """Solves the 1D convection equation using the provided initial condition and configuration."""

    dx = compute_dx(config)
    dt = compute_convective_dt(config)

    u = initial_condition.copy()

    history = np.zeros((config.max_iterations + 1, config.num_grid_points))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        convection_term = compute_convection_1d_term(un, dx, dt)

        u[1:] = un[1:] - convection_term[1:]
        
        history[n] = u
    
    return history
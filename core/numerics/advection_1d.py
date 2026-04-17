import numpy as np

from .operators import compute_advection_1d_term

from ..config import Advection1DConfig
from ..setup.grids import compute_dx
from ..setup.time_stepping import compute_convective_dt


def solve_advection_1d(
    initial_condition: np.ndarray,
    config: Advection1DConfig,
) -> np.ndarray:
    """Solves the 1D linear advection equation using the provided initial condition and configuration."""

    dx = compute_dx(config)
    dt = compute_convective_dt(config)

    u = initial_condition.copy()
    
    history = np.zeros((config.max_iterations + 1, config.num_grid_points))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        advection_term = compute_advection_1d_term(un, config.wavespeed, dx, dt)
        
        u[1:] = un[1:] - advection_term[1:]
        
        history[n] = u
    
    return history
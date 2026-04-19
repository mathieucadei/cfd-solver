"""Numerical solver for the 1D diffusion equation."""



import numpy as np

from .operators import compute_diffusion_1d_term
from .boundary_conditions import apply_diffusion_boundary_1d

from ..config import Diffusion1DConfig
from ..setup.grids import compute_dx
from ..setup.time_stepping import compute_diffusive_dt_1d


def solve_diffusion_1d(
    initial_condition: np.ndarray,
    config: Diffusion1DConfig,
) -> np.ndarray:
    """Solve the 1D diffusion equation with an explicit central finite-difference scheme."""

    dx = compute_dx(config)
    dt = compute_diffusive_dt_1d(config)

    u = initial_condition.copy()

    history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        diffusion_term = compute_diffusion_1d_term(un, dx, dt, config.viscosity)
        
        u[1:-1] = un[1:-1] + diffusion_term[1:-1]

        apply_diffusion_boundary_1d(
            u=u,
            un=un,
            dt=dt,
            dx=dx,
            nu=config.viscosity,
        )

        history[n] = u
    
    return history
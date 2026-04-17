import numpy as np

from .operators import compute_diffusion_1d_term
from .boundary_conditions import apply_periodic_diffusion_boundary_1d

from ..config import Diffusion1DConfig
from ..setup.grids import compute_dx
from ..setup.time_stepping import compute_diffusive_dt


def solve_diffusion_1d(
    initial_condition: np.ndarray,
    config: Diffusion1DConfig,
) -> np.ndarray:
    """Solves the numerical 1D diffusion equation using the provided initial condition and configuration."""

    dx = compute_dx(config)
    dt = compute_diffusive_dt(config)

    u = initial_condition.copy()

    history = np.zeros((config.max_iterations + 1, config.num_grid_points))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        diffusion_term = compute_diffusion_1d_term(un, dx, dt, config.viscosity)
        
        u[1:-1] = un[1:-1] + diffusion_term[1:-1]

        apply_periodic_diffusion_boundary_1d(
            u=u,
            un=un,
            dt=dt,
            dx=dx,
            nu=config.viscosity,
        )

        history[n] = u
    
    return history
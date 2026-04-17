import numpy as np

from .operators import compute_convection_1d_term, compute_diffusion_1d_term

from ..config import BurgersEquation1DConfig
from ..setup.grids import compute_dx, compute_cole_hopf_dx
from ..setup.time_stepping import compute_diffusive_dt, compute_cole_hopf_dt


def solve_burgers_equation_1d(
    initial_condition: np.ndarray,
    config: BurgersEquation1DConfig,
) -> np.ndarray:
    """Solves the numerical 1D Burgers' equation using the provided initial condition and configuration."""

    if config.grid_type == "hat":
        dx = compute_dx(config)
        dt = compute_diffusive_dt(config)
    
    elif config.grid_type == "cole_hopf":

        dx = compute_cole_hopf_dx(config)
        dt = compute_cole_hopf_dt(config)
    
    else:
        raise ValueError("grid_type must be 'hat' or 'cole_hopf'")

    u = initial_condition.copy()

    history = np.zeros((config.max_iterations + 1, config.num_grid_points))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        convection_term = compute_convection_1d_term(un, dx, dt)
        diffusion_term = compute_diffusion_1d_term(un, dx, dt, config.viscosity)

        u[1:-1] = un[1:-1] - convection_term[1:-1] \
            + diffusion_term[1:-1]
        
        u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) \
            + config.viscosity * dt / dx**2 * (un[1] - 2 * un[0] + un[-2])
        u[-1] = un[0]
        
        history[n] = u
    
    return history
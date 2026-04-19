"""Numerical solver for the 2D diffusion equation."""



import numpy as np

from .operators import compute_diffusion_2d_term
from .boundary_conditions import apply_periodic_diffusion_boundary_2d

from ..config import Diffusion2DConfig
from ..setup.grids import compute_dx, compute_dy
from ..setup.time_stepping import compute_diffusive_dt_2d



def solve_diffusion_2d(
    initial_condition: np.ndarray,
    config: Diffusion2DConfig,
) -> np.ndarray:
    """Solve the 2D diffusion equation with an explicit upwind finite-difference scheme."""

    dx = compute_dx(config)
    dy = compute_dy(config)
    dt = compute_diffusive_dt_2d(config)

    u = initial_condition.copy()
    
    history = np.zeros((config.max_iterations + 1, config.num_grid_points_y, config.num_grid_points_x))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        diffusion_term = compute_diffusion_2d_term(un, dx, dy, dt, config.viscosity)

        u[1:-1, 1:-1] = un[1:-1, 1:-1] + diffusion_term[1:-1, 1:-1]

        apply_periodic_diffusion_boundary_2d(u, config.u_min)

        history[n] = u
    
    return history
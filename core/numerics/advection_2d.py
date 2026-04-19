"""Numerical solver for the 2D linear advection equation."""



import numpy as np

from .operators import compute_advection_2d_term
from .boundary_conditions import apply_periodic_advection_boundary_2d

from ..config import Advection2DConfig
from ..setup.grids import compute_dx, compute_dy
from ..setup.time_stepping import compute_advective_dt_2d



def solve_advection_2d(
    initial_condition: np.ndarray,
    config: Advection2DConfig,
) -> np.ndarray:
    """Solve the 2D linear advection equation with an explicit upwind finite-difference scheme."""

    dx = compute_dx(config)
    dy = compute_dy(config)
    dt = compute_advective_dt_2d(config)

    u = initial_condition.copy()
    
    history = np.zeros((config.max_iterations + 1, config.num_grid_points_x, config.num_grid_points_y))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        advection_term = compute_advection_2d_term(un, config.wavespeed, dx, dy, dt)

        u[1:, 1:] = un[1:, 1:] - advection_term[1:, 1:]

        apply_periodic_advection_boundary_2d(u, config.u_min)

        history[n] = u
    
    return history
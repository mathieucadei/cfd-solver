"""Numerical solver for the 2D Burgers' equation."""



import numpy as np

from .operators import compute_convection_2d_term, compute_diffusion_2d_term
from .boundary_conditions import apply_periodic_convection_boundary_2d

from ..config import Advection2DConfig
from ..setup.grids import compute_dx_2d, compute_dy_2d
from ..setup.time_stepping import compute_diffusive_dt_2d



def solve_burgers_equation_2d(
    initial_condition: np.ndarray,
    config: Advection2DConfig,
) -> np.ndarray:
    """Solve the 2D Burgers' equation with an explicit finite-difference scheme."""

    dx = compute_dx_2d(config)
    dy = compute_dy_2d(config)
    dt = compute_diffusive_dt_2d(config)

    u = initial_condition[0].copy()
    v = initial_condition[1].copy()
    
    u_history = np.zeros((config.max_iterations + 1, config.num_grid_points_x, config.num_grid_points_y))
    v_history = np.zeros((config.max_iterations + 1, config.num_grid_points_x, config.num_grid_points_y))

    u_history[0] = initial_condition[0]
    v_history[0] = initial_condition[1]

    for n in range(1, config.max_iterations + 1):

        un = u.copy()
        vn = v.copy()

        convection_u_term, convection_v_term = compute_convection_2d_term(un, vn, dx, dy, dt)
        diffusion_u_term = compute_diffusion_2d_term(un, dx, dy, dt, config.viscosity)
        diffusion_v_term = compute_diffusion_2d_term(vn, dx, dy, dt, config.viscosity)

        u[1:-1, 1:-1] = un[1:-1, 1:-1] - convection_u_term[1:-1, 1:-1] + diffusion_u_term[1:-1, 1:-1]
        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - convection_v_term[1:-1, 1:-1] + diffusion_v_term[1:-1, 1:-1]

        apply_periodic_convection_boundary_2d(u, v, config.u_min, config.v_min)

        u_history[n] = u
        v_history[n] = v
    
    return u_history, v_history
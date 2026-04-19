"""Numerical solver for the 1D Burgers' equation."""



import numpy as np

from .operators import compute_convection_1d_term, compute_diffusion_1d_term
from .boundary_conditions import apply_periodic_burgers_boundary_1d

from ..config import BurgersEquation1DConfig
from ..setup.grids import compute_cole_hopf_dx, compute_dx
from ..setup.time_stepping import compute_cole_hopf_dt, compute_diffusive_dt_1d



def solve_burgers_equation_1d(
    initial_condition: np.ndarray,
    config: BurgersEquation1DConfig,
) -> np.ndarray:
    """Solve the 1D Burgers' equation with an explicit finite-difference scheme."""

    if config.grid_type == "hat":
        dx = compute_dx(config)
        dt = compute_diffusive_dt_1d(config)
    
    elif config.grid_type == "cole_hopf":

        dx = compute_cole_hopf_dx(config)
        dt = compute_cole_hopf_dt(config)
    
    else:
        raise ValueError("grid_type must be 'hat' or 'cole_hopf'")

    u = initial_condition.copy()

    history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        convection_term = compute_convection_1d_term(un, dx, dt)
        diffusion_term = compute_diffusion_1d_term(un, dx, dt, config.viscosity)

        u[1:-1] = un[1:-1] - convection_term[1:-1] \
            + diffusion_term[1:-1]
        
        apply_periodic_burgers_boundary_1d(
            u=u,
            un=un,
            dt=dt,
            dx=dx,
            nu=config.viscosity,
        )
        
        history[n] = u
    
    return history
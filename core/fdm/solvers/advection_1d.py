"""Numerical solver for the 1D advection equation."""



import numpy as np

from ..operators import compute_advection_1d_term

from ..config import Advection1DConfig
from ..grids import compute_dx
from ..time_stepping import compute_advective_dt_1d



def solve_advection_1d(
    initial_condition: np.ndarray,
    config: Advection1DConfig,
) -> np.ndarray:
    """Solve the 1D advection equation with an explicit upwind finite-difference scheme."""

    dx = compute_dx(config)
    dt = compute_advective_dt_1d(config)

    if config.scheme == 'upwind':

        u = initial_condition.copy()
    
        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            advection_term = compute_advection_1d_term(un, config.wavespeed, dx, dt, config.scheme)

            u[1:] = un[1:] - advection_term[1:]

            history[n] = u
        
    elif config.scheme == 'leapfrog':

        uo = initial_condition.copy()
    
        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        init_advection_term = compute_advection_1d_term(uo, config.wavespeed, dx, dt, 'upwind')

        un = uo.copy()

        un[1:] = uo[1:] - init_advection_term[1:]

        history[0] = uo
        history[1] = un

        for n in range(1, config.max_iterations):

            u = un.copy()

            advection_term = compute_advection_1d_term(un, config.wavespeed, dx, dt, config.scheme)

            u[1:-1] = uo[1:-1] - advection_term[1:-1]

            history[n+1] = u

            uo = un
            un = u
    
    elif config.scheme == 'lax-friedrichs':

        u = initial_condition.copy()
    
        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            advection_term = compute_advection_1d_term(un, config.wavespeed, dx, dt, config.scheme)

            u[1:-1] = (un[2:] + un[:-2]) / 2  - advection_term[1:-1]

            history[n] = u

    elif config.scheme == 'lax-wendroff':

        u = initial_condition.copy()
    
        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            advection_term = compute_advection_1d_term(un, config.wavespeed, dx, dt, config.scheme)

            u[1:-1] = un[1:-1] - advection_term[1:-1]

            history[n] = u
    
    else:
        
        raise ValueError("basis must be 'upwind', 'leapfrog', 'lax-friedrichs', or 'lax-wendroff'")

    return history
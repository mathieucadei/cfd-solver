"""Numerical solver for the 1D convection equation."""


import numpy as np

from .operators import compute_convection_1d_term

from ..config import Convection1DConfig
from ..setup.grids import compute_dx
from ..setup.time_stepping import compute_convective_dt_1d



def solve_convection_1d(
    initial_condition: np.ndarray,
    config: Convection1DConfig,
) -> np.ndarray:
    """Solve the 1D convection equation with an explicit upwind finite-difference scheme."""

    dx = compute_dx(config)
    dt = compute_convective_dt_1d(config)

    if config.scheme == 'upwind':

        u = initial_condition.copy()

        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            convection_term = compute_convection_1d_term(un, dx, dt)

            u[1:] = un[1:] - convection_term[1:]
            
            history[n] = u

    
    elif config.scheme == 'conservative-lax-friedrichs':

        u = initial_condition.copy()
    
        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            e = un**2 / 2

            convection_term = compute_convection_1d_term(e, dx, dt, config.scheme)

            u[1:-1] = (un[2:] + un[:-2]) / 2  - convection_term[1:-1]

            history[n] = u
    
    elif config.scheme == 'conservative-richtmyer':

        un_half = initial_condition.copy()

        u = initial_condition.copy()
    
        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            un_half = un.copy()

            e = un**2 / 2

            convection_term_1 = compute_convection_1d_term(e, dx, dt, 'conservative-lax-friedrichs')

            un_half[1:-1] = (un[2:] + un[:-2]) / 2  - convection_term_1[1:-1]

            u = un_half.copy()

            e = un_half**2 / 2

            convection_term_2 = compute_convection_1d_term(e, dx, dt, 'conservative-leapfrog')
            
            u[1:-1] = un[1:-1] - convection_term_2[1:-1]

            history[n] = u

    elif config.scheme == 'conservative-lax-wendroff':

        un_half = initial_condition.copy()

        u = initial_condition.copy()
    
        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            un_half = un.copy()

            e = un**2 / 2

            convection_term_1 = compute_convection_1d_term(e, dx, dt, 'conservative-lax-friedrichs-lw')

            un_half = (un[1:] + un[:-1]) / 2  - convection_term_1[1:]

            e =  un_half**2 / 2

            convection_term_2 = compute_convection_1d_term(e, dx, dt, 'conservative-leapfrog-lw')
            
            u[1:-1] = un[1:-1] - convection_term_2[1:]

            history[n] = u       

    return history
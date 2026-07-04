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
    
    elif config.scheme == 'conservative-upwind':

        u = initial_condition.copy()

        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            e = un**2 / 2

            convection_term = compute_convection_1d_term(e, dx, dt, config.scheme)

            u[1:] = un[1:] - convection_term[1:]
            
            history[n] = u
    

    elif config.scheme == 'lax-friedrichs':

        u = initial_condition.copy()
    
        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            convection_term = compute_convection_1d_term(un, dx, dt, config.scheme)

            u[1:-1] = (un[2:] + un[:-2]) / 2  - convection_term[1:-1]

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


    elif config.scheme == '1-step-lax-wendroff':

        u = initial_condition.copy()
    
        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            u[1:-1] =  un[1:-1] - \
                        (un[1:-1] * dt/(dx)) / 2 * (un[2:] + un[:-2]) + \
                        (un[1:-1] * dt/(dx))**2 / 2 * (un[2:] - 2 * un[1:-1] + un[:-2])

            history[n] = u  
    

    elif config.scheme == '1-step-conservative-lax-wendroff':

        u = initial_condition.copy()
    
        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            e = un**2 / 2

            u[1:-1] =  un[1:-1] - \
                        dt/(2*dx) * (e[2:] - e[:-2]) + \
                        (dt/(2*dx))**2 * ((un[2:] + un[1:-1]) * (e[2:] - e[1:-1]) - \
                                           (un[1:-1] + un[:-2]) * (e[1:-1] - e[:-2]))

            history[n] = u   
    

    elif config.scheme == 'richtmyer':

        un_half = initial_condition.copy()

        u = initial_condition.copy()
    
        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            un_half = un.copy()

            convection_term_1 = compute_convection_1d_term(un, dx, dt, 'lax-friedrichs-half')

            un_half[1:-1] = (un[2:] + un[:-2]) / 2  - convection_term_1[1:-1]

            convection_term_2 = compute_convection_1d_term(un_half, dx, dt, 'leapfrog')
            
            u[1:-1] = un[1:-1] - convection_term_2[1:-1]

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

            convection_term_1 = compute_convection_1d_term(e, dx, dt, 'conservative-lax-friedrichs-half')

            un_half[1:-1] = (un[2:] + un[:-2]) / 2  - convection_term_1[1:-1]

            e = un_half**2 / 2

            convection_term_2 = compute_convection_1d_term(e, dx, dt, 'conservative-leapfrog')
            
            u[1:-1] = un[1:-1] - convection_term_2[1:-1]

            history[n] = u
    

    elif config.scheme == '2-step-lax-wendroff':

        un_half = initial_condition.copy()

        u = initial_condition.copy()
    
        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            un_half = un.copy()

            convection_term_1 = compute_convection_1d_term(un, dx, dt, 'lax-friedrichs-lw')

            un_half = (un[1:] + un[:-1]) / 2  - convection_term_1[1:]

            convection_term_2 = compute_convection_1d_term(un_half, dx, dt, 'leapfrog-lw')
            
            u[1:-1] = un[1:-1] - convection_term_2[1:]

            history[n] = u


    elif config.scheme == '2-step-conservative-lax-wendroff':

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
    

    elif config.scheme == 'mac-cormack':

        un_star = initial_condition.copy()

        u = initial_condition.copy()
    
        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            un_star = un.copy()

            convection_term_1 = compute_convection_1d_term(un, dx, dt, 'downwind')

            un_star[1:-1] = un[1:-1]  - convection_term_1[1:-1]

            convection_term_2 = compute_convection_1d_term(un_star, dx, dt, 'upwind')
            
            u[1:-1] = (un[1:-1] + un_star[1:-1] - convection_term_2[1:-1]) / 2

            history[n] = u

    
    elif config.scheme == 'conservative-mac-cormack':

        un_star = initial_condition.copy()

        u = initial_condition.copy()
    
        history = np.zeros((config.max_iterations + 1, config.num_grid_points_x))

        history[0] = initial_condition

        for n in range(1, config.max_iterations + 1):

            un = u.copy()

            un_star = un.copy()

            e = un**2 / 2

            convection_term_1 = compute_convection_1d_term(e, dx, dt, 'conservative-downwind')

            un_star[1:-1] = un[1:-1]  - convection_term_1[1:-1]

            e = un_star**2 / 2

            convection_term_2 = compute_convection_1d_term(e, dx, dt, 'conservative-upwind')
            
            u[1:-1] = (un[1:-1] + un_star[1:-1] - convection_term_2[1:-1]) / 2

            history[n] = u


    else:
        
        raise ValueError(
        "basis must be " \
        "'upwind', " \
        "'conservative-upwind, " \
        "'lax-friedrichs', " \
        "'conservative-lax-friedrichs', " \
        "'richtmyer', " \
        "'conservative-richtmyer', " \
        "'2-step-lax-wendroff'," \
        "'2-step-conservative-lax-wendroff'," \
        "'mac-cormack' or" \
        "'conservative-mac-cormack'"
        )       

    return history
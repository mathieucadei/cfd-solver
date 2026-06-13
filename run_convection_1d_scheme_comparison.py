"""Run the 1D convection solver and generate solution plots."""



from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from core import (
    Convection1DConfig,
    heaviside_initial_condition_1d,
    make_x_grid,
    solve_convection_1d,
)
from post_processing import (
    show_solution_1d_animation,
    show_solution_contour_map,
    show_solution_overview,
    show_solution_surface,
    show_solution_traces,
)



# Pre-processing
# Simulation parameters

domain_length_x = 4.0
num_grid_points_x = 41
max_iterations = 40
sigmas = [1.0, 0.5]
hat_start = 2
hat_end = 0
u_min = 0
u_max = 1.0
schemes = [
    'upwind',
    'conservative-upwind',
    'lax-friedrichs',
    'conservative-lax-friedrichs',
    'richtmyer',  
    'conservative-richtmyer',
    'lax-wendroff',
    'conservative-lax-wendroff'
]

scheme_colors = {
    'upwind': 'tab:blue',
    'lax-friedrichs': 'tab:orange',
    'richtmyer': 'tab:green',
    'lax-wendroff': 'tab:red',
}


# Visualization parameters

step_stride = 20
case_name = '1d convection'
title = True
save = False
show_individual_plots = False


# Create the configuration object
for sigma in sigmas: 

    plt.figure()

    for scheme in schemes:

        case_name_scheme = f'{case_name} - {scheme}'

        convection_1d_config = Convection1DConfig(
            domain_length_x=domain_length_x,
            num_grid_points_x=num_grid_points_x,
            max_iterations=max_iterations,
            sigma=sigma,
            hat_start=hat_start,
            hat_end=hat_end,
            u_min=u_min,
            u_max=u_max,
            scheme=scheme,
        )


        # Generate the grid and time array

        x_array = make_x_grid(convection_1d_config)
        time_array = np.arange(0, convection_1d_config.max_iterations + 1)

        # Initialize the initial condition

        initial_condition = heaviside_initial_condition_1d(x_array, convection_1d_config)



        # Solve the convection equation

        solution_history = solve_convection_1d(initial_condition, convection_1d_config)

        base_scheme = scheme.replace('conservative-', '')


        # Post-processing

        plt.plot(
            x_array,
            solution_history[-1],
            color=scheme_colors[base_scheme],
            linestyle='--' if 'conservative' in scheme else '-',
            label=scheme,
        )
        
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Heaviside Solution @ Final Time Step = {max_iterations}')

    plt.legend()
    plt.show()

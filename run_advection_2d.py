"""Run the 1D advection solver and generate solution plots."""



import numpy as np

from core import (
    Advection2DConfig,
    hat_initial_condition_2d,
    make_1d_grid,
    solve_advection_2d,
)
from post_processing import (
    show_solution_1d_animation,
    show_solution_contour,
    show_solution_overview,
    show_solution_surface,
    show_solution_traces,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2.0
domain_length_y: float = 2.0
num_grid_points_x: int = 81
num_grid_points_y: int = 81
max_iterations: int = 100
sigma: float = 0.2
wavespeed: float = 1.0
hat_start_x: float = 0.5
hat_start_y: float = 0.5
hat_end_x: float = 1.0
hat_end_y: float = 1.0
u_min: float = 1.0
u_max: float = 2.0


# Visualization parameters

step_stride = 20
case_name = '2d advection'
title = True
save = False


# Create the configuration object

advection_2d_config = Advection2DConfig(
    domain_length_x=domain_length_x,
    domain_length_y=domain_length_y,
    num_grid_points_x=num_grid_points_x,
    num_grid_points_y=num_grid_points_y,
    max_iterations=max_iterations,
    sigma=sigma,
    wavespeed=wavespeed,
    hat_start_x=hat_start_x,
    hat_start_y=hat_start_y,
    hat_end_x=hat_end_x,
    hat_end_y= hat_end_y,
    u_min=u_min,
    u_max=u_max,
)


# Generate the grid and time array

x_array = np.linspace(0.0, domain_length_x, num_grid_points_x)
y_array = np.linspace(0.0, domain_length_y, num_grid_points_y)
time_array = np.arange(0, advection_2d_config.max_iterations + 1)


# Initialize the initial condition

initial_condition = hat_initial_condition_2d(advection_2d_config)



# Solve the advection equation

history = solve_advection_2d(initial_condition, advection_2d_config)



# Post-processing

# show_solution_traces(
#     x_values=x_array,
#     num_solution_matrix=history,
#     cut_values=time_array,
#     step_stride=step_stride,
#     case_name=case_name,
#     title=title,
#     save=save,
# )

# show_solution_traces(
#     x_values=time_array,
#     num_solution_matrix=history,
#     cut_values=x_array,
#     axis=1,
#     step_stride=step_stride,
#     cut_label='x',
#     case_name=case_name,
#     title=title,
#     save=save,
# )

# show_solution_contour(
#     x_values=x_array,
#     y_values=time_array,
#     solution_matrix=history,
#     case_name=case_name,
#     title=title,
#     save=save,
# )

show_solution_surface(
    x_values=x_array,
    y_values=y_array,
    solution_matrix=history[-1],
    case_name=case_name,
    title=title,
    save=save,
)

# show_solution_overview(
#     x_values=x_array, 
#     y_values=time_array, 
#     num_solution_matrix=history, 
#     step_stride=step_stride,
#     case_name=case_name,
#     title=title,
#     save=save,
# )

# show_solution_1d_animation(
#     x_values=x_array,
#     num_solution_matrix=history,
#     case_name=case_name,
#     save=save,
# )
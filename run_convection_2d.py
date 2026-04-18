"""Run the 2D convection solver and generate solution plots."""



import numpy as np

from core import (
    Convection2DConfig,
    hat_convective_initial_condition_2d,
    make_1d_grid,
    solve_convection_2d,
)
from post_processing import (
    show_solutions_2d_animation,
    show_solution_contour,
    show_solution_overview,
    show_solution_surfaces, 
    show_solution_traces,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2.0
domain_length_y: float = 2.0
num_grid_points_x: int = 81
num_grid_points_y: int = 81
max_iterations: int = 80
sigma: float = 0.2
wavespeed: float = 1.0
hat_start_x: float = 0.5
hat_start_y: float = 0.5
hat_end_x: float = 1.0
hat_end_y: float = 1.0
u_min: float = 1.0
u_max: float = 2.0
v_min: float = 1.0
v_max: float = 2.0


# Visualization parameters

step_stride = 10
case_name = '2d convection'
title = True
save = False


# Create the configuration object

advection_2d_config = Convection2DConfig(
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
    v_min=v_min,
    v_max=v_max,   
)


# Generate the grid and time array

x_array = np.linspace(0.0, domain_length_x, num_grid_points_x)
y_array = np.linspace(0.0, domain_length_y, num_grid_points_y)
time_array = np.arange(0, advection_2d_config.max_iterations + 1)


# Initialize the initial condition

initial_condition = hat_convective_initial_condition_2d(advection_2d_config)



# Solve the advection equation

solution_matrix = solve_convection_2d(initial_condition, advection_2d_config)

u_solution_matrix = solution_matrix[0]

v_solution_matrix = solution_matrix[1]

# solution_final = u_solution_matrix[-1, ...]

# solution_final_x = u_solution_matrix[-1, :, :]

# solution_final_y = solution_final_x.T



# Post-processing

# show_solution_traces(
#     x_values=x_array,
#     cut_values=y_array,
#     num_solution_matrix=solution_final_x,
#     step_stride=step_stride,
#     cut_label='y',
#     case_name=case_name,
#     title=title,
#     save=save,
# )

# show_solution_traces(
#     x_values=y_array,
#     cut_values=x_array,
#     num_solution_matrix=solution_final_y,
#     step_stride=step_stride,
#     cut_label='x',
#     case_name=case_name,
#     title=title,
#     x_label='y',
#     save=save,
# )

# show_solution_contour(
#     x_values=x_array,
#     y_values=y_array,
#     solution_matrix=solution_final,
#     case_name=case_name,
#     title=title,
#     y_label='y',
#     save=save,
# )

# show_solution_surfaces(
#     x_values=x_array,
#     y_values=y_array,
#     u_solution_matrix=u_solution_matrix,
#     v_solution_matrix=v_solution_matrix,
#     case_name=case_name,
#     title=title,
#     y_label='y',
#     save=save,
# )

# show_solution_overview(
#     x_values=x_array, 
#     y_values=y_array, 
#     num_solution_matrix=solution_final,
#     y_label='y',
#     step_stride=step_stride,
#     case_name=case_name,
#     title=title,
#     save=save,
# )

show_solutions_2d_animation(
    x_values=x_array,
    y_values=y_array, 
    u_solution_history=u_solution_matrix,
    v_solution_history=v_solution_matrix,
    case_name=case_name,
    save=save,
)
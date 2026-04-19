"""Run the 2D convection solver and generate solution plots."""



import numpy as np

from core import (
    BurgersEquation2DConfig,
    hat_convective_initial_condition_2d,
    make_x_grid,
    make_y_grid,
    solve_burgers_equation_2d,
)
from post_processing import (
    show_solution_contour,
    show_solution_overview,
    show_solution_uv_2d_animations,
    show_solution_uv_surfaces, 
    show_solution_traces,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2.0
domain_length_y: float = 2.0
num_grid_points_x: int = 41
num_grid_points_y: int = 41
max_iterations: int = 120
sigma: float = 0.0009
viscosity: float = 0.01
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
case_name = '2d burgers'
title = True
save = False


# Create the configuration object

burgers_equation_2d_config = BurgersEquation2DConfig(
    domain_length_x=domain_length_x,
    domain_length_y=domain_length_y,
    num_grid_points_x=num_grid_points_x,
    num_grid_points_y=num_grid_points_y,
    max_iterations=max_iterations,
    sigma=sigma,
    viscosity=viscosity,
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

x_array = make_x_grid(burgers_equation_2d_config)
y_array = make_y_grid(burgers_equation_2d_config)
time_array = np.arange(0, burgers_equation_2d_config.max_iterations + 1)


# Initialize the initial condition

initial_condition = hat_convective_initial_condition_2d(burgers_equation_2d_config)



# Solve the advection equation

solution_matrix = solve_burgers_equation_2d(initial_condition, burgers_equation_2d_config)

u_solution_matrix = solution_matrix[0]

v_solution_matrix = solution_matrix[1]

u_solution_matrix_final = u_solution_matrix[-1, ...]

v_solution_matrix_final = v_solution_matrix[-1, ...]



# Post-processing

show_solution_uv_surfaces(
    x_values=x_array,
    y_values=y_array,
    u_solution_matrix=u_solution_matrix_final,
    v_solution_matrix=v_solution_matrix_final,
    case_name=f'{case_name} final',
    title=title,
    save=save,
)

show_solution_uv_2d_animations(
    x_values=x_array,
    y_values=y_array, 
    u_solution_history=u_solution_matrix,
    v_solution_history=v_solution_matrix,
    case_name=case_name,
    save=save,
)
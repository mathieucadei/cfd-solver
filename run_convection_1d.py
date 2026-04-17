"""Run the 1D convection solver and generate solution plots."""



import numpy as np
from pathlib import Path

from core import (
    Convection1DConfig,
    make_1d_grid,
    hat_initial_condition,
    solve_convection_1d,
)

from post_processing import (
    show_solution_traces,
    show_solution_contour,
    show_solution_surface, 
    show_solution_overview, 
    show_solution_1d_animation,
)



# Pre-processing
# Simulation parameters

domain_length = 2.0
num_grid_points = 101
max_iterations = 100
sigma = 0.2
hat_start = 0.5
hat_end = 1.0
u_min = 1.0
u_max = 2.0


# Visualization parameters

step_stride = 20
equation_name = '1d convection'
title = True
save = False


# Create the configuration object

convection_1d_config = Convection1DConfig(
    domain_length=domain_length,
    num_grid_points=num_grid_points,
    max_iterations=max_iterations,
    sigma=sigma,
    hat_start=hat_start,
    hat_end=hat_end,
    u_min=u_min,
    u_max=u_max,
)


# Generate the grid and time array

x_array = make_1d_grid(convection_1d_config)
time_array = np.arange(0, convection_1d_config.max_iterations + 1)

# Initialize the initial condition

initial_condition = hat_initial_condition(x_array, convection_1d_config)



# Solve the convection equation

history = solve_convection_1d(initial_condition, convection_1d_config)



# Post-processing

show_solution_traces(
    x_values=x_array,
    num_solution_matrix=history,
    cut_values=time_array,
    step_stride=step_stride,
    equation_name=equation_name,
    title=title,
    save=save,
)

show_solution_traces(
    x_values=time_array,
    num_solution_matrix=history,
    cut_values=x_array,
    axis=1,
    step_stride=step_stride,
    cut_label='x',
    equation_name=equation_name,
    title=title,
    save=save,
)

show_solution_contour(
    x_values=x_array,
    y_values=time_array,
    solution_matrix=history,
    equation_name=equation_name,
    title=title,
    save=save,
)

show_solution_surface(
    x_values=x_array,
    y_values=time_array,
    solution_matrix=history,
    equation_name=equation_name,
    title=title,
    save=save,
)

show_solution_overview(
    x_array, 
    time_array, 
    history, 
    step_stride=step_stride,
    equation_name=equation_name,
    title=title,
    save=save,
)

show_solution_1d_animation(
    x_array, 
    history, 
    equation_name=equation_name,
    save=save,
)
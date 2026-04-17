"""Run the 1D diffusion solver and generate solution plots."""



import numpy as np

from core import (
    Diffusion1DConfig,
    hat_initial_condition_1d,
    make_1d_grid,
    solve_diffusion_1d,
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

domain_length = 2.0
num_grid_points = 41
max_iterations = 41
sigma = 0.2
viscosity = 0.3
hat_start = 0.5
hat_end = 1.0
u_min = 1.0
u_max = 2.0


# Visualization parameters

step_stride = 20
case_name = '1d diffusion'
title = True
save = False


# Create the configuration object

diffusion_1d_config = Diffusion1DConfig(
    domain_length=domain_length,
    num_grid_points=num_grid_points,
    max_iterations=max_iterations,
    sigma=sigma,
    viscosity=viscosity,
    hat_start=hat_start,
    hat_end=hat_end,
    u_min=u_min,
    u_max=u_max,
)


# Generate the grid and time array

x_array = make_1d_grid(diffusion_1d_config)
time_array = np.arange(0, diffusion_1d_config.max_iterations + 1)


# Initialize the initial condition

initial_condition = hat_initial_condition_1d(x_array, diffusion_1d_config)



# Solve the diffusion equation

history = solve_diffusion_1d(initial_condition, diffusion_1d_config)



# Post-processing

show_solution_traces(
    x_values=x_array,
    num_solution_matrix=history,
    cut_values=time_array,
    step_stride=step_stride,
    case_name=case_name,
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
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_contour(
    x_values=x_array,
    y_values=time_array,
    solution_matrix=history,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_surface(
    x_values=x_array,
    y_values=time_array,
    solution_matrix=history,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_overview(
    x_values=x_array, 
    y_values=time_array, 
    num_solution_matrix=history, 
    step_stride=step_stride,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_1d_animation(
    x_values=x_array,
    num_solution_matrix=history,
    case_name=case_name,
    save=save,
)
"""Run the 1D advection solver and generate solution plots."""



import numpy as np

from core import (
    Advection1DConfig,
    hat_initial_condition_1d,
    make_x_grid,
    solve_advection_1d,
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

domain_length_x = 2.0
num_grid_points_x = 81
max_iterations = 80
sigma = 1
wavespeed = 1.0
hat_start = 0.5
hat_end = 1.0
u_min = 1.0
u_max = 2.0


# Visualization parameters

step_stride = 20
case_name = '1d advection'
title = True
save = False
show_individual_plots = False


# Create the configuration object

advection_1d_config = Advection1DConfig(
    domain_length_x=domain_length_x,
    num_grid_points_x=num_grid_points_x,
    max_iterations=max_iterations,
    sigma=sigma,
    wavespeed=wavespeed,
    hat_start=hat_start,
    hat_end=hat_end,
    u_min=u_min,
    u_max=u_max,
)


# Generate the grid and time array

x_array = make_x_grid(advection_1d_config)
time_array = np.arange(0, advection_1d_config.max_iterations + 1)


# Initialize the initial condition

initial_condition = hat_initial_condition_1d(x_array, advection_1d_config)



# Solve the advection equation

solution_history = solve_advection_1d(initial_condition, advection_1d_config)



# Post-processing

if show_individual_plots:
    show_solution_traces(
        x_values=x_array,
        cut_values=time_array,
        num_solution_matrix=solution_history,
        step_stride=step_stride,
        case_name=case_name,
        title=title,
        save=save,
    )

    show_solution_traces(
        x_values=time_array,
        cut_values=x_array,
        num_solution_matrix=solution_history,
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
        solution_matrix=solution_history,
        case_name=case_name,
        title=title,
        save=save,
    )

    show_solution_surface(
        x_values=x_array,
        y_values=time_array,
        solution_matrix=solution_history,
        case_name=case_name,
        title=title,
        save=save,
    )

show_solution_overview(
    x_values=x_array, 
    y_values=time_array, 
    num_solution_matrix=solution_history, 
    step_stride=step_stride,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_1d_animation(
    x_values=x_array,
    num_solution_history=solution_history,
    case_name=case_name,
    save=save,
)
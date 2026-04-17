"""Run the 1D diffusion & heat solvers and generate solution & comparision plots."""



import numpy as np

from core import (
    Diffusion1DConfig,
    compute_coefficients,
    compute_diffusive_dt,
    compute_series_terms,
    generate_mode_indices,
    hat_initial_condition_1d,
    make_1d_grid,
    solve_diffusion_1d,
    solve_heat_equation_1d,
)
from post_processing import (
    show_solution_1d_animation,
    show_solution_contour,
    show_solution_overview,
    show_solution_surface,
    show_solution_traces,
)



# Pre-processing
# Numerical simulation parameters

domain_length = 2.0
num_grid_points = 101
max_iterations = 1001
sigma = 0.2
viscosity = 0.3
hat_start = 0.5
hat_end = 1.0
u_min = 1.0
u_max = 2.0


# Analytical simulation parameters

num_modes = 100
basis = "cosine"  # "periodic" or "cosine"


# Visualization parameters

step_stride = 100
case_name = '1d diffusion vs heat'
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

dt = compute_diffusive_dt(diffusion_1d_config)

time_array = np.arange(0, max_iterations + 1) * dt


# Initialize the numerical initial condition

initial_condition = hat_initial_condition_1d(x_array, diffusion_1d_config)


# Fourier-series setup

mode_indices = generate_mode_indices(num_modes)

mode_coefficients = compute_coefficients(
    initial_condition, 
    x_array, 
    mode_indices, 
    basis=basis,
)

series_terms = compute_series_terms(mode_indices, mode_coefficients, x_array, basis=basis)



# Solve
# Numerical diffusion equation

history_num = solve_diffusion_1d(initial_condition, diffusion_1d_config)


# Heat analytical equation

history_ana = solve_heat_equation_1d(
    series_terms, 
    mode_indices,
    x_array,
    time_array, 
    diffusion_1d_config.viscosity,
    basis=basis)



# Post-processing

show_solution_traces(
    x_values=x_array,
    num_solution_matrix=history_num,
    cut_values=time_array,
    ana_solution_matrix=history_ana,
    step_stride=step_stride,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_traces(
    x_values=time_array,
    num_solution_matrix=history_num,
    cut_values=x_array,
    axis=1,
    ana_solution_matrix=history_ana,
    step_stride=step_stride,
    cut_label='x',
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_contour(
    x_values=x_array,
    y_values=time_array,
    solution_matrix=history_num,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_surface(
    x_values=x_array,
    y_values=time_array,
    solution_matrix=history_num,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_overview(
    x_values=x_array, 
    y_values=time_array, 
    num_solution_matrix=history_num,
    ana_solution_matrix=history_ana, 
    step_stride=step_stride,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_1d_animation(
    x_values=x_array,
    num_solution_matrix=history_num,
    ana_solution_matrix=history_ana, 
    case_name=case_name,
    save=save,
)
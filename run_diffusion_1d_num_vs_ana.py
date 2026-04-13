import numpy as np
from pathlib import Path

from core import (
    Diffusion1DConfig,
    make_1d_grid,
    compute_dx,
    hat_initial_condition,
    compute_dt,
    solve_diffusion_1d,
    solve_heat_equation_1d,
    generate_mode_indices,
    compute_coefficients,
    compute_series_terms,
    plot_snapshots,
    plot_animation,
)


# Inputs

## Configuration parameters for the 1D diffusion simulation

domain_length = 2.0
num_grid_points = 101
max_iterations = 1001
sigma = 0.2
viscosity = 0.3
hat_start = 0.5
hat_end = 1.0

u_min = 1.0
u_max = 2.0

## Configuration parameters for the analytical solution

num_modes = 100
basis = "cosine"  # "periodic" or "cosine"

## Visualization parameters

step_stride = 200
save_fig = False


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


# Generate the grid, initial condition, and solve the diffusion equation

## Numerical solution

x_array = make_1d_grid(diffusion_1d_config)

initial_condition = hat_initial_condition(x_array, diffusion_1d_config)

history = solve_diffusion_1d(initial_condition, diffusion_1d_config)

## Analytical solution

dx = compute_dx(diffusion_1d_config)

dt = compute_dt(diffusion_1d_config)

time_array = np.arange(0, max_iterations + 1) * dt

mode_indices = generate_mode_indices(num_modes)

mode_coefficients = compute_coefficients(
    initial_condition, 
    x_array, 
    mode_indices, 
    basis=basis,
)

series_terms = compute_series_terms(mode_indices, mode_coefficients, x_array, basis=basis)

history_ana = solve_heat_equation_1d(
    series_terms, 
    mode_indices,
    x_array,
    time_array, 
    diffusion_1d_config.viscosity,
    basis=basis)


# Vistualize the results

## Extract the script name and equation name for plotting

script_name = Path(__file__).name
equation = script_name.split('.')[0].split('_')[1:]
equation[0], equation[1] = equation[1], equation[0]
equation_name = ' '.join(equation)

## Plot the results

plot_snapshots(x_array, history_num=history, history_ana=history_ana, equation=equation_name, step_stride=step_stride, save_fig=save_fig)
plot_animation(x_array, history_num=history, history_ana=history_ana, equation=equation_name, save_fig=save_fig)
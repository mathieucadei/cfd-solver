import numpy as np
from pathlib import Path

from core import (
    Diffusion1DConfig,
    make_1d_grid,
    compute_1d_grid_points_spacing,
    hat_initial_condition,
    compute_time_step,
    solve_diffusion_1d_num,
    solve_diffusion_1d_ana,
    generate_mode_indices,
    compute_coefficients,
    compute_series_terms,
    compute_series,
    plot_snapshots,
    plot_animation,
)


# Inputs

## Configuration parameters for the 1D diffusion simulation

domain_length = 2.0
num_grid_points = 41
max_iterations = 41
sigma = 0.2
viscosity = 0.3
hat_start = 0.5
hat_end = 1.0

u_min = 1.0
u_max = 2.0


## Configuration parameters for the analytical solution

num_modes = 1000

## Visualization parameters

step_stride = 10
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

x = make_1d_grid(diffusion_1d_config)

u0 = hat_initial_condition(x, diffusion_1d_config)

history = solve_diffusion_1d_num(u0, diffusion_1d_config)

# Compute the analytical solution

dx = compute_1d_grid_points_spacing(diffusion_1d_config)
dt = compute_time_step(dx, diffusion_1d_config)
time_array = np.arange(1, max_iterations + 1) * dt
print(dt, time_array)

mode_indices = generate_mode_indices(num_modes)

mode_coefficients = compute_coefficients(
    u0, 
    x, 
    mode_indices, 
    basis="cosine"
)

series_terms = compute_series_terms(mode_indices, mode_coefficients, x, basis="cosine")

history_ana = solve_diffusion_1d_ana(
    series_terms, 
    mode_indices,
    x,
    time_array, 
    diffusion_1d_config.viscosity,
    basis="cosine")



# Vistualize the results

## Extract the script name and equation name for plotting

script_name = Path(__file__).name
equation = script_name.split('.')[0].split('_')[1:]
equation[0], equation[1] = equation[1], equation[0]
equation_name = ' '.join(equation)


## Plot the results

plot_snapshots(x, history_num=history, history_ana=history_ana, equation=equation_name, step_stride=step_stride, save_fig=save_fig)
plot_animation(x, history_num=history, history_ana=history_ana, equation=equation_name, save_fig=save_fig)
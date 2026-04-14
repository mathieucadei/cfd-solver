import numpy as np
from pathlib import Path

from core import (
    Diffusion1DConfig,
    make_1d_grid,
    hat_initial_condition,
    solve_diffusion_1d,
)

from post_processing import (plot_snapshots, plot_animation)


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

x_array = make_1d_grid(diffusion_1d_config)

time_array = np.arange(0, diffusion_1d_config.max_iterations + 1)

initial_condition = hat_initial_condition(x_array, diffusion_1d_config)

history = solve_diffusion_1d(initial_condition, diffusion_1d_config)


# Vistualize the results

## Extract the script name and equation name for plotting

script_name = Path(__file__).name
equation = script_name.split('.')[0].split('_')[1:]
equation[0], equation[1] = equation[1], equation[0]
equation_name = ' '.join(equation)

## Plot the results

plot_snapshots(x_array, time_array, history, equation=equation_name, step_stride=step_stride, save_fig=save_fig)
plot_animation(x_array, history, equation=equation_name, save_fig=save_fig)
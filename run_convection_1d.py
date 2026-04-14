import numpy as np
from pathlib import Path

from core import (
    Convection1DConfig,
    make_1d_grid,
    hat_initial_condition,
    solve_convection_1d,
)

from post_processing import (plot_snapshots, plot_animation)


# Inputs

## Configuration parameters for the 1D convection simulation

domain_length = 2.0
num_grid_points = 101
max_iterations = 100
time_step = 0.0025
hat_start = 0.5
hat_end = 1.0
u_min = 1.0
u_max = 2.0

## Visualization parameters

step_stride = 20
save_fig = False


# Create the configuration object

convection_1d_config = Convection1DConfig(
    domain_length=domain_length,
    num_grid_points=num_grid_points,
    max_iterations=max_iterations,
    time_step=time_step,
    hat_start=hat_start,
    hat_end=hat_end,
    u_min=u_min,
    u_max=u_max,
)


# Generate the grid, initial condition, and solve the convection equation

x_array = make_1d_grid(convection_1d_config)

time_array = np.arange(0, convection_1d_config.max_iterations + 1)

initial_condition = hat_initial_condition(x_array, convection_1d_config)

history = solve_convection_1d(initial_condition, convection_1d_config)


# Visualize the results

## Extract the script name and equation name for plotting

script_name = Path(__file__).name
equation = script_name.split('.')[0].split('_')[1:]
equation[0], equation[1] = equation[1], equation[0]
equation_name = ' '.join(equation)

## Plot the results

plot_snapshots(x_array, time_array, history, equation=equation_name, step_stride=step_stride, save_fig=save_fig)
plot_animation(x_array, history, equation=equation_name, save_fig=save_fig)
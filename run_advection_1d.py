import numpy as np
from pathlib import Path

from mini_ns import (
    Advection1DConfig,
    make_1d_grid,
    hat_initial_condition,
    solve_advection_1d,
    plot_snapshots,
    plot_animation,
)


# Inputs

## Configuration parameters for the 1D advection simulation

domain_length = 2.0
num_grid_points = 81
max_iterations = 25
time_step = 0.025
wavespeed = 1.0
hat_start = 0.5
hat_end = 1.0
u_min = 0.0
u_max = 2.0

## Visualization parameters

step_stride = 5
save_fig = True


# Create the configuration object

advection_1d_config = Advection1DConfig(
    domain_length=domain_length,
    num_grid_points=num_grid_points,
    max_iterations=max_iterations,
    time_step=time_step,
    wavespeed=wavespeed,
    hat_start=hat_start,
    hat_end=hat_end,
    u_min=u_min,
    u_max=u_max,
)


# Generate the grid, initial condition, and solve the advection equation

x = make_1d_grid(advection_1d_config)
u0 = hat_initial_condition(x, advection_1d_config)
history = solve_advection_1d(u0, advection_1d_config)

script_name = Path(__file__).name
equation = script_name.split('.')[0].split('_')[1:]
equation[0], equation[1] = equation[1], equation[0]
equation_name = ' '.join(equation).title()
print(equation_name)


# Plot the results

plot_snapshots(x, history, equation=equation_name, step_stride=step_stride, save_fig=save_fig)
plot_animation(x, history, equation=equation_name, save_fig=save_fig)
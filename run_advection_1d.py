import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core import (
    Advection1DConfig,
    make_1d_grid,
    hat_initial_condition,
    solve_advection_1d,
)

from post_processing import (
    plot_line,
    plot_line_cuts, 
    plot_snapshots, 
    plot_animation
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
u_min = 1.0
u_max = 2.0

## Visualization parameters

step_stride = 20
save_fig = False


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

x_array = make_1d_grid(advection_1d_config)
time_array = np.arange(0, advection_1d_config.max_iterations + 1)
initial_condition = hat_initial_condition(x_array, advection_1d_config)
history = solve_advection_1d(initial_condition, advection_1d_config)


# Plot the results

plot_line_cuts(
    x_values = x_array, 
    y_matrix_num = history,
    equation_name = '1D Advection',
    title = True,
    save = True,
)

plt.show()

# plot_snapshots(x_array, time_array, history, equation=equation_name, step_stride=step_stride, save_fig=save_fig)
# plot_animation(x_array, history, equation=equation_name, save_fig=save_fig)
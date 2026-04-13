import numpy as np
from pathlib import Path

from core import (
    BurgersEquation1DConfig,
    make_cole_hopf_1d_grid,
    cole_hopf_initial_condition,
    solve_burgers_equation_1d,
    solve_cole_hopf_1d,
)

from post_processing import (plot_snapshots, plot_animation)


# Inputs

## Configuration parameters for the 1D Burgers' equation simulation

domain_length = 6.0
num_grid_points = 101
max_iterations = 100
time_step = 0.0025
sigma = 0.2
viscosity = 0.07
hat_start = 0.5
hat_end = 1.0
u_min = 1.0
u_max = 2.0

## Visualization parameters

step_stride = 25
save_fig = False


# Create the configuration object

burgers_1d_config = BurgersEquation1DConfig(
    domain_length=domain_length,
    num_grid_points=num_grid_points,
    max_iterations=max_iterations,
    time_step=time_step,
    sigma=sigma,
    viscosity=viscosity,
    hat_start=hat_start,
    hat_end=hat_end,
    u_min=u_min,
    u_max=u_max,
)


# Generate the grid, initial condition, and solve the Burgers' equation

## Numerical solution

x_array = make_cole_hopf_1d_grid(burgers_1d_config)

initial_condition = cole_hopf_initial_condition(x_array, burgers_1d_config)

history_num = solve_burgers_equation_1d(initial_condition, burgers_1d_config)

## Analytical solution

# time_array = np.linspace(0, burgers_1d_config.max_iterations * burgers_1d_config.time_step, burgers_1d_config.max_iterations + 1)

# history_ana = solve_cole_hopf_1d(x_array, time_array, initial_condition, burgers_1d_config)

# Vistualize the results

## Extract the script name and equation name for plotting

script_name = Path(__file__).name
equation = script_name.split('.')[0].split('_')[1:]
equation[0], equation[1] = equation[1], equation[0]
equation_name = ' '.join(equation)

## Plot the results

plot_snapshots(x_array, history_num=history_num, equation=equation_name, step_stride=step_stride, save_fig=save_fig)
plot_animation(x_array, history_num=history_num, equation=equation_name, save_fig=save_fig)
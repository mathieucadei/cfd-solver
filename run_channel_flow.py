"""Run the 2D diffusion solver and generate solution plots."""



import os

from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import numpy as np
import math
from matplotlib import cm
import matplotlib.pyplot as plt

from core import (
    ChannelFlowConfig,
    channel_flow_initial_condition,
    make_x_grid,
    make_y_grid,
    solve_channel_flow,
)

from post_processing import (
    show_channel_flow_solution,
    show_channel_flow_solution_animation,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2
domain_length_y: float = 1
num_grid_points_x: int = 41
num_grid_points_y: int = 41
max_iterations: int = 10
max_pseudo_iterations: int = 50
time_step: float = 0.001
source: float = 1.0
density: float = 1.0
viscosity: float = 0.1
u_l1_norm_target: float = 0.001


# Visualization parameters

step_stride = 10
case_name = 'channel flow'
title = True
save = True
show_individual_plots = False


# Create the configuration object

channel_flow_config = ChannelFlowConfig(
    domain_length_x=domain_length_x,
    domain_length_y=domain_length_y,
    num_grid_points_x=num_grid_points_x,
    num_grid_points_y=num_grid_points_y,
    max_iterations=max_iterations,
    max_pseudo_iterations=max_pseudo_iterations,
    time_step=time_step,
    source=source,
    density=density,
    viscosity=viscosity,
    u_l1_norm_target=u_l1_norm_target
)


# Generate the grid and time array

x_array = make_x_grid(channel_flow_config)
y_array = make_y_grid(channel_flow_config)


# Initialize the initial condition

initial_condition = channel_flow_initial_condition(channel_flow_config)



# Solve the poisson equation

solution_matrix = solve_channel_flow(initial_condition, config=channel_flow_config)

u_solution_matrix = solution_matrix[0]

v_solution_matrix = solution_matrix[1]

u_solution_matrix_final = u_solution_matrix[-1, ...]

v_solution_matrix_final = v_solution_matrix[-1, ...]


# Post-processing

show_channel_flow_solution(
    x_values=x_array,
    y_values=y_array,
    u_solution_matrix=u_solution_matrix_final,
    v_solution_matrix=v_solution_matrix_final,
    case_name=case_name,
    title=title,
    save=save,
)

show_channel_flow_solution_animation(
    x_values=x_array,
    y_values=y_array,
    u_solution_history=u_solution_matrix,
    v_solution_history=v_solution_matrix,
    case_name=case_name,
    save=save,
)
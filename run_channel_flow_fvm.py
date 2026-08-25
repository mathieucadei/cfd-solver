"""Run the 2D diffusion solver and generate solution plots."""



import os

from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import numpy as np
import math
from matplotlib import cm
import matplotlib.pyplot as plt

from core import (
    ChannelFlowFVMConfig,
    channel_flow_initial_condition_fvm,
    build_h_spacing,
    build_face_positions,
    build_centers,
    solve_channel_flow_fvm,
)

from post_processing import (
    show_channel_flow_solution,
    show_channel_flow_solution_animation,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2
domain_length_y: float = 1
num_cells_x: int = 40
num_cells_y: int = 40
expansion_ratio_x: float = 0.
expansion_ratio_y: float = 0.
max_iterations: int = 500
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
save = False
show_individual_plots = False


# Create the configuration object

channel_flow_config = ChannelFlowFVMConfig(
    domain_length_x=domain_length_x,
    domain_length_y=domain_length_y,
    num_cells_x=num_cells_x,
    num_cells_y=num_cells_y,
    expansion_ratio_x=expansion_ratio_x,
    expansion_ratio_y=expansion_ratio_y,
    max_iterations=max_iterations,
    max_pseudo_iterations=max_pseudo_iterations,
    time_step=time_step,
    source=source,
    density=density,
    viscosity=viscosity,
    u_l1_norm_target=u_l1_norm_target
)


# Generate the grid and time array

hx_array, hy_array = build_h_spacing(channel_flow_config)
xc_array, yc_array = build_centers(channel_flow_config)


# Initialize the initial condition

initial_condition = channel_flow_initial_condition_fvm(channel_flow_config)



# Solve the poisson equation

solution_matrix = solve_channel_flow_fvm(initial_condition, config=channel_flow_config)

u_solution_matrix = solution_matrix[0]

v_solution_matrix = solution_matrix[1]

u_solution_matrix_final = u_solution_matrix[-1, ...]

v_solution_matrix_final = v_solution_matrix[-1, ...]


# Post-processing

show_channel_flow_solution(
    x_values=xc_array,
    y_values=yc_array,
    u_solution_matrix=u_solution_matrix_final,
    v_solution_matrix=v_solution_matrix_final,
    case_name=case_name,
    title=title,
    save=save,
)

show_channel_flow_solution_animation(
    x_values=xc_array,
    y_values=yc_array,
    u_solution_history=u_solution_matrix,
    v_solution_history=v_solution_matrix,
    source=source,
    case_name=case_name,
    save=save,
)
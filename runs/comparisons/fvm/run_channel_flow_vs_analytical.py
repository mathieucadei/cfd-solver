"""Run the 2D channel flow solver and compare with Poiseuille flow."""



import os

from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import numpy as np
import math
from matplotlib import cm
import matplotlib.pyplot as plt

from pathlib import Path

from core import fvm, analytical

from post_processing import (
    show_channel_flow_solution,
    show_channel_flow_solution_overview,
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
max_iterations: int = 10
max_pseudo_iterations: int = 50
time_step: float = 0.001
source: float = 1.0
density: float = 1.0
viscosity: float = 0.1
u_l1_norm_target: float = 1e-6


# Visualization parameters

step_stride = 10
cut_indices=[num_cells_x // 2]
case_name = 'channel flow'
case_name_as_title = True
save = False
show_individual_plots = False


# Create the configuration object

channel_flow_config = fvm.ChannelFlowConfig(
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

hx_array, hy_array = fvm.build_h_spacing(channel_flow_config)
xc_array, yc_array = fvm.build_centers(channel_flow_config)


# Initialize the initial condition

initial_condition = fvm.channel_flow_initial_condition(channel_flow_config)



# Solve the poisson equation

solution_matrix = fvm.solve_channel_flow(initial_condition, config=channel_flow_config)

u_solution_matrix = solution_matrix[0]

v_solution_matrix = solution_matrix[1]

u_solution_matrix_final = u_solution_matrix[-1, ...]

v_solution_matrix_final = v_solution_matrix[-1, ...]


# Poiseulle flow

u_analytical = analytical.compute_poiseuille_flow(
    y_array=yc_array,
    config=channel_flow_config,
)

u_analytical_2d = np.tile(u_analytical[:, None], (1, num_cells_x))

# Post-processing

x_index = num_cells_x // 2

u_numerical = u_solution_matrix_final[:, x_index]

error = u_numerical - u_analytical
error_2d = u_solution_matrix_final - u_analytical_2d

l2_error = (
    np.linalg.norm(u_numerical - u_analytical)
    / np.linalg.norm(u_analytical)
)

metrics = (
    f"Analytical umax = {np.max(u_analytical):.6f}\n"
    f"Numerical umax = {np.max(u_numerical):.6f}\n"
    f"Maximum |v| = {np.max(np.abs(v_solution_matrix_final)):.2e}\n"
    f"Relative L2 error = {l2_error:.2e}"
)


show_channel_flow_solution_overview(
    x_values=xc_array,
    y_values=yc_array,
    u_solution_matrix=u_solution_matrix_final,
    v_solution_matrix=v_solution_matrix_final,
    ana_u_x_values=yc_array,
    ana_u_values=u_analytical,
    error=error,
    error_2d=error_2d,
    metrics=metrics,
    case_name=case_name,
    case_name_as_title=case_name_as_title,
    save=save,
    cut_indices=cut_indices,
)
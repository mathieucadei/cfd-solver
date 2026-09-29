"""Run the 2D channel flow FVM solver and compare with Poiseuille flow."""



import os

from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import math
from matplotlib import cm
import matplotlib.pyplot as plt

from pathlib import Path

from core import fdm, fvm, analytical

from post_processing import (
    show_channel_flow_solution_overview,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2
domain_length_y: float = 1
num_grid_points_x: int = 40
num_grid_points_y: int = 40
max_iterations: int = 10
max_pseudo_iterations: int = 50
time_step: float = 0.001
source: float = 1.0
density: float = 1.0
viscosity: float = 0.1
u_l1_norm_target: float = 0.1

num_cells_x: int = 40
num_cells_y: int = 40
expansion_ratio_x: float = 0.
expansion_ratio_y: float = 0.


# Visualization parameters

step_stride = 10
cut_indices=[num_grid_points_x // 2]
case_name = 'channel flow fdm vs fvm comparison'
case_name_as_title = True
save = False
show_individual_plots = False


# Create the configuration object

fdm_channel_flow_config = fdm.ChannelFlowConfig(
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

fvm_channel_flow_config = fvm.ChannelFlowConfig(
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

x_array = fdm.make_x_grid(fdm_channel_flow_config)
y_array = fdm.make_y_grid(fdm_channel_flow_config)

hx_array, hy_array = fvm.build_h_spacing(fvm_channel_flow_config)
xc_array, yc_array = fvm.build_centers(fvm_channel_flow_config)


# Initialize the initial condition

fdm_initial_condition = fdm.channel_flow_initial_condition(fdm_channel_flow_config)

fvm_initial_condition = fvm.channel_flow_initial_condition(fvm_channel_flow_config)



# Solve the poisson equation

fdm_solution_matrix = fdm.solve_channel_flow(fdm_initial_condition, config=fdm_channel_flow_config)

fdm_u_solution_matrix = fdm_solution_matrix[0]

fdm_v_solution_matrix = fdm_solution_matrix[1]

fdm_u_solution_matrix_final = fdm_u_solution_matrix[-1, ...]

fdm_v_solution_matrix_final = fdm_v_solution_matrix[-1, ...]


fvm_solution_matrix = fvm.solve_channel_flow(fvm_initial_condition, config=fvm_channel_flow_config)

fvm_u_solution_matrix = fvm_solution_matrix[0]

fvm_v_solution_matrix = fvm_solution_matrix[1]

fvm_u_solution_matrix_final = fvm_u_solution_matrix[-1, ...]

fvm_v_solution_matrix_final = fvm_v_solution_matrix[-1, ...]


# OpenFOAM

# u_analytical = analytical.compute_poiseuille_flow(
#     y_array=yc_array,
#     config=channel_flow_config,
# )

# u_analytical_2d = np.tile(u_analytical[:, None], (1, num_cells_x))

DATA = Path(__file__).resolve().parents[3] / 'data'
openfoam = pd.read_csv(DATA / 'openfoam_channel_flow_2d_u_profile.csv')
u_openfoam=openfoam['U:0'].to_numpy()

# Post-processing

x_index = num_cells_x // 2

fdm_u_numerical = fdm_u_solution_matrix_final[:, x_index]
fvm_u_numerical = fvm_u_solution_matrix_final[:, x_index]

error = fdm_u_numerical - fvm_u_numerical
error_2d = fdm_u_solution_matrix_final - fvm_u_solution_matrix_final

l2_error = (
    np.linalg.norm(fdm_u_numerical - fvm_u_numerical)
    / np.linalg.norm(fvm_u_numerical)
)

metrics = (
    f"FVM umax = {np.max(fvm_u_numerical):.6f}\n"
    f"FDM umax = {np.max(fdm_u_numerical):.6f}\n"
    f"Maximum |v| = {np.max(np.abs(fdm_v_solution_matrix_final)):.2e}\n"
    f"Relative L2 error = {l2_error:.2e}"
)


show_channel_flow_solution_overview(
    x_values=xc_array,
    y_values=yc_array,
    u_solution_matrix=fdm_u_solution_matrix_final,
    v_solution_matrix=fdm_v_solution_matrix_final,
    comp_u_values=fvm_u_numerical,
    comp_label='FVM',
    error=error,
    error_2d=error_2d,
    metrics=metrics,
    case_name=case_name,
    case_name_as_title=case_name_as_title,
    save=save,
    cut_indices=cut_indices,
)
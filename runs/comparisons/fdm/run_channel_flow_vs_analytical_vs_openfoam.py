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

from core import fdm, analytical

from post_processing import (
    show_channel_flow_solution,
    show_channel_flow_solution_overview,
    show_channel_flow_solution_animation,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2
domain_length_y: float = 1
num_grid_points_x: int = 40
num_grid_points_y: int = 40
max_iterations: int = 10000
max_pseudo_iterations: int = 50
time_step: float = 0.001
source: float = 1.0
density: float = 1.0
viscosity: float = 0.1
u_l1_norm_target: float = 1e-6


# Visualization parameters

step_stride = 10
cut_indices=[num_grid_points_x // 2]
case_name = 'channel flow'
case_name_as_title = True
save = False
show_individual_plots = False


# Create the configuration object

channel_flow_config = fdm.ChannelFlowConfig(
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

x_array = fdm.make_x_grid(channel_flow_config)
y_array = fdm.make_y_grid(channel_flow_config)

# Initialize the initial condition

initial_condition = fdm.channel_flow_initial_condition(channel_flow_config)



# Solve the poisson equation

solution_matrix = fdm.solve_channel_flow(initial_condition, config=channel_flow_config)

u_solution_matrix = solution_matrix[0]

v_solution_matrix = solution_matrix[1]

u_solution_matrix_final = u_solution_matrix[-1, ...]

v_solution_matrix_final = v_solution_matrix[-1, ...]


# OpenFOAM

u_analytical = analytical.compute_poiseuille_flow(
    y_array=y_array,
    config=channel_flow_config,
)

u_analytical_2d = np.tile(u_analytical[:, None], (1, num_grid_points_x))

DATA = Path(__file__).resolve().parents[3] / 'data'
openfoam = pd.read_csv(DATA / 'openfoam_channel_flow_2d_u_profile.csv')
u_openfoam=openfoam['U:0'].to_numpy()
u_openfoam_2d = np.tile(u_openfoam[:, None], (1, num_grid_points_x))

# Post-processing

x_index = num_grid_points_x // 2

u_numerical = u_solution_matrix_final[:, x_index]

show_channel_flow_solution_overview(
    x_values=x_array,
    y_values=y_array,
    u_solution_matrix=u_solution_matrix_final,
    v_solution_matrix=v_solution_matrix_final,
    ana_u_values=u_analytical,
    comp_u_values=u_openfoam,
    case_name=case_name,
    case_name_as_title=case_name_as_title,
    save=save,
    cut_indices=cut_indices,
)
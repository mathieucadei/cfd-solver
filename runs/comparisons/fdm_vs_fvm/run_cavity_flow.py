"""Run the 2D lid-driven cavity flow FVM solver and compare with Ghia et al. (1982)."""



import os

from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from pathlib import Path

from core import fdm, fvm

from post_processing import (
    show_cavity_flow_solution_overview,
)



# Pre-processing
# Simulation parameters

reynolds_number = 100

domain_length_x: float = 1.0
domain_length_y: float = 1.0
num_grid_points_x: int = 40
num_grid_points_y: int = 40
max_iterations = 10000
max_pseudo_iterations: int = 50
time_step: float = 0.001
u_lid: float = 1.0
density: float = 1.0
viscosity: float = u_lid*domain_length_x/reynolds_number

num_cells_x: int = 40
num_cells_y: int = 40
expansion_ratio_x: float = 0.
expansion_ratio_y: float = 0.


# Visualization parameters

step_stride = 10
# cut_indices=[(num_cells_x+1) // 2]
case_name = f'lid-driven cavity flow FDM - Re {reynolds_number} FDM vs FVM'
case_name_as_title = True
save = False
show_individual_plots = False


# Create the configuration object

fdm_cavity_flow_config = fdm.CavityFlowConfig(
    domain_length_x=domain_length_x,
    domain_length_y=domain_length_y,
    num_grid_points_x=num_grid_points_x,
    num_grid_points_y=num_grid_points_y,
    max_iterations=max_iterations,
    max_pseudo_iterations=max_pseudo_iterations,
    time_step=time_step,
    u_lid=u_lid,
    density=density,
    viscosity=viscosity,
)

fvm_cavity_flow_config = fvm.CavityFlowConfig(
    domain_length_x=domain_length_x,
    domain_length_y=domain_length_y,
    num_cells_x=num_cells_x,
    num_cells_y=num_cells_y,
    expansion_ratio_x=expansion_ratio_x,
    expansion_ratio_y=expansion_ratio_y,
    max_iterations=max_iterations,
    max_pseudo_iterations=max_pseudo_iterations,
    time_step=time_step,
    u_lid=u_lid,
    density=density,
    viscosity=viscosity,
)


# Generate the grid and time array

x_array = fdm.make_x_grid(fdm_cavity_flow_config)
y_array = fdm.make_y_grid(fdm_cavity_flow_config)

hx_array, hy_array = fvm.build_h_spacing(fvm_cavity_flow_config)
xc_array, yc_array = fvm.build_centers(fvm_cavity_flow_config)



# Initialize the initial condition

fdm_initial_condition = fdm.cavity_flow_initial_condition(fdm_cavity_flow_config)

fvm_initial_condition = fvm.cavity_flow_initial_condition(fvm_cavity_flow_config)



# Solve the poisson equation

fdm_solution_matrix = fdm.solve_cavity_flow(fdm_initial_condition, config=fdm_cavity_flow_config)

fdm_u_solution_matrix = fdm_solution_matrix[0]

fdm_v_solution_matrix = fdm_solution_matrix[1]

fdm_p_solution_matrix = fdm_solution_matrix[2]

fdm_u_solution_matrix_final = fdm_u_solution_matrix[-1, ...]

fdm_v_solution_matrix_final = fdm_v_solution_matrix[-1, ...]

fdm_p_solution_matrix_final = fdm_p_solution_matrix[-1, ...]


fvm_solution_matrix = fvm.solve_cavity_flow(fvm_initial_condition, config=fvm_cavity_flow_config)

fvm_u_solution_matrix = fvm_solution_matrix[0]

fvm_v_solution_matrix = fvm_solution_matrix[1]

fvm_p_solution_matrix = fvm_solution_matrix[2]

fvm_u_solution_matrix_final = fvm_u_solution_matrix[-1, ...]

fvm_v_solution_matrix_final = fvm_v_solution_matrix[-1, ...]

fvm_p_solution_matrix_final = fvm_p_solution_matrix[-1, ...]


# Ghia et al. (1982)

# DATA = Path(__file__).resolve().parents[3] / 'data'
# openfoam_table_1 = pd.read_csv(DATA / 'openfoam_table_1.csv')
# openfoam_table_2 = pd.read_csv(DATA / 'openfoam_table_2.csv')

# validation_u_x_values=openfoam_table_1['arc_length']
# validation_v_x_values=openfoam_table_2['arc_length']
# validation_u_values=openfoam_table_1['U:0']
# validation_v_values=openfoam_table_2['U:1']

# u_scatter_label='openfoam'
# v_scatter_label='openfoam'

DATA = Path(__file__).resolve().parents[3] / 'data'
openfoam_field = pd.read_csv(DATA / 'openfoam_field.csv')

comp_u_values=openfoam_field['U:0'].to_numpy().reshape(num_grid_points_y, num_grid_points_x)
comp_v_values=openfoam_field['U:1'].to_numpy().reshape(num_grid_points_y, num_grid_points_x)

comp_label='FVM'

# Post-processing


show_cavity_flow_solution_overview(
    x_values=x_array,
    y_values=y_array,
    u_solution_matrix=fdm_u_solution_matrix_final,
    v_solution_matrix=fdm_v_solution_matrix_final,
    p_solution_matrix=fdm_p_solution_matrix_final,
    comp_u_values=fvm_u_solution_matrix_final,
    comp_v_values=fvm_v_solution_matrix_final,
    case_name=case_name,
    case_name_as_title=case_name_as_title,
    save=save,
    step_stride=step_stride,
    comp_label=comp_label,
)
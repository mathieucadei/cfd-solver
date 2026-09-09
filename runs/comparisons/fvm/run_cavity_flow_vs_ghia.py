"""Run the 2D lid-driven cavity flow solver and generate solution plots."""



import os

from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from pathlib import Path

from core import fvm

from post_processing import (
    show_cavity_flow_solution_overview,
)



# Pre-processing
# Simulation parameters

reynolds_number = 100

domain_length_x: float = 1.0
domain_length_y: float = 1.0
num_cells_x: int = 40
num_cells_y: int = 40
expansion_ratio_x: float = 0.
expansion_ratio_y: float = 0.
max_iterations = 10000
max_pseudo_iterations: int = 50
time_step: float = 0.001
u_lid: float = 1.0
density: float = 1.0
viscosity: float = u_lid*domain_length_x/reynolds_number


# Visualization parameters

step_stride = 10
cut_indices=[(num_cells_x+1) // 2]
case_name = f'lid-driven cavity flow FVM - Re {reynolds_number}'
case_name_as_title = True
save = False
show_individual_plots = False


# Create the configuration object

cavity_flow_config = fvm.CavityFlowConfig(
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

hx_array, hy_array = fvm.build_h_spacing(cavity_flow_config)
xc_array, yc_array = fvm.build_centers(cavity_flow_config)


# Initialize the initial condition

initial_condition = fvm.cavity_flow_initial_condition(cavity_flow_config)



# Solve the poisson equation

solution_matrix = fvm.solve_cavity_flow(initial_condition, config=cavity_flow_config)

u_solution_matrix = solution_matrix[0]

v_solution_matrix = solution_matrix[1]

p_solution_matrix = solution_matrix[2]

u_solution_matrix_final = u_solution_matrix[-1, ...]

v_solution_matrix_final = v_solution_matrix[-1, ...]

p_solution_matrix_final = p_solution_matrix[-1, ...]


# Ghia et al. (1982)

DATA = Path(__file__).resolve().parents[3] / 'data'
ghia_table_1 = pd.read_csv(DATA / 'ghia_table_1.csv')
ghia_table_2 = pd.read_csv(DATA / 'ghia_table_2.csv')

validation_u_x_values=ghia_table_1['y']
validation_v_x_values=ghia_table_2['x']
validation_u_values=ghia_table_1['100']
validation_v_values=ghia_table_2['100']

u_scatter_label='x=0.5 - Ghia et al. (1982)'
v_scatter_label='y=0.5 - Ghia et al. (1982)'


# Post-processing


show_cavity_flow_solution_overview(
    x_values=xc_array,
    y_values=yc_array,
    u_solution_matrix=u_solution_matrix_final,
    v_solution_matrix=v_solution_matrix_final,
    p_solution_matrix=p_solution_matrix_final,
    validation_u_x_values=validation_u_x_values,
    validation_v_x_values=validation_v_x_values,
    validation_u_values=validation_u_values,
    validation_v_values=validation_v_values,
    case_name=case_name,
    case_name_as_title=case_name_as_title,
    save=save,
    cut_indices=cut_indices,
    u_scatter_label=u_scatter_label,
    v_scatter_label=v_scatter_label,
)
"""Run the 2D diffusion solver and generate solution plots."""



import os

from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

from core import fvm

from post_processing import (
    show_cavity_flow_solution,
    show_cavity_flow_solution_animation,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2.0
domain_length_y: float = 1.0
num_cells_x: int = 30
num_cells_y: int = 30
expansion_ratio_x: float = 0.
expansion_ratio_y: float = 0.
max_iterations: int = 500
max_pseudo_iterations: int = 50
time_step: float = 0.001
u_lid: float = 1.0
density: float = 1.0
viscosity: float = 0.1


# Visualization parameters

step_stride = 10
case_name = 'cavity flow FVM'
title = True
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


# Post-processing

show_cavity_flow_solution(
    x_values=xc_array,
    y_values=yc_array,
    u_solution_matrix=u_solution_matrix_final,
    v_solution_matrix=v_solution_matrix_final,
    p_solution_matrix=p_solution_matrix_final,
    case_name=case_name,
    title=title,
    save=save,
)


show_cavity_flow_solution_animation(
    x_values=xc_array,
    y_values=yc_array,
    u_solution_history=u_solution_matrix,
    v_solution_history=v_solution_matrix,
    p_solution_history=p_solution_matrix,
    u_lid=u_lid,
    case_name=case_name,
    save=save,
)
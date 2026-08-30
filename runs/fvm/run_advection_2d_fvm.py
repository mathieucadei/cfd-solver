"""Run the 2D advection solver and generate solution plots."""



import numpy as np
import matplotlib.pyplot as plt

from core import (
    Advection2DFVMConfig,
    hat_initial_condition_2d_fvm,
    build_h_spacing,
    build_face_positions,
    build_centers,
    solve_advection_2d_fvm,
)

from post_processing import (
    show_solution_2d_animation,
    show_solution_overview,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2.0
domain_length_y: float = 2.0
num_cells_x: int = 80
num_cells_y: int = 80
expansion_ratio_x: float = 0.
expansion_ratio_y: float = 0.
max_iterations: int = 80
sigma: float = 0.2
wavespeed: float = 1.0
hat_start_x: float = 0.5
hat_start_y: float = 0.5
hat_end_x: float = 1.0
hat_end_y: float = 1.0
u_min: float = 1.0
u_max: float = 2.0


# Visualization parameters

step_stride = 10
case_name = '2d advection'
title = True
save = False
show_individual_plots = False


# Create the configuration object

advection_2d_config = Advection2DFVMConfig(
    domain_length_x=domain_length_x,
    domain_length_y=domain_length_y,
    num_cells_x=num_cells_x,
    num_cells_y=num_cells_y,
    expansion_ratio_x=expansion_ratio_x,
    expansion_ratio_y=expansion_ratio_y,
    max_iterations=max_iterations,
    sigma=sigma,
    wavespeed=wavespeed,
    hat_start_x=hat_start_x,
    hat_start_y=hat_start_y,
    hat_end_x=hat_end_x,
    hat_end_y= hat_end_y,
    u_min=u_min,
    u_max=u_max,
)


# Generate the grid and time array

hx_array, hy_array = build_h_spacing(advection_2d_config)
xc_array, yc_array = build_centers(advection_2d_config)
time_array = np.arange(0, advection_2d_config.max_iterations + 1)


# Initialize the initial condition

initial_condition = hat_initial_condition_2d_fvm(advection_2d_config)



# Solve the advection equation

solution_matrix = solve_advection_2d_fvm(initial_condition, advection_2d_config)

solution_final_x = solution_matrix[-1, :, :]

solution_final_y = solution_final_x.T

solution_final = solution_matrix[-1]

xf, yf = build_face_positions(advection_2d_config)



# Post-processing

fig, ax = plt.subplots(figsize=(10,3))
pc = ax.pcolormesh(xf, yf, solution_final[:, :], edgecolors='k', linewidth=0.3)
fig.colorbar(pc, label='u')
plt.show()

show_solution_overview(
    x_values=xc_array, 
    y_values=yc_array, 
    num_solution_matrix=solution_final,
    y_label='y',
    step_stride=step_stride,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_2d_animation(
    x_values=xc_array,
    y_values=yc_array, 
    solution_history=solution_matrix,
    case_name=case_name,
    save=save,
)
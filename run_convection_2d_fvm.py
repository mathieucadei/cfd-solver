"""Run the 2D advection solver and generate solution plots."""



import numpy as np
import matplotlib.pyplot as plt

from core import (
    Convection2DFVMConfig,
    hat_convective_initial_condition_2d_fvm,
    build_h_spacing,
    build_face_positions,
    build_centers,
    solve_convection_2d_fvm,
)

from post_processing import (
    show_solution_uv_surfaces,
    show_solution_uv_2d_animations,
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
hat_start_x: float = 0.5
hat_start_y: float = 0.5
hat_end_x: float = 1.0
hat_end_y: float = 1.0
u_min: float = 1.0
u_max: float = 2.0
v_min: float = 1.0
v_max: float = 2.0


# Visualization parameters

step_stride = 10
case_name = '2d advection'
title = True
save = False
show_individual_plots = False


# Create the configuration object

convection_2d_config = Convection2DFVMConfig(
    domain_length_x=domain_length_x,
    domain_length_y=domain_length_y,
    num_cells_x=num_cells_x,
    num_cells_y=num_cells_y,
    expansion_ratio_x=expansion_ratio_x,
    expansion_ratio_y=expansion_ratio_y,
    max_iterations=max_iterations,
    sigma=sigma,
    hat_start_x=hat_start_x,
    hat_start_y=hat_start_y,
    hat_end_x=hat_end_x,
    hat_end_y= hat_end_y,
    u_min=u_min,
    u_max=u_max,
)


# Generate the grid and time array

hx_array, hy_array = build_h_spacing(convection_2d_config)
xc_array, yc_array = build_centers(convection_2d_config)
time_array = np.arange(0, convection_2d_config.max_iterations + 1)


# Initialize the initial condition

initial_condition = hat_convective_initial_condition_2d_fvm(convection_2d_config)



# Solve the advection equation

u_solution_matrix, v_solution_matrix = solve_convection_2d_fvm(initial_condition, convection_2d_config)

# u_solution_final, v_solution_final = u_solution_matrix[-1, ...], v_solution_matrix[-1, ...]

u_solution_final_x, v_solution_final_x = u_solution_matrix[-1, :, :], v_solution_matrix[-1, :, :]

u_solution_final_y, v_solution_final_y = u_solution_final_x.T, v_solution_final_x.T

u_solution_matrix_final, v_solution_matrix_final = u_solution_matrix[-1], v_solution_matrix[-1]

xf, yf = build_face_positions(convection_2d_config)



# Post-processing

fig, ax = plt.subplots(2, 1, figsize=(12,6))

ax0 = ax[0].pcolormesh(xf, yf, u_solution_matrix_final[:, :], edgecolors='k', linewidth=0.3)
fig.colorbar(ax0, label='u')

ax1 = ax[1].pcolormesh(xf, yf, v_solution_matrix_final[:, :], edgecolors='k', linewidth=0.3)
fig.colorbar(ax1, label='v')

plt.show()

show_solution_uv_surfaces(
    x_values=xc_array,
    y_values=yc_array,
    u_solution_matrix=u_solution_matrix_final,
    v_solution_matrix=v_solution_matrix_final,
    case_name=f'{case_name} final',
    title=title,
    save=save,
)

show_solution_uv_2d_animations(
    x_values=xc_array,
    y_values=yc_array, 
    u_solution_history=u_solution_matrix,
    v_solution_history=v_solution_matrix,
    case_name=case_name,
    save=save,
)
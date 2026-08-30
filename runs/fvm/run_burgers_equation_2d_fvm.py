"""Run the 2D Burgers' equation solver and generate solution plots."""



import numpy as np
import matplotlib.pyplot as plt

from core import (
    BurgersEquation2DFVMConfig,
    hat_convective_initial_condition_2d_fvm,
    build_h_spacing,
    build_face_positions,
    build_centers,
    solve_burgers_equation_2d_fvm,
)

from post_processing import (
    show_solution_uv_2d_animations,
    show_solution_uv_surfaces,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2.0
domain_length_y: float = 2.0
num_cells_x: int = 30
num_cells_y: int = 30
expansion_ratio_x: float = 0.
expansion_ratio_y: float = 0.
max_iterations: int = 120
sigma: float = 0.0009
viscosity: float = 0.01
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
case_name = '2d burgers'
title = True
save = False


# Create the configuration object

burgers_equation_2d_config = BurgersEquation2DFVMConfig(
    domain_length_x=domain_length_x,
    domain_length_y=domain_length_y,
    num_cells_x=num_cells_x,
    num_cells_y=num_cells_y,
    expansion_ratio_x=expansion_ratio_x,
    expansion_ratio_y=expansion_ratio_y,
    max_iterations=max_iterations,
    sigma=sigma,
    viscosity=viscosity,
    hat_start_x=hat_start_x,
    hat_start_y=hat_start_y,
    hat_end_x=hat_end_x,
    hat_end_y= hat_end_y,
    u_min=u_min,
    u_max=u_max,
    v_min=v_min,
    v_max=v_max,   
)


# Generate the grid and time array

hx_array, hy_array = build_h_spacing(burgers_equation_2d_config)
xc_array, yc_array = build_centers(burgers_equation_2d_config)
time_array = np.arange(0, burgers_equation_2d_config.max_iterations + 1)


# Initialize the initial condition

initial_condition = hat_convective_initial_condition_2d_fvm(burgers_equation_2d_config)



# Solve the advection equation

u_solution_matrix, v_solution_matrix = solve_burgers_equation_2d_fvm(initial_condition, burgers_equation_2d_config)

u_solution_final_x, v_solution_final_x = u_solution_matrix[-1, :, :], v_solution_matrix[-1, :, :]

u_solution_final_y, v_solution_final_y = u_solution_final_x.T, v_solution_final_x.T

u_solution_matrix_final, v_solution_matrix_final = u_solution_matrix[-1], v_solution_matrix[-1]

xf, yf = build_face_positions(burgers_equation_2d_config)



# Post-processing

fig, ax = plt.subplots(2, 1, figsize=(12,6))

ax0 = ax[0].pcolormesh(xf, yf, u_solution_matrix_final[:, :], edgecolors='k', linewidth=0.3)
fig.colorbar(ax0, label='u')

ax1 = ax[1].pcolormesh(xf, yf, v_solution_matrix_final[:, :], edgecolors='k', linewidth=0.3)
fig.colorbar(ax1, label='v')

plt.show()


X, Y = np.meshgrid(xc_array, xc_array)

fig, ax = plt.subplots(1, 2, figsize=(12,6))

ax[0].quiver(X[::3, ::3], Y[::3, ::3], u_solution_matrix_final[::3, ::3], v_solution_matrix_final[::3, ::3])

div = np.gradient(u_solution_matrix_final, xc_array, axis=1) + np.gradient(v_solution_matrix_final, yc_array, axis=0)
pc = ax[1].pcolormesh(X, Y, div)   # or contourf
fig.colorbar(pc, label='v')

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
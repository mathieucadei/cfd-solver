"""Run the 2D channel flow solver and compare with Poiseuille flow."""



import os

from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import numpy as np
import math
from matplotlib import cm
import matplotlib.pyplot as plt

from pathlib import Path

from core import fdm

from post_processing import (
    show_channel_flow_solution,
    show_channel_flow_solution_animation,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2
domain_length_y: float = 1
num_grid_points_x: int = 41
num_grid_points_y: int = 41
max_iterations: int = 10
max_pseudo_iterations: int = 50
time_step: float = 0.001
source: float = 1.0
density: float = 1.0
viscosity: float = 0.1
u_l1_norm_target: float = 1e-6


# Visualization parameters

step_stride = 10
case_name = 'channel flow'
title = True
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


# Post-processing
u_analytical = (
    source
    / (2 * viscosity)
    * y_array
    * (domain_length_y - y_array)
)
u_analytical_2d = np.tile(u_analytical[:, None], (1, num_grid_points_x))

x_index = num_grid_points_x // 2

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

X, Y = np.meshgrid(x_array, y_array)
M = np.linalg.norm([u_solution_matrix_final, v_solution_matrix_final], axis=0)
norm = Normalize(vmin=np.min(M), vmax=np.max(M))

fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

ax0 = ax[0, 0].quiver(
    X[::2,::2], 
    Y[::2,::2], 
    u_solution_matrix_final[::2,::2], 
    v_solution_matrix_final[::2,::2],
    M[::2,::2],
    scale=20,
    cmap=cm.plasma,
    norm=norm,
)
ax[0, 0].set_xlim(0, 2)
ax[0, 0].set_ylim(0, 1)
ax[0, 0].set_xlabel('x')
ax[0, 0].set_ylabel('y', rotation=0)
ax[0, 0].set_title('Velocity Field')
fig.colorbar(ax0, label='Velocity magnitude')

ax1 = ax[0, 1].contourf(X, Y, error_2d, cmap=cm.plasma)

ax[0, 1].set_xlabel('x')
ax[0, 1].set_ylabel('y', rotation=0)
ax[0, 1].set_title('Velocity Error Field')

fig.colorbar(ax1, label='u numerical - u analytical')

ax[1, 0].plot(
    u_numerical,
    y_array,
    label="Numerical"
)
ax[1, 0].plot(
    u_analytical,
    y_array,
    "--",
    label="Poiseuille analytical"
)
ax[1, 0].grid(alpha=0.3)
ax[1, 0].set_xlim(0, 2)
ax[1, 0].set_xlabel('u')
ax[1, 0].set_ylabel('y', rotation=0)
ax[1, 0].legend()
ax[1, 0].set_title('Velocity Profile at x = 1.0')

ax[1, 1].plot(
    error,
    y_array,
    label="Error"
)
ax[1, 1].axvline(0, color='k', linewidth=0.8)
ax[1, 1].grid(alpha=0.3)
ax[1, 1].set_xlabel('Error')
ax[1, 1].set_ylabel('y', rotation=0)
ax[1, 1].legend()
ax[1, 1].set_title('Numerical − Analytical Error')
ax[1, 1].text(
    0.03,
    0.97,
    metrics,
    transform=ax[1, 1].transAxes,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.8)
)

fig.suptitle('Plane Poiseuille Flow — Numerical vs Analytical')
plt.show()



# show_channel_flow_solution(
#     x_values=x_array,
#     y_values=y_array,
#     u_solution_matrix=u_solution_matrix_final,
#     v_solution_matrix=v_solution_matrix_final,
#     case_name=case_name,
#     title=title,
#     save=save,
# )

# show_channel_flow_solution_animation(
#     x_values=x_array,
#     y_values=y_array,
#     u_solution_history=u_solution_matrix,
#     v_solution_history=v_solution_matrix,
#     source=source,
#     case_name=case_name,
#     save=save,
# )
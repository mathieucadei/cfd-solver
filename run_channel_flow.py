"""Run the 2D diffusion solver and generate solution plots."""



import os

from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import numpy as np
import math
from matplotlib import cm
import matplotlib.pyplot as plt

from core import (
    ChannelFlowConfig,
    channel_flow_initial_condition,
    make_x_grid,
    make_y_grid,
    solve_channel_flow,
)

from post_processing import (
    show_cavity_flow_solution,
    show_cavity_flow_solution_animation,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2.0
domain_length_y: float = 2.0
num_grid_points_x: int = 41
num_grid_points_y: int = 41
max_iterations: int = 10
max_pseudo_iterations: int = 50
time_step: float = 0.001
source: float = 1.0
density: float = 1.0
viscosity: float = 0.1
u_l1_norm_target: float = 0.001


# Visualization parameters

step_stride = 10
case_name = 'channel flow'
title = True
save = False
show_individual_plots = False


# Create the configuration object

channel_flow_config = ChannelFlowConfig(
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

x_array = make_x_grid(channel_flow_config)
y_array = make_y_grid(channel_flow_config)


# Initialize the initial condition

initial_condition = channel_flow_initial_condition(channel_flow_config)



# Solve the poisson equation

solution_matrix = solve_channel_flow(initial_condition, config=channel_flow_config)

u_solution_matrix = solution_matrix[0]

v_solution_matrix = solution_matrix[1]

p_solution_matrix = solution_matrix[2]

u_solution_matrix_final = u_solution_matrix[-1, ...]

v_solution_matrix_final = v_solution_matrix[-1, ...]

p_solution_matrix_final = p_solution_matrix[-1, ...]


# Post-processing
X, Y = np.meshgrid(x_array, y_array)

magnitude = np.sqrt(u_solution_matrix_final[::3, ::3]**2 + v_solution_matrix_final[::3, ::3]**2)
plt.quiver(X[::3, ::3], Y[::3, ::3], u_solution_matrix_final[::3, ::3], v_solution_matrix_final[::3, ::3], magnitude, cmap='plasma')
plt.colorbar(label='Velocity Magnitude')
plt.show()

fig, ax = plt.subplots()

U = u_solution_matrix[:, ::3, ::3]
V = v_solution_matrix[:, ::3, ::3]

M = np.sqrt(U**2 + V**2)

vmin = np.min(M)
vmax = np.max(M)
norm = Normalize(vmin=vmin, vmax=vmax)

levels = np.linspace(vmin, vmax, 11)

qvr = ax.quiver(
    X[::3, ::3], 
    Y[::3, ::3], 
    U[0], 
    V[0], 
    M[0], 
    cmap='plasma',
    norm=norm,
    angles='xy',
    scale_units='xy',
    scale=5,
    width=0.003)

cbar = fig.colorbar(qvr, ax=ax, ticks=levels, label='Velocity Magnitude')

ax.set_aspect("equal")

def update(frame):

    qvr.set_UVC(U[frame], V[frame], M[frame])

    ax.set_title(f'Time Step = {frame}')

    return qvr,

ani = FuncAnimation(fig, update, frames=u_solution_matrix.shape[0], interval=100, blit=False)

plt.show()

# fig, ax = plt.subplots()

# magnitude = np.sqrt(u_solution_matrix[0, ::3, ::3]**2 + v_solution_matrix[0, ::3, ::3]**2)
# levels = np.linspace(math.floor(np.min(u_solution_matrix)), math.ceil(np.max(u_solution_matrix)), 11)
# qvr = ax.quiver(X[::3, ::3], Y[::3, ::3], u_solution_matrix[0, ::3, ::3], v_solution_matrix[0, ::3, ::3], magnitude, cmap='plasma')

# def update(frame):

#     ax.clear()

#     magnitude = np.sqrt(u_solution_matrix[frame, ::3, ::3]**2 + v_solution_matrix[frame, ::3, ::3]**2)
#     ax.quiver(X[::3, ::3], Y[::3, ::3], u_solution_matrix[frame, ::3, ::3], v_solution_matrix[frame, ::3, ::3], magnitude, cmap='plasma')

#     ax.set_title(f'Time Step = {frame}')

# fig.colorbar(qvr, ax=ax, ticks=levels, label='Velocity Magnitude')

# ani = FuncAnimation(fig, update, frames=u_solution_matrix.shape[0], interval=100, blit=False)

# plt.show()

# show_cavity_flow_solution(
#     x_values=x_array,
#     y_values=y_array,
#     u_solution_matrix=u_solution_matrix_final,
#     v_solution_matrix=v_solution_matrix_final,
#     p_solution_matrix=p_solution_matrix_final,
#     case_name=case_name,
#     title=title,
#     save=save,
# )


# show_cavity_flow_solution_animation(
#     x_values=x_array,
#     y_values=y_array,
#     u_solution_history=u_solution_matrix,
#     v_solution_history=v_solution_matrix,
#     p_solution_history=p_solution_matrix,
#     case_name=case_name,
#     save=save,
# )
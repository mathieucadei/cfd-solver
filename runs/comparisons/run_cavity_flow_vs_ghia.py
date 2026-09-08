"""Run the 2D lid-driven cavity flow solver and generate solution plots."""



import os

from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from core import fdm

from post_processing import (
    show_cavity_flow_solution,
    show_cavity_flow_solution_animation,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 1.0
domain_length_y: float = 1.0
num_grid_points_x: int = 41
num_grid_points_y: int = 41
max_iterations = 10000
max_pseudo_iterations: int = 50
time_step: float = 0.001
u_lid: float = 1.0
density: float = 1.0
viscosity: float = 0.01


# Visualization parameters

step_stride = 10
case_name = 'cavity flow'
title = True
save = False
show_individual_plots = False


# Create the configuration object

cavity_flow_config = fdm.CavityFlowConfig(
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


# Generate the grid and time array

x_array = fdm.make_x_grid(cavity_flow_config)
y_array = fdm.make_y_grid(cavity_flow_config)


# Initialize the initial condition

initial_condition = fdm.cavity_flow_initial_condition(cavity_flow_config)



# Solve the poisson equation

solution_matrix = fdm.solve_cavity_flow(initial_condition, config=cavity_flow_config)

u_solution_matrix = solution_matrix[0]

v_solution_matrix = solution_matrix[1]

p_solution_matrix = solution_matrix[2]

u_solution_matrix_final = u_solution_matrix[-1, ...]

v_solution_matrix_final = v_solution_matrix[-1, ...]

p_solution_matrix_final = p_solution_matrix[-1, ...]


# Post-processing
X, Y = np.meshgrid(x_array, y_array)

u = u_solution_matrix_final[...,num_grid_points_x//2]
v = v_solution_matrix_final[num_grid_points_y//2,...]
M = np.sqrt(u_solution_matrix_final**2+v_solution_matrix_final**2)
p = p_solution_matrix_final

ghia_table_1 = pd.read_csv('../../data/ghia_table_1.csv')
ghia_table_2 = pd.read_csv('../../data/ghia_table_2.csv')

fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

ax0 = ax[0, 0].contourf(X, Y, p, alpha=0.5)
ax[0, 0].contour(X, Y, p, cmap=cm.viridis)
stream = ax[0, 0].streamplot(X, Y, u_solution_matrix_final, v_solution_matrix_final, color='k', linewidth=0.8)
stream.lines.set_alpha(0.5)
stream.arrows.set_alpha(0.5)
ax[0, 0].set_xlim(0, 1)
ax[0, 0].set_ylim(0, 1)
ax[0, 0].set_xlabel('x')
ax[0, 0].set_ylabel('y', rotation=0)
ax[0, 0].set_title('Pressure and Streamlines')
fig.colorbar(ax0, label='p')

ax1 = ax[0, 1].quiver(
    X[::2, ::2], 
    Y[::2, ::2], 
    u_solution_matrix_final[::2, ::2], 
    v_solution_matrix_final[::2, ::2],
    M[::2, ::2],
    scale=20,
    cmap=cm.plasma,
)
ax[0, 1].set_xlim(0, 1)
ax[0, 1].set_ylim(0, 1)
ax[0, 1].set_xlabel('x')
ax[0, 1].set_ylabel('y', rotation=0)
ax[0, 1].set_title('Velocity Field')
fig.colorbar(ax1, label='Velocity magnitude')

ax[1, 0].scatter(ghia_table_1['y'], ghia_table_1['100'], color='r', label='Ghia et al. (1982)')
ax[1, 0].plot(y_array, u, label='FDM')
ax[1, 0].grid(alpha=0.3)
ax[1, 0].set_xlim(0, 1)
ax[1, 0].set_xlabel('y')
ax[1, 0].set_ylabel('u', rotation=0)
ax[1, 0].legend()
ax[1, 0].set_title('u at x=0.5')

ax[1, 1].scatter(ghia_table_2['x'], ghia_table_2['100'], color='r', label='Ghia et al. (1982)')
ax[1, 1].plot(x_array, v, label='FDM')
ax[1, 1].grid(alpha=0.3)
ax[1, 1].set_xlim(0, 1)
ax[1, 1].set_xlabel('x')
ax[1, 1].set_ylabel('v', rotation=0)
ax[1, 1].legend()
ax[1, 1].set_title('v at y=0.5')

fig.suptitle('Lid-Driven Cavity Flow — Re = 100')

plt.show()
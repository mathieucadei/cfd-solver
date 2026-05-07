"""Run the 2D diffusion solver and generate solution plots."""



import os

from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

from core import (
    CavityFlowConfig,
    cavity_flow_initial_condition,
    make_x_grid,
    make_y_grid,
    solve_cavity_flow,
)

from post_processing import (
    show_solution_2d_animation,
    show_solution_contour,
    show_solution_overview,
    show_solution_surface,
    show_solution_traces,
    show_cavity_flow_solution,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2.0
domain_length_y: float = 1.0
num_grid_points_x: int = 41
num_grid_points_y: int = 41
max_iterations: int = 500
max_pseudo_iterations: int = 50
time_step: float = 0.001
u_lid: float = 1.0
density: float = 1.0
viscosity: float = 0.1


# Visualization parameters

step_stride = 10
case_name = '2d cavity flow'
title = True
save = False
show_individual_plots = False


# Create the configuration object

cavity_flow_config = CavityFlowConfig(
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

x_array = make_x_grid(cavity_flow_config)
y_array = make_y_grid(cavity_flow_config)


# Initialize the initial condition

initial_condition = cavity_flow_initial_condition(cavity_flow_config)



# Solve the poisson equation

solution_matrix = solve_cavity_flow(initial_condition, config=cavity_flow_config)

u_solution_matrix = solution_matrix[0]

v_solution_matrix = solution_matrix[1]

p_solution_matrix = solution_matrix[2]

u_solution_matrix_final = u_solution_matrix[-1, ...]

v_solution_matrix_final = v_solution_matrix[-1, ...]

p_solution_matrix_final = p_solution_matrix[-1, ...]


# Post-processing

show_cavity_flow_solution(
    x_values=x_array,
    y_values=y_array,
    u_solution_matrix=u_solution_matrix_final,
    v_solution_matrix=v_solution_matrix_final,
    p_solution_matrix=p_solution_matrix_final,
    case_name=case_name,
    title=title,
    save=save,
)

# X, Y = np.meshgrid(x_array, y_array)

# # plt.figure(figsize=(8, 4))
# # plt.contourf(X, Y, p_solution_matrix_final, cmap='viridis')
# # plt.colorbar(label='Pressure')
# # plt.contour(X, Y, p_solution_matrix_final, colors='k', linewidths=0.5)
# # plt.quiver(X[::2, ::2], Y[::2, ::2], u_solution_matrix_final[::2, ::2], v_solution_matrix_final[::2, ::2], scale=20)
# # plt.title('Cavity Flow: Pressure Contours and Velocity Vectors')
# # plt.show()

# fig, ax = plt.subplots(figsize=(8, 4))
# contour = ax.contourf(X, Y, p_solution_matrix_final, cmap='viridis')
# fig.colorbar(contour, ax=ax, label='Pressure')
# ax.contour(X, Y, p_solution_matrix_final, colors='k', linewidths=0.5)
# ax.quiver(X[::2, ::2], Y[::2, ::2], u_solution_matrix[0, ::2, ::2], v_solution_matrix[0, ::2, ::2], scale=20)
# ax.set_title('Cavity Flow: Pressure Contours and Velocity Vectors')

# def update(frame):
#     ax.clear()
#     ax.contourf(X, Y, p_solution_matrix[frame], cmap='viridis')
#     ax.contour(X, Y, p_solution_matrix[frame], colors='k', linewidths=0.5)
#     ax.quiver(X[::2, ::2], Y[::2, ::2], u_solution_matrix[frame, ::2, ::2], v_solution_matrix[frame, ::2, ::2], scale=20)
    
#     ax.set_xlim(0, cavity_flow_config.domain_length_x)
#     ax.set_ylim(0, cavity_flow_config.domain_length_y)
    
#     ax.set_title(f'Cavity Flow: Pressure Contours and Velocity Vectors (Iteration {frame})')

# ani = FuncAnimation(fig, update, frames=range(0, max_iterations + 1, step_stride), repeat=False)

# plt.show()

# ani.save(f'results/animations/2d/cavity_flow_solution.mp4', writer='ffmpeg')


# def update(frame):
#     plt.clf()
#     # plt.contourf(X, Y, p_solution_matrix[frame], cmap='viridis')
#     # plt.colorbar(label='Pressure')
#     # plt.contour(X, Y, p_solution_matrix[frame], colors='k', linewidths=0.5)
#     plt.quiver(X[::2, ::2], Y[::2, ::2], u_solution_matrix[frame][::2, ::2], v_solution_matrix[frame][::2, ::2], scale=20)
#     # plt.title(f'Cavity Flow: Pressure Contours and Velocity Vectors (Iteration {frame})')

# anim = FuncAnimation(plt.gcf(), update, frames=range(0, max_iterations + 1, step_stride), repeat=False)

# show_solution_contour(
#     x_values=x_array,
#     y_values=y_array,
#     solution_matrix=p_solution_matrix_final,
#     case_name=case_name,
#     title=title,
#     y_label='y',
#     save=save,
# )

# if show_individual_plots:
#     show_solution_traces(
#         x_values=x_array,
#         cut_values=y_array,
#         num_solution_matrix=solution_final_x,
#         step_stride=step_stride,
#         cut_label='y',
#         case_name=case_name,
#         title=title,
#         save=save,
#     )

#     show_solution_traces(
#         x_values=y_array,
#         cut_values=x_array,
#         num_solution_matrix=solution_final_y,
#         step_stride=step_stride,
#         cut_label='x',
#         case_name=case_name,
#         title=title,
#         x_label='y',
#         save=save,
#     )

#     show_solution_contour(
#         x_values=x_array,
#         y_values=y_array,
#         solution_matrix=solution_final,
#         case_name=case_name,
#         title=title,
#         y_label='y',
#         save=save,
#     )

#     show_solution_surface(
#         x_values=x_array,
#         y_values=y_array,
#         solution_matrix=solution_final,
#         case_name=case_name,
#         title=title,
#         y_label='y',
#         save=save,
#     )

# show_solution_overview(
#     x_values=x_array, 
#     y_values=y_array, 
#     num_solution_matrix=solution_final,
#     y_label='y',
#     step_stride=step_stride,
#     case_name=case_name,
#     title=title,
#     save=save,
# )

# show_solution_2d_animation(
#     x_values=x_array,
#     y_values=y_array, 
#     solution_history=solution_matrix,
#     z_limits=(np.min(solution_final), np.max(solution_final)),
#     case_name=case_name,
#     save=save,
# )
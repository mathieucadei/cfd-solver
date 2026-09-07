"""Run the 2D diffusion solver and generate solution plots."""



import numpy as np
import matplotlib.pyplot as plt

from core import fvm

from post_processing import (
    show_solution_2d_animation,
    show_solution_overview,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2.0
domain_length_y: float = 2.0
num_cells_x: int = 30
num_cells_y: int = 30
expansion_ratio_x = 0.
expansion_ratio_y = 0.
max_iterations: int = 50
sigma: float = 0.25
viscosity: float = 0.05
hat_start_x: float = 0.5
hat_start_y: float = 0.5
hat_end_x: float = 1.0
hat_end_y: float = 1.0
u_min: float = 1.0
u_max: float = 2.0


# Visualization parameters

step_stride = 10
case_name = '2d diffusion'
title = True
save = False
show_individual_plots = False


# Create the configuration object

diffusion_2d_config = fvm.Diffusion2DConfig(
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
)


# Generate the grid and time array

hx_array, hy_array = fvm.build_h_spacing(diffusion_2d_config)
xc_array, yc_array = fvm.build_centers(diffusion_2d_config)
time_array = np.arange(0, diffusion_2d_config.max_iterations + 1)


# Initialize the initial condition

initial_condition = fvm.hat_initial_condition_2d(diffusion_2d_config)



# Solve the advection equation

solution_matrix = fvm.solve_diffusion_2d(initial_condition, diffusion_2d_config)

solution_final = solution_matrix[-1, ...]

solution_final_x = solution_matrix[-1, :, :]

solution_final_y = solution_final_x.T

xf, yf = fvm.build_face_positions(diffusion_2d_config)

# Post-processing

fig, ax = plt.subplots(figsize=(12,6))

ax = ax.pcolormesh(xf, yf, solution_final[:, :], edgecolors='k', linewidth=0.3)
fig.colorbar(ax, label='u')

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
    z_limits=(u_min, u_max),
    case_name=case_name,
    save=save,
)
"""Run the 1D diffusion solver and generate solution plots."""



import numpy as np
import matplotlib.pyplot as plt

from core import fvm

from post_processing import (
    show_solution_1d_animation,
    show_solution_contour_map,
    show_solution_overview,
    show_solution_surface,
    show_solution_traces,
)



# Pre-processing
# Simulation parameters

domain_length_x = 2.0
num_cells_x = 80
expansion_ratio_x = 0.
max_iterations = 25
sigma = 0.2
viscosity = 0.3
hat_start = 0.5
hat_end = 1.0
u_min = 1.0
u_max = 2.0


# Visualization parameters

step_stride = 20
case_name = '1d diffusion'
title = True
save = False
show_individual_plots = False


# Create the configuration object

diffusion_1d_config = fvm.Diffusion1DConfig(
    domain_length_x=domain_length_x,
    num_cells_x=num_cells_x,
    expansion_ratio_x=expansion_ratio_x,
    max_iterations=max_iterations,
    sigma=sigma,
    viscosity=viscosity,
    hat_start=hat_start,
    hat_end=hat_end,
    u_min=u_min,
    u_max=u_max,
)


# Generate the grid and time array

hx_array = fvm.build_hx_spacing(diffusion_1d_config)
xc_array = fvm.build_x_centers(diffusion_1d_config)
time_array = np.arange(0, diffusion_1d_config.max_iterations + 1)


# Initialize the initial condition

initial_condition = fvm.hat_initial_condition_1d(hx_array, diffusion_1d_config)



# Solve the diffusion equation

solution_history = fvm.solve_diffusion_1d(initial_condition, diffusion_1d_config)



solution_final = solution_history[-1]

xf = fvm.build_x_face_positions(diffusion_1d_config)



# Post-processing

fig, ax = plt.subplots(figsize=(10,3))
pc = ax.pcolormesh(xf, [0, 1], solution_final[None, :], edgecolors='k', linewidth=0.3)
fig.colorbar(pc, label='u')
plt.show()

show_solution_overview(
    x_values=xc_array, 
    y_values=time_array, 
    num_solution_matrix=solution_history, 
    step_stride=step_stride,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_1d_animation(
    x_values=xc_array,
    num_solution_history=solution_history,
    case_name=case_name,
    save=save,
)
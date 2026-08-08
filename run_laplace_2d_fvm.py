"""Run the 2D diffusion solver and generate solution plots."""



import numpy as np
import matplotlib.pyplot as plt

from core import (
    Laplace2DFVMConfig,
    laplace_initial_condition_2d_fvm,
    build_h_spacing,
    build_face_positions,
    build_centers,
    solve_laplace_2d_fvm,
)
from post_processing import (
    show_solution_2d_animation,
    show_solution_overview,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2.0
domain_length_y: float = 1.0
num_cells_x: int = 30
num_cells_y: int = 30
expansion_ratio_x: float = 0.
expansion_ratio_y: float = 0.
l1_norm_target: float = 1e-4

# Visualization parameters

step_stride = 10
case_name = '2d laplace'
title = True
save = False
show_individual_plots = False


# Create the configuration object

laplace_2d_config = Laplace2DFVMConfig(
    domain_length_x=domain_length_x,
    domain_length_y=domain_length_y,
    num_cells_x=num_cells_x,
    num_cells_y=num_cells_y,
    l1_norm_target=l1_norm_target,
)


# Generate the grid and time array

hx_array, hy_array = build_h_spacing(laplace_2d_config)
xc_array, yc_array = build_centers(laplace_2d_config)


# Initialize the initial condition

initial_condition = laplace_initial_condition_2d_fvm(laplace_2d_config)
bottom_boundary = 0
top_boundary = 0 
right_boundary = yc_array
left_boundary = 0



# Solve the advection equation

solution_matrix = solve_laplace_2d_fvm(initial_condition, bottom_boundary=bottom_boundary, top_boundary=top_boundary, right_boundary=right_boundary, left_boundary=left_boundary, config=laplace_2d_config)

# print(type(solution_matrix))

solution_final = solution_matrix[-1, ...]

solution_final_x = solution_matrix[-1, :, :]

solution_final_y = solution_final_x.T



# Post-processing

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
    z_limits=(np.min(solution_final), np.max(solution_final)),
    case_name=case_name,
    save=save,
)
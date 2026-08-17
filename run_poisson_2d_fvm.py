"""Run the 2D diffusion solver and generate solution plots."""



import numpy as np
import matplotlib.pyplot as plt

from core import (
    SourceTermFVM,
    Poisson2DFVMConfig,
    poisson_initial_condition_2d_fvm,
    build_h_spacing,
    build_face_positions,
    build_centers,
    solve_poisson_2d_fvm,
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
max_iterations: int = 100
pressure_init: float = 0.0
source_terms=[
    SourceTermFVM(x=0.25, y=0.25, value=100.0),
    SourceTermFVM(x=0.75, y=0.75, value=-100.0),
]
l1_norm_target: float = 1e-4


# Visualization parameters

step_stride = 10
case_name = '2d poisson'
title = True
save = False
show_individual_plots = False


# Create the configuration object

poisson_2d_config = Poisson2DFVMConfig(
    domain_length_x=domain_length_x,
    domain_length_y=domain_length_y,
    num_cells_x=num_cells_x,
    num_cells_y=num_cells_y,
    expansion_ratio_x=expansion_ratio_x,
    expansion_ratio_y=expansion_ratio_y,
    max_iterations=max_iterations,
    pressure_init=pressure_init,
    source_terms=source_terms,
    l1_norm_target=l1_norm_target,
)


# Generate the grid and time array

hx_array, hy_array = build_h_spacing(poisson_2d_config)
xc_array, yc_array = build_centers(poisson_2d_config)

# Initialize the initial condition

initial_condition = poisson_initial_condition_2d_fvm(poisson_2d_config)



# Solve the poisson equation

solution_matrix = solve_poisson_2d_fvm(initial_condition, config=poisson_2d_config)

# print(type(solution_matrix))

solution_final = solution_matrix[-1, ...]

solution_final_x = solution_matrix[-1, :, :]

solution_final_y = solution_final_x.T

xf, yf = build_face_positions(poisson_2d_config)

# Post-processing

fig, ax = plt.subplots(figsize=(12,6))

ax = ax.pcolormesh(xf, yf, solution_final[:, :], edgecolors='k', linewidth=0.3)
fig.colorbar(ax, label='p')

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
    z_limits=(np.min(solution_final), np.max(solution_final)),
    case_name=case_name,
    save=save,
)
"""Run the 2D diffusion solver and generate solution plots."""



import numpy as np

from core import (
    SourceTerm,
    Poisson2DConfig,
    poisson_initial_condition_2d,
    make_x_grid,
    make_y_grid,
    solve_laplace_2d,
)
from core.numerics.poisson_2d import solve_poisson_2d
from post_processing import (
    show_solution_2d_animation,
    show_solution_contour,
    show_solution_overview,
    show_solution_surface,
    show_solution_traces,
)



# Pre-processing
# Simulation parameters

domain_length_x: float = 2.0
domain_length_y: float = 1.0
num_grid_points_x: int = 31
num_grid_points_y: int = 31
max_iterations: int = 100
pressure_init: float = 0.0
source_terms=[
    SourceTerm(x=0.25, y=0.25, value=100.0),
    SourceTerm(x=0.75, y=0.75, value=-100.0),
]
l1_norm_target: float = 1e-4


# Visualization parameters

step_stride = 10
case_name = '2d poisson'
title = True
save = False
show_individual_plots = False


# Create the configuration object

poisson_2d_config = Poisson2DConfig(
    domain_length_x=domain_length_x,
    domain_length_y=domain_length_y,
    num_grid_points_x=num_grid_points_x,
    num_grid_points_y=num_grid_points_y,
    max_iterations=max_iterations,
    pressure_init=pressure_init,
    source_terms=source_terms,
    l1_norm_target=l1_norm_target,
)


# Generate the grid and time array

x_array = make_x_grid(poisson_2d_config)
y_array = make_y_grid(poisson_2d_config)


# Initialize the initial condition

initial_condition = poisson_initial_condition_2d(poisson_2d_config)



# Solve the poisson equation

solution_matrix = solve_poisson_2d(initial_condition, config=poisson_2d_config)

# print(type(solution_matrix))

solution_final = solution_matrix[-1, ...]

solution_final_x = solution_matrix[-1, :, :]

solution_final_y = solution_final_x.T



# Post-processing

if show_individual_plots:
    show_solution_traces(
        x_values=x_array,
        cut_values=y_array,
        num_solution_matrix=solution_final_x,
        step_stride=step_stride,
        cut_label='y',
        case_name=case_name,
        title=title,
        save=save,
    )

    show_solution_traces(
        x_values=y_array,
        cut_values=x_array,
        num_solution_matrix=solution_final_y,
        step_stride=step_stride,
        cut_label='x',
        case_name=case_name,
        title=title,
        x_label='y',
        save=save,
    )

    show_solution_contour(
        x_values=x_array,
        y_values=y_array,
        solution_matrix=solution_final,
        case_name=case_name,
        title=title,
        y_label='y',
        save=save,
    )

    show_solution_surface(
        x_values=x_array,
        y_values=y_array,
        solution_matrix=solution_final,
        case_name=case_name,
        title=title,
        y_label='y',
        save=save,
    )

show_solution_overview(
    x_values=x_array, 
    y_values=y_array, 
    num_solution_matrix=solution_final,
    y_label='y',
    step_stride=step_stride,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_2d_animation(
    x_values=x_array,
    y_values=y_array, 
    solution_history=solution_matrix,
    z_limits=(np.min(solution_final), np.max(solution_final)),
    case_name=case_name,
    save=save,
)
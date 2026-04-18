"""Run the 1D Burgers' & Cole-Hopf solvers and generate solution & comparision plots."""



import numpy as np

from core import (
    BurgersEquation1DConfig,
    cole_hopf_initial_condition,
    make_cole_hopf_1d_grid,
    solve_burgers_equation_1d,
    solve_cole_hopf_1d,
)
from post_processing import (
    show_solution_1d_animation,
    show_solution_contour,
    show_solution_overview,
    show_solution_surface,
    show_solution_traces,
)



# Pre-processing
# Simulation parameters

domain_length = 6.0
num_grid_points = 101
max_iterations = 100
time_step = 0.0025
grid_type: str = "cole_hopf"
sigma = 0.02
viscosity = 0.07
hat_start = 0.5
hat_end = 1.0
u_min = 1.0
u_max = 2.0


# Visualization parameters

step_stride = 20
case_name = '1d burgers vs cole-hopf'
title = True
save = False


# Create the configuration object

burgers_1d_config = BurgersEquation1DConfig(
    domain_length=domain_length,
    num_grid_points=num_grid_points,
    max_iterations=max_iterations,
    time_step=time_step,
    grid_type=grid_type,
    sigma=sigma,
    viscosity=viscosity,
    hat_start=hat_start,
    hat_end=hat_end,
    u_min=u_min,
    u_max=u_max,
)


# Generate the grid and time array

x_array = make_cole_hopf_1d_grid(burgers_1d_config)
time_array = np.arange(0, burgers_1d_config.max_iterations + 1)

# Initialize the initial condition

initial_condition = cole_hopf_initial_condition(x_array, burgers_1d_config)



# Solve
# Numerical Burgers' equation

solution_history_num = solve_burgers_equation_1d(initial_condition, burgers_1d_config)


# Analytical Cole-Hopf equation

solution_history_ana = solve_cole_hopf_1d(x_array, burgers_1d_config)



# Post-processing

show_solution_traces(
    x_values=x_array,
    cut_values=time_array,
    num_solution_matrix=solution_history_num,
    ana_solution_matrix=solution_history_ana,
    step_stride=step_stride,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_traces(
    x_values=time_array,
    cut_values=x_array,
    num_solution_matrix=solution_history_num,
    axis=1,
    ana_solution_matrix=solution_history_ana,
    step_stride=step_stride,
    cut_label='x',
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_contour(
    x_values=x_array,
    y_values=time_array,
    solution_matrix=solution_history_num,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_surface(
    x_values=x_array,
    y_values=time_array,
    solution_matrix=solution_history_num,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_overview(
    x_values=x_array, 
    y_values=time_array, 
    num_solution_matrix=solution_history_num, 
    ana_solution_matrix=solution_history_ana, 
    step_stride=step_stride,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_1d_animation(
    x_values=x_array,
    num_solution_history=solution_history_num,
    ana_solution_history=solution_history_ana, 
    case_name=case_name,
    save=save,
)
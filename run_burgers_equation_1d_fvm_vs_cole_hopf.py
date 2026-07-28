"""Run the 1D Burgers' & Cole-Hopf solvers and generate solution & comparision plots."""



import numpy as np
import matplotlib.pyplot as plt

from core import (
    BurgersEquation1DFVMConfig,
    cole_hopf_initial_condition_1d,
    build_hx_spacing,
    build_x_face_positions,
    build_x_centers,
    make_cole_hopf_x_mesh,
    solve_burgers_equation_1d_fvm,
    solve_cole_hopf_1d_fvm,
)

from post_processing import (
    show_solution_1d_animation,
    show_solution_overview,
)



# Pre-processing
# Simulation parameters

domain_length_x = 6.0
num_cells_x = 200
expansion_ratio_x = 0.
max_iterations = 100
time_step = 0.0025
grid_type = "cole_hopf"
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
show_individual_plots = False


# Create the configuration object

burgers_1d_config = BurgersEquation1DFVMConfig(
    domain_length_x=domain_length_x,
    num_cells_x=num_cells_x,
    expansion_ratio_x=expansion_ratio_x,
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

hx_array = build_hx_spacing(burgers_1d_config)
xc_array = build_x_centers(burgers_1d_config)
time_array = np.arange(0, burgers_1d_config.max_iterations + 1)

# Initialize the initial condition

initial_condition = cole_hopf_initial_condition_1d(xc_array, burgers_1d_config)



# Solve
# Numerical Burgers' equation

solution_history_num = solve_burgers_equation_1d_fvm(initial_condition, burgers_1d_config)

solution_final = solution_history_num[-1]

xf = build_x_face_positions(burgers_1d_config)

# Analytical Cole-Hopf equation

solution_history_ana = solve_cole_hopf_1d_fvm(xc_array, burgers_1d_config)



# Post-processing

fig, ax = plt.subplots(figsize=(10,3))
pc = ax.pcolormesh(xf, [0, 1], solution_final[None, :], edgecolors='k', linewidth=0.3)
fig.colorbar(pc, label='u')
plt.show()


show_solution_overview(
    x_values=xc_array, 
    y_values=time_array, 
    num_solution_matrix=solution_history_num, 
    ana_solution_matrix=solution_history_ana, 
    step_stride=step_stride,
    case_name=case_name,
    title=title,
    save=save,
)

show_solution_1d_animation(
    x_values=xc_array,
    num_solution_history=solution_history_num,
    ana_solution_history=solution_history_ana, 
    case_name=case_name,
    save=save,
)
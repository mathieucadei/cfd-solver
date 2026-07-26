"""Run the 1D diffusion solver and generate solution plots."""



import numpy as np
import matplotlib.pyplot as plt

from core import (
    Diffusion1DFVMConfig,
    Diffusion1DConfig,
    hat_initial_condition_1d_fvm,
    build_hx_spacing,
    build_x_face_positions,
    build_x_centers,
    solve_diffusion_1d_fvm,
    make_x_grid,
    compute_coefficients,
    compute_diffusive_dt_1d,
    compute_series_terms,
    generate_mode_indices,
    solve_heat_equation_1d,
    compute_diffusive_dt_1d_fvm,
)

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
num_cells_x = 100
expansion_ratio_x = 0.
max_iterations = 25
sigma = 0.2
viscosity = 0.3
hat_start = 0.5
hat_end = 1.0
u_min = 1.0
u_max = 2.0


# Analytical simulation parameters

num_grid_points_x = num_cells_x
num_modes = 100
basis = "cosine"  # "periodic" or "cosine"


# Visualization parameters

step_stride = 100
case_name = '1d diffusion vs heat'
title = True
save = False
show_individual_plots = False


# Create the configuration object

diffusion_1d_config = Diffusion1DFVMConfig(
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

# Create the configuration object

diffusion_1d_config_ana = Diffusion1DConfig(
    domain_length_x=domain_length_x,
    num_grid_points_x=num_grid_points_x,
    max_iterations=max_iterations,
    sigma=sigma,
    viscosity=viscosity,
    hat_start=hat_start,
    hat_end=hat_end,
    u_min=u_min,
    u_max=u_max,
)


# Generate the grid and time array

hx_array = build_hx_spacing(diffusion_1d_config)
xc_array = build_x_centers(diffusion_1d_config)
time_array = np.arange(0, diffusion_1d_config.max_iterations + 1)

x_array = make_x_grid(diffusion_1d_config_ana)
dt = compute_diffusive_dt_1d(diffusion_1d_config_ana)
time_array = np.arange(0, max_iterations + 1) * dt


# Initialize the initial condition

initial_condition = hat_initial_condition_1d_fvm(hx_array, diffusion_1d_config)

initial_condition_ana = hat_initial_condition_1d_fvm(x_array, diffusion_1d_config_ana)


# Fourier-series setup

mode_indices = generate_mode_indices(num_modes)

mode_coefficients = compute_coefficients(
    initial_condition_ana, 
    x_array, 
    mode_indices, 
    basis=basis,
)

series_terms = compute_series_terms(mode_indices, mode_coefficients, x_array, basis=basis)


# Solve the diffusion equation

solution_history_num = solve_diffusion_1d_fvm(initial_condition, diffusion_1d_config)


# Heat analytical equation

solution_history_ana = solve_heat_equation_1d(
    series_terms, 
    mode_indices,
    x_array,
    time_array, 
    diffusion_1d_config_ana.viscosity,
    basis=basis)


solution_final = solution_history_num[-1]

xf = build_x_face_positions(diffusion_1d_config)



# Post-processing

fig, ax = plt.subplots(figsize=(10,3))
pc = ax.pcolormesh(xf, [0, 1], solution_final[None, :], edgecolors='k', linewidth=0.3)
fig.colorbar(pc, label='u')
plt.show()


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
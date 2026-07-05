"""Run the 1D convection solver and generate solution plots."""



import matplotlib.pyplot as plt

from core import (
    Convection1DConfig,
    heaviside_initial_condition_1d,
    make_x_grid,
    solve_convection_1d,
)


# Pre-processing
# Simulation parameters

domain_length_x = 4.0
step_location  = 2
epsilon = 0.1
u_min = 0
u_max = 1.0
schemes = [
    'conservative-upwind',
    'conservative-lax-friedrichs',
    'conservative-richtmyer',
    '1-step-conservative-lax-wendroff',
    '2-step-conservative-lax-wendroff',
    'conservative-mac-cormack',
    'conservative-implicit-beam-warming',
    'conservative-damped-implicit-beam-warming',
]


# Visualization parameters

fig, ax = plt.subplots(4, 2, figsize=(12, 10), sharey=True)


# Comparison cases parameters

cases = [
    (ax[0, 0], 81, 40, 1.0, 0.05),
    (ax[0, 1], 81, 80, 0.5, 0.05),
    (ax[1, 0], 81, 40, 1.0, 0.1),
    (ax[1, 1], 81, 80, 0.5, 0.1),
    (ax[2, 0], 81, 40, 1.0, 0.125),
    (ax[2, 1], 81, 80, 0.5, 0.125),
    (ax[3, 0], 81, 40, 1.0, 0.1275),
    (ax[3, 1], 81, 80, 0.5, 0.1275),
]


# Comparison loop

for current_ax, nx, n_iter, sigma, epsilon in cases:

    for scheme in schemes:


        # Create the configuration object

        convection_1d_config = Convection1DConfig(
            domain_length_x=domain_length_x,
            num_grid_points_x=nx,
            max_iterations=n_iter,
            sigma=sigma,
            epsilon=epsilon,
            hat_start=step_location ,
            u_min=u_min,
            u_max=u_max,
            scheme=scheme,
        )


        # Generate the grid

        x_array = make_x_grid(convection_1d_config)


        # Initialize the initial condition

        initial_condition = heaviside_initial_condition_1d(x_array, convection_1d_config)


        # Solve the convection equation

        solution_history = solve_convection_1d(initial_condition, convection_1d_config)


        # Post-processing

        base_scheme = scheme.replace('conservative-', '')

        current_ax.plot(
            x_array,
            solution_history[-1],
            label=scheme,
        )

    current_ax.set_xticks(x_array, minor=True)
    current_ax.set_xticks(x_array[::5] if nx == 41 else x_array[::10])
    current_ax.grid(True, which='minor', alpha=0.15)
    current_ax.grid(True, which='major', alpha=0.4)
    # current_ax.legend()
    current_ax.set_xlabel('x')
    current_ax.set_ylabel('u', rotation=0)

    final_time = n_iter * sigma * domain_length_x / ((nx - 1) * u_max)
    
    current_ax.set_title(
        f'nx={nx}, sigma={sigma}, epsilon={epsilon} steps={n_iter}, t={final_time:.2f}'
    )
    current_ax.tick_params(labelleft=True)


handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5)

fig.suptitle('1D Convection Scheme Comparison: Heaviside Step', y=0.98)
fig.tight_layout(rect=[0, 0.10, 1, 0.96])
plt.show()
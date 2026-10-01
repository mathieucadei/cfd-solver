import numpy as np
import matplotlib.pyplot as plt

from core import fvm, analytical, signal_processing


def test_diffusion_2d_matches_heat():

    config = fvm.Diffusion2DConfig(
        domain_length_x=2.0,
        domain_length_y=2.0,
        num_cells_x=100,
        num_cells_y=100,
        expansion_ratio_x=0.,
        expansion_ratio_y=0.,
        max_iterations=50,
        sigma=0.25,
        viscosity=0.05,
        hat_start_x=0.5,
        hat_start_y=0.5,
        hat_end_x=1.0,
        hat_end_y=1.0,
        u_min=1.0,
        u_max=2.0,
    )

    hx_array, hy_array = fvm.build_h_spacing(config)
    xc_array, yc_array = fvm.build_centers(config)
    dt = fvm.compute_diffusive_dt_2d(config)
    time_array = np.arange(0, config.max_iterations + 1) * dt

    initial_condition = fvm.hat_initial_condition_2d(config)

    config_x = fvm.Diffusion1DConfig(
        domain_length_x=config.domain_length_x,
        num_cells_x=config.num_cells_x,
        expansion_ratio_x=config.expansion_ratio_x,
        max_iterations=1,
        sigma=config.sigma,
        viscosity=config.viscosity,
        hat_start=config.hat_start_x,
        hat_end=config.hat_end_x,
        u_min=0.0,
        u_max=1.0,
    )
    initial_condition_x = fvm.hat_initial_condition_1d(hx_array, config_x)

    config_y = fvm.Diffusion1DConfig(
        domain_length_x=config.domain_length_y,
        num_cells_x=config.num_cells_y,
        expansion_ratio_x=config.expansion_ratio_y,
        max_iterations=1,
        sigma=config.sigma,
        viscosity=config.viscosity,
        hat_start=config.hat_start_y,
        hat_end=config.hat_end_y,
        u_min=0.0,
        u_max=1.0,
    )
    initial_condition_y = fvm.hat_initial_condition_1d(hy_array, config_y)

    u_numerical = fvm.solve_diffusion_2d(initial_condition, config)[-1]

    # x
    num_modes = 100
    basis = "cosine"  # "periodic" or "cosine"

    mode_indices = signal_processing.generate_mode_indices(num_modes)

    mode_coefficients_x = signal_processing.compute_coefficients(
        initial_condition_x,
        xc_array,
        mode_indices,
        basis=basis,
    )

    mode_coefficients_y = signal_processing.compute_coefficients(
        initial_condition_y,
        yc_array,
        mode_indices,
        basis=basis,
    )

    series_terms_x = signal_processing.compute_series_terms(mode_indices, mode_coefficients_x, xc_array, basis=basis)
    series_terms_y = signal_processing.compute_series_terms(mode_indices, mode_coefficients_y, yc_array, basis=basis)

    u_analytical_x = analytical.solve_heat_equation_1d(
        series_terms_x,
        mode_indices,
        xc_array,
        time_array,
        config.viscosity,
        basis=basis)[-1]

    u_analytical_y = analytical.solve_heat_equation_1d(
        series_terms_y,
        mode_indices,
        yc_array,
        time_array,
        config.viscosity,
        basis=basis)[-1]

    u_analytical = config.u_min + (config.u_max - config.u_min) * u_analytical_y[:, None] * u_analytical_x[None, :]

    peak = u_numerical.max()
    exact = u_analytical.max()

    assert abs(peak - exact) < 1e-3


# if __name__ == '__main__':

#     config = fvm.Diffusion2DConfig(
#         domain_length_x=2.0,
#         domain_length_y=2.0,
#         num_cells_x=30,
#         num_cells_y=30,
#         expansion_ratio_x=0.,
#         expansion_ratio_y=0.,
#         max_iterations=50,
#         sigma=0.25,
#         viscosity=0.05,
#         hat_start_x=0.5,
#         hat_start_y=0.5,
#         hat_end_x=1.0,
#         hat_end_y=1.0,
#         u_min=1.0,
#         u_max=2.0,
#     )

#     hx_array, hy_array = fvm.build_h_spacing(config)
#     xc_array, yc_array = fvm.build_centers(config)
#     dt = fvm.compute_diffusive_dt_2d(config)
#     time_array = np.arange(0, config.max_iterations + 1) * dt

#     initial_condition = fvm.hat_initial_condition_2d(config)

#     config_x = fvm.Diffusion1DConfig(
#         domain_length_x=config.domain_length_x,
#         num_cells_x=config.num_cells_x,
#         expansion_ratio_x=config.expansion_ratio_x,
#         max_iterations=1,
#         sigma=config.sigma,
#         viscosity=config.viscosity,
#         hat_start=config.hat_start_x,
#         hat_end=config.hat_end_x,
#         u_min=0.0,
#         u_max=1.0,
#     )
#     initial_condition_x = fvm.hat_initial_condition_1d(hx_array, config_x)

#     config_y = fvm.Diffusion1DConfig(
#         domain_length_x=config.domain_length_y,
#         num_cells_x=config.num_cells_y,
#         expansion_ratio_x=config.expansion_ratio_y,
#         max_iterations=1,
#         sigma=config.sigma,
#         viscosity=config.viscosity,
#         hat_start=config.hat_start_y,
#         hat_end=config.hat_end_y,
#         u_min=0.0,
#         u_max=1.0,
#     )
#     initial_condition_y = fvm.hat_initial_condition_1d(hy_array, config_y)

#     u_numerical = fvm.solve_diffusion_2d(initial_condition, config)[-1]

#     # x
#     num_modes = 100
#     basis = "cosine"  # "periodic" or "cosine"

#     mode_indices = signal_processing.generate_mode_indices(num_modes)

#     mode_coefficients_x = signal_processing.compute_coefficients(
#         initial_condition_x,
#         xc_array,
#         mode_indices,
#         basis=basis,
#     )

#     mode_coefficients_y = signal_processing.compute_coefficients(
#         initial_condition_y,
#         yc_array,
#         mode_indices,
#         basis=basis,
#     )

#     series_terms_x = signal_processing.compute_series_terms(mode_indices, mode_coefficients_x, xc_array, basis=basis)
#     series_terms_y = signal_processing.compute_series_terms(mode_indices, mode_coefficients_y, yc_array, basis=basis)

#     u_analytical_x = analytical.solve_heat_equation_1d(
#         series_terms_x,
#         mode_indices,
#         xc_array,
#         time_array,
#         config.viscosity,
#         basis=basis)[-1]

#     u_analytical_y = analytical.solve_heat_equation_1d(
#         series_terms_y,
#         mode_indices,
#         yc_array,
#         time_array,
#         config.viscosity,
#         basis=basis)[-1]

#     u_analytical = config.u_min + (config.u_max - config.u_min) * u_analytical_y[:, None] * u_analytical_x[None, :]

#     X, Y = np.meshgrid(xc_array, yc_array)

#     fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

#     ax[0].plot_surface(X, Y, u_numerical, cmap='plasma')
#     ax[1].plot_surface(X, Y, u_analytical, cmap='plasma')
#     plt.show()
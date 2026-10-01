import numpy as np
import matplotlib.pyplot as plt

from core import fdm, analytical, signal_processing


def test_diffusion_2d_matches_heat():
    config = fdm.Diffusion2DConfig(
        domain_length_x=2.0,
        domain_length_y=2.0,
        num_grid_points_x=101,
        num_grid_points_y=101,
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

    x_array = fdm.make_x_grid(config)
    y_array = fdm.make_y_grid(config)
    dt = fdm.compute_diffusive_dt_2d(config)
    time_array = np.arange(0, config.max_iterations + 1) * dt

    initial_condition = fdm.hat_initial_condition_2d(config)

    config_x = fdm.Diffusion1DConfig(
        domain_length_x=config.domain_length_x,
        num_grid_points_x=config.num_grid_points_x,
        max_iterations=1,
        sigma=config.sigma,
        viscosity=config.viscosity,
        hat_start=config.hat_start_x,
        hat_end=config.hat_end_x,
        u_min=0.0,
        u_max=1.0,
    )
    initial_condition_x = fdm.hat_initial_condition_1d(x_array, config_x)

    config_y = fdm.Diffusion1DConfig(
        domain_length_x=config.domain_length_y,
        num_grid_points_x=config.num_grid_points_y,
        max_iterations=1,
        sigma=config.sigma,
        viscosity=config.viscosity,
        hat_start=config.hat_start_y,
        hat_end=config.hat_end_y,
        u_min=0.0,
        u_max=1.0,
    )
    initial_condition_y = fdm.hat_initial_condition_1d(y_array, config_y)

    u_numerical = fdm.solve_diffusion_2d(initial_condition, config)[-1]

    # x
    num_modes = 100
    basis = "cosine"  # "periodic" or "cosine"

    mode_indices = signal_processing.generate_mode_indices(num_modes)

    mode_coefficients_x = signal_processing.compute_coefficients(
        initial_condition_x,
        x_array,
        mode_indices,
        basis=basis,
    )

    mode_coefficients_y = signal_processing.compute_coefficients(
        initial_condition_y,
        y_array,
        mode_indices,
        basis=basis,
    )

    series_terms_x = signal_processing.compute_series_terms(mode_indices, mode_coefficients_x, x_array, basis=basis)
    series_terms_y = signal_processing.compute_series_terms(mode_indices, mode_coefficients_y, y_array, basis=basis)

    u_analytical_x = analytical.solve_heat_equation_1d(
        series_terms_x,
        mode_indices,
        x_array,
        time_array,
        config.viscosity,
        basis=basis)[-1]

    u_analytical_y = analytical.solve_heat_equation_1d(
        series_terms_y,
        mode_indices,
        y_array,
        time_array,
        config.viscosity,
        basis=basis)[-1]

    u_analytical = config.u_min + (config.u_max - config.u_min) * u_analytical_y[:, None] * u_analytical_x[None, :]

    peak = u_numerical.max()
    exact = u_analytical.max()

    assert abs(peak - exact) < 1e-3
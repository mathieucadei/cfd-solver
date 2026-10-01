import numpy as np

from core import fdm, analytical, signal_processing


def test_diffusion_1d_matches_heat():
    config = fdm.Diffusion1DConfig(
        domain_length_x=2.0,
        num_grid_points_x=101,
        max_iterations=501,
        sigma=0.2,
        viscosity=0.3,
        hat_start=0.5,
        hat_end=1.0,
        u_min=1.0,
        u_max=2.0,
    )

    x_array = fdm.make_x_grid(config)
    dt = fdm.compute_diffusive_dt_1d(config)
    time_array = np.arange(0, config.max_iterations + 1) * dt
    initial_condition = fdm.hat_initial_condition_1d(x_array, config)

    u_numerical = fdm.solve_diffusion_1d(
        initial_condition, config=config
    )[-1]

    num_modes = 100
    basis = "cosine"  # "periodic" or "cosine"

    mode_indices = signal_processing.generate_mode_indices(num_modes)

    mode_coefficients = signal_processing.compute_coefficients(
        initial_condition,
        x_array,
        mode_indices,
        basis=basis,
    )

    series_terms = signal_processing.compute_series_terms(mode_indices, mode_coefficients, x_array, basis=basis)

    u_analytical = analytical.solve_heat_equation_1d(
        series_terms,
        mode_indices,
        x_array,
        time_array,
        config.viscosity,
        basis=basis)[-1]

    peak = u_numerical.max()
    exact = u_analytical.max()

    assert abs(peak - exact) < 0.01
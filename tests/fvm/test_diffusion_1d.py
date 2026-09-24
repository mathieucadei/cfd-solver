import numpy as np

from core import fvm, analytical, signal_processing


def test_diffusion_1d_matches_heat():
    config = fvm.Diffusion1DConfig(
        domain_length_x=2.0,
        num_cells_x=100,
        expansion_ratio_x=0.,
        max_iterations=81,
        sigma=0.2,
        viscosity=0.3,
        hat_start=0.5,
        hat_end=1.0,
        u_min=1.0,
        u_max=2.0,
    )

    hx_array = fvm.build_hx_spacing(config)
    xc_array = fvm.build_x_centers(config)
    time_array = np.arange(0, config.max_iterations + 1)

    dt = fvm.compute_diffusive_dt_1d(config)
    time_array = np.arange(0, config.max_iterations + 1) * dt
    initial_condition = fvm.hat_initial_condition_1d(hx_array, config)

    u_numerical = fvm.solve_diffusion_1d(
        initial_condition, config=config
    )[-1]

    num_modes = 100
    basis = "cosine"  # "periodic" or "cosine"

    mode_indices = signal_processing.generate_mode_indices(num_modes)

    mode_coefficients = signal_processing.compute_coefficients(
        initial_condition,
        xc_array,
        mode_indices,
        basis=basis,
    )

    series_terms = signal_processing.compute_series_terms(mode_indices, mode_coefficients, xc_array, basis=basis)

    u_analytical = analytical.solve_heat_equation_1d(
        series_terms,
        mode_indices,
        xc_array,
        time_array,
        config.viscosity,
        basis=basis)[-1]

    peak = u_numerical.max()
    exact = u_analytical.max()

    assert abs(peak - exact) < 0.01
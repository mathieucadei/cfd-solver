import numpy as np

from core import fvm, analytical, signal_processing


def test_burgers_1d_matches_cole_hopf():
    config = fvm.BurgersEquation1DConfig(
        domain_length_x=6.0,
        num_cells_x=425,
        expansion_ratio_x=0.,
        max_iterations=100,
        time_step=0.0025,
        grid_type="cole_hopf",
        sigma=0.02,
        viscosity=0.07,
        hat_start=0.5,
        hat_end=1.0,
        u_min=1.0,
        u_max = 2.0,
    )

    xc_array = fvm.build_cole_hopf_x_centers(config)

    initial_condition = fvm.cole_hopf_initial_condition_1d(xc_array, config)

    u_numerical = fvm.solve_burgers_equation_1d(
        initial_condition, config=config
    )[-1]

    dt = fvm.compute_cole_hopf_dt_1d(config)

    u_analytical = analytical.solve_cole_hopf_1d(xc_array, dt, config)[-1]

    peak = u_numerical.max()
    exact = u_analytical.max()

    assert abs(peak - exact) < 0.015
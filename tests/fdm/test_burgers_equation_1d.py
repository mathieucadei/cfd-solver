import numpy as np

from core import fdm, analytical, signal_processing


def test_burgers_1d_matches_cole_hopf():
    config = fdm.BurgersEquation1DConfig(
        domain_length_x=6.0,
        num_grid_points_x=401,
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

    x_array = fdm.make_cole_hopf_x_grid(config)
    time_array = np.arange(0, config.max_iterations + 1)

    initial_condition = fdm.cole_hopf_initial_condition_1d(x_array, config)

    u_numerical = fdm.solve_burgers_equation_1d(
        initial_condition, config=config
    )[-1]

    dt = fdm.compute_cole_hopf_dt_1d(config)

    u_analytical = analytical.solve_cole_hopf_1d(x_array, dt, config)[-1]

    peak = u_numerical.max()
    exact = u_analytical.max()

    assert abs(peak - exact) < 0.03
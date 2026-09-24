import numpy as np

from core import fdm



def test_advection_1d_translates_exactly():
    config = fdm.Advection1DConfig(
        domain_length_x=2.0,
        num_grid_points_x=81,
        max_iterations=20,
        sigma=1.0,
        wavespeed=1.0,
        hat_start=0.5,
        hat_end=1.0,
        u_min=1.0,
        u_max=2.0,
        scheme='upwind',    
    )

    x_array = fdm.make_x_grid(config)

    initial_condition = fdm.hat_initial_condition_1d(x_array, config)

    final = fdm.solve_advection_1d(initial_condition, config)[-1]

    assert np.abs(final[config.max_iterations:] - initial_condition[:-config.max_iterations]).max() < 1e-12

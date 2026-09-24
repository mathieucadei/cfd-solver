import numpy as np

from core import fdm



def test_advection_2d_translates_exactly():
    config = fdm.Advection2DConfig(
        domain_length_x=2.0,
        domain_length_y=2.0,
        num_grid_points_x=81,
        num_grid_points_y=81,
        max_iterations=20,
        sigma=1.0,
        wavespeed=1.0,
        hat_start_x=0.5,
        hat_start_y=0.5,
        hat_end_x=1.0,
        hat_end_y=1.0,
        u_min=1.0,
        u_max=2.0,   
    )

    initial_condition = fdm.hat_initial_condition_2d(config)

    final = fdm.solve_advection_2d(initial_condition, config)[-1]

    assert np.max([
        np.abs(final[1][config.max_iterations:] - initial_condition[1][:-config.max_iterations]).max(),
        np.abs(final[0][config.max_iterations:] - initial_condition[0][:-config.max_iterations]).max()
        ]) < 1e-12

import numpy as np

from core import fvm



def test_advection_2d_translates_exactly():
    config = fvm.Advection2DConfig(
        domain_length_x=2.0,
        domain_length_y=2.0,
        num_cells_x=80,
        num_cells_y=80,
        expansion_ratio_x=0.,
        expansion_ratio_y=0.,
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

    initial_condition = fvm.hat_initial_condition_2d(config)

    final = fvm.solve_advection_2d(initial_condition, config)[-1]

    assert np.max([
        np.abs(final[1][config.max_iterations:] - initial_condition[1][:-config.max_iterations]).max(),
        np.abs(final[0][config.max_iterations:] - initial_condition[0][:-config.max_iterations]).max()
        ]) < 1e-12

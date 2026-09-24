import numpy as np

from core import fvm



def test_advection_1d_translates_exactly():
    config = fvm.Advection1DConfig(
        domain_length_x=2.0,
        num_cells_x=80,
        expansion_ratio_x=0.,
        max_iterations=20,
        sigma=1.0,
        wavespeed=1.0,
        hat_start=0.5,
        hat_end=1.0,
        u_min=1.0,
        u_max=2.0,  
    )

    xc_array = fvm.build_x_centers(config)

    initial_condition = fvm.hat_initial_condition_1d(xc_array, config)

    final = fvm.solve_advection_1d(initial_condition, config)[-1]

    assert np.abs(final[config.max_iterations:] - initial_condition[:-config.max_iterations]).max() < 1e-12

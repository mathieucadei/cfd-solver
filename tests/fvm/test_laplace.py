import numpy as np
import pandas as pd

from core import fvm

from pathlib import Path

def direct_solve(config, bottom, top, left, right):

    n = config.num_cells_x

    dist_x, dist_y = fvm.build_dist(config)
    face_areas_x, face_areas_y = fvm.build_face_areas(config)
    cell_volumes = fvm.compute_cell_volumes(config)   
    xc, yc = fvm.build_centers(config)

    a_w = face_areas_x[1:-1, 1:-1] / dist_x[:-1]
    a_e = face_areas_x[1:-1, 2:] / dist_x[1:]
    a_s = face_areas_y[1:-1, 1:-1] / dist_y[:-1, None]
    a_n = face_areas_y[2:, 1:-1] / dist_y[1:, None]

    A = np.zeros((n*n, n*n))
    b = np.zeros(n*n)
    k = np.arange(n*n).reshape(n, n)

    A = np.diag(np.full(n*n, (a_w + a_e + a_s + a_n))) \
        + np.diag(np.full(n*n-1, -a_w), k=-1) \
        + np.diag(np.full(n*n-1, -a_e), k=1) \
        + np.diag(np.full(n*n-3, -a_s), k=-3) \
        + np.diag(np.full(n*n-3, -a_n), k=3)


    b[k[:, 0]] = left
    b[k[:, -1]] = right
    b[k[0, :]] = bottom
    b[k[-1, :]] = top

    phi = np.linalg.solve(A, b)

    return phi

def test_laplace_matches_direct_solve():

    config = fvm.Laplace2DConfig(
        domain_length_x=2.0,
        domain_length_y=1.0,
        num_cells_x=30,
        num_cells_y=30,
        expansion_ratio_x=0.0,
        expansion_ratio_y=0.0,
        l1_norm_target=1e-10,
    )

    xc_array, yc_array = fvm.build_centers(config)

    initial_condition = fvm.laplace_initial_condition_2d(config)
    bottom_boundary = np.zeros_like(initial_condition[0, :])
    top_boundary = np.zeros_like(initial_condition[-1, :])
    right_boundary = yc_array
    left_boundary = np.zeros_like(initial_condition[:, 0])

    numerical_solution = fvm.solve_laplace_2d(
        initial_condition, 
        bottom_boundary=bottom_boundary, 
        top_boundary=top_boundary, 
        right_boundary=right_boundary, 
        left_boundary=left_boundary, 
        config=config)[-1]

    # assert abs(numerical_solution - exact) < 1e-6
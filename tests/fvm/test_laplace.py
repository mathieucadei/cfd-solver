import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core import fvm

def direct_solve(config, bottom, top, left, right):

    n = config.num_cells_x

    dist_x, dist_y = fvm.build_dist(config)
    face_areas_x, face_areas_y = fvm.build_face_areas(config)  
    xc, yc = fvm.build_centers(config)

    A = np.zeros((n*n, n*n))
    b = np.zeros(n*n)
    k = np.arange(n*n).reshape(n, n)

    for j in np.arange(n):
        for i in np.arange(n):

            if i > 0:
                a_w = face_areas_x[j, i] / dist_x[i-1]
                A[k[j, i], k[j, i]] += a_w
                A[k[j, i], k[j, i - 1]] = -a_w # west
            else:
                if not (type(left) == str and left == 'zero_gradient'):
                    a_w = face_areas_x[j, i] / xc[0]
                    A[k[j, i], k[j, i]] += a_w  # west
                    b[k[j, i]] += left[j] * a_w

            if i < n-1:
                a_e = face_areas_x[j, i] / dist_x[i]
                A[k[j, i], k[j, i]] += a_e     # current cell
                A[k[j, i], k[j, i + 1]] = -a_e  # east
            else:
                if not (type(right) == str and right == 'zero_gradient'):
                    a_e = face_areas_x[j, i] / (config.domain_length_x - xc[-1])
                    A[k[j, i], k[j, i]] += a_e  # east
                    b[k[j, i]] += right[j] * a_e

            if j > 0:
                a_s = face_areas_y[j, i] / dist_y[j-1]
                A[k[j, i], k[j, i]] += a_s     # current cell
                A[k[j, i], k[j - 1, i]] = -a_s # south
            else:

                if not (type(bottom) == str and bottom == 'zero_gradient'):
                    a_s = face_areas_y[j, i] / yc[0]
                    A[k[j, i], k[j, i]] += a_s # south
                    b[k[j, i]] += bottom[i] * a_s

            if j < n-1:
                a_n = face_areas_y[j, i] / dist_y[j]
                A[k[j, i], k[j, i]] += a_n     # current cell
                A[k[j, i], k[j + 1, i]] = -a_n # north
            else:

                if not (type(top) == str and top == 'zero_gradient'):
                    a_n = face_areas_y[j, i] / (config.domain_length_y - yc[-1])
                    A[k[j, i], k[j, i]] += a_n # north
                    b[k[j, i]] += top[i] * a_n

    phi = np.linalg.solve(A, b)

    return phi.reshape(n, n)

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

    yc_array = fvm.build_centers(config)[1]

    initial_condition = fvm.laplace_initial_condition_2d(config)
    bottom_boundary = 'zero_gradient'
    top_boundary = 'zero_gradient'
    right_boundary = yc_array
    left_boundary = np.zeros_like(initial_condition[:, 0])

    numerical_solution = fvm.solve_laplace_2d(
        initial_condition, 
        bottom_boundary=bottom_boundary, 
        top_boundary=top_boundary, 
        right_boundary=right_boundary, 
        left_boundary=left_boundary, 
        config=config)[-1]

    direct_solve_solution = direct_solve(
        config=config,
        bottom=bottom_boundary,
        top=top_boundary, 
        right=right_boundary, 
        left=left_boundary,         
    )

    assert abs(numerical_solution - direct_solve_solution).max() < 1e-6

# if __name__ == '__main__':

#     config = fvm.Laplace2DConfig(
#         domain_length_x=2.0,
#         domain_length_y=1.0,
#         num_cells_x=30,
#         num_cells_y=30,
#         expansion_ratio_x=0.0,
#         expansion_ratio_y=0.0,
#         l1_norm_target=1e-10,
#     )

#     x = np.linspace(0, 1, config.num_cells_x)
#     y = np.linspace(0, 1, config.num_cells_y)
#     n = config.num_cells_x
    
#     initial_condition = fvm.laplace_initial_condition_2d(config)
#     phi = direct_solve(
#         config,
#         bottom='zero_gradient',
#         top='zero_gradient',
#         right = y,
#         left = np.zeros_like(initial_condition[:, 0]),
#     )

#     X, Y = np.meshgrid(np.linspace(0, 1, config.num_cells_x), y)
#     Z = phi.reshape(n, n)

#     fig = plt.figure(figsize=(11, 7), dpi=100)
#     ax = fig.add_subplot(projection='3d')
#     ax.plot_surface(X, Y, Z, cmap='plasma')
#     plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core import fdm

def direct_solve(config, bottom, top, left, right):

    nx = config.num_grid_points_x
    ny = config.num_grid_points_y

    dx = fdm.compute_dx(config)
    dy = fdm.compute_dy(config)

    a_w = dy**2
    a_e = dy**2
    a_s = dx**2
    a_n = dx**2

    A = np.zeros((ny*nx, ny*nx))
    b = np.zeros(ny*nx)
    k = np.arange(ny*nx).reshape(ny, nx)

    for j in np.arange(ny):
        for i in np.arange(nx):

            A[k[j, i], k[j, i]] = a_w + a_e + a_s + a_n      # current cell

            if i > 0:
                A[k[j, i], k[j, i - 1]] = -a_w  # west
            else:
                if type(left) == str and left == 'zero_gradient':
                    A[k[j, i], k[j, i]] -= a_w  # west
                else:
                    b[k[j, i]] += left * a_w

            if i < nx-1:
                A[k[j, i], k[j, i + 1]] = -a_e  # east
            else:
                if type(right) == str and right == 'zero_gradient':
                    A[k[j, i], k[j, i]] -= a_e  # east
                else:
                    b[k[j, i]] += right[j] * a_e

            if j > 0:
                A[k[j, i], k[j - 1, i]] = -a_s # south
            else:
                if type(bottom) == str and bottom == 'zero_gradient':
                    A[k[j, i], k[j, i]] -= a_s # south
                else:
                    b[k[j, i]] += bottom[i] * a_s

            if j < ny-1:
                A[k[j, i], k[j + 1, i]] = -a_n # north
            else:
                if type(top) == str and top == 'zero_gradient':
                    A[k[j, i], k[j, i]] -= a_n # north
                else:
                    b[k[j, i]] += top[i] * a_n


    phi = np.linalg.solve(A, b)

    return phi.reshape(ny, nx)

def test_laplace_matches_direct_solve():

    config = fdm.Laplace2DConfig(
        domain_length_x=2.0,
        domain_length_y=1.0,
        num_grid_points_x=31,
        num_grid_points_y=31,
        l1_norm_target=1e-10,
    )

    y_array = fdm.make_y_grid(config)

    initial_condition = fdm.laplace_initial_condition_2d(config)
    bottom_boundary = initial_condition[1, :]
    top_boundary = initial_condition[-2, :] 
    right_boundary = y_array
    left_boundary = 0

    numerical_solution = fdm.solve_laplace_2d(
        initial_condition, 
        bottom_boundary=bottom_boundary, 
        top_boundary=top_boundary, 
        right_boundary=right_boundary, 
        left_boundary=left_boundary, 
        config=config)[-1]

    direct_solve_solution = direct_solve(
        config=config,
        bottom='zero_gradient',
        top='zero_gradient', 
        right=right_boundary, 
        left=left_boundary,         
    )

    assert abs(numerical_solution - direct_solve_solution).max() < 1e-6


if __name__ == '__main__':

    config = fdm.Laplace2DConfig(
        domain_length_x=1.0,
        domain_length_y=1.0,
        num_grid_points_x=31,
        num_grid_points_y=31,
        l1_norm_target=1e-10,
    )

    x = np.linspace(0, 1, config.num_grid_points_x)
    y = np.linspace(0, 1, config.num_grid_points_y)
    n = config.num_grid_points_x
    
    initial_condition = fdm.laplace_initial_condition_2d(config)
    phi = direct_solve(
        config,
        bottom='zero_gradient',
        top='zero_gradient',
        right = y,
        left = 0,
    )

    X, Y = np.meshgrid(np.linspace(0, 1, config.num_grid_points_x), y)
    Z = phi.reshape(n, n)

    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')
    plt.show()
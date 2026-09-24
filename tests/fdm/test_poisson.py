import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core import fdm

def direct_solve(
        config,
        initial_condition, 
        bottom, 
        top, 
        left, 
        right):

    nx = config.num_grid_points_x
    ny = config.num_grid_points_y

    dx = fdm.compute_dx(config)
    dy = fdm.compute_dy(config)

    a_w = dy**2
    a_e = dy**2
    a_s = dx**2
    a_n = dx**2

    A = np.zeros((ny*nx, ny*nx))
    b = initial_condition[1].flatten()
    k = np.arange(ny*nx).reshape(ny, nx)

    for j in np.arange(ny):
            for i in np.arange(nx):

                if j == 0 or j == ny-1 or i == 0 or i == nx-1:
                    A[k[j, i], k[j, i]] = 1

                    if j == 0:
                        if type(bottom) == str and bottom == 'zero_gradient':
                            A[k[j, i], k[j + 1, i]] = -1
                        else:
                            b[k[j, i]] = bottom[i]

                    elif j == ny-1:
                        if type(top) == str and top == 'zero_gradient':
                            A[k[j, i], k[j - 1, i]] = -1
                        else:
                            b[k[j, i]] = top[i]

                    elif i == 0:
                        if type(left) == str and left == 'zero_gradient':
                            A[k[j, i], k[j, i + 1]] = -1
                        else:
                            b[k[j, i]] = left[j]

                    else:
                        if type(right) == str and right == 'zero_gradient':
                            A[k[j, i], k[j, i - 1]] = -1
                        else:
                            b[k[j, i]] = right[j]
                        
                else:

                    A[k[j, i], k[j, i]] = a_w + a_e + a_s + a_n      # current cell
                    A[k[j, i], k[j, i - 1]] = -a_w  # west
                    A[k[j, i], k[j, i + 1]] = -a_e  # east
                    A[k[j, i], k[j - 1, i]] = -a_s # south
                    A[k[j, i], k[j + 1, i]] = -a_n # north
                    b[k[j, i]] = -b[k[j, i]] * dx**2 * dy**2


    phi = np.linalg.solve(A, b)

    return phi.reshape(ny, nx)

def test_laplace_matches_direct_solve():

    config = fdm.Poisson2DConfig(
        domain_length_x=2.0,
        domain_length_y=1.0,
        num_grid_points_x=31,
        num_grid_points_y=31,
        max_iterations = 10000,
        pressure_init = 0.0,
        source_terms=[
            fdm.SourceTerm(x=0.25, y=0.25, value=100.0),
            fdm.SourceTerm(x=0.75, y=0.75, value=-100.0),
        ],
        l1_norm_target=1e-10,
    )

    y_array = fdm.make_y_grid(config)

    initial_condition = fdm.poisson_initial_condition_2d(config)
    bottom_boundary = np.zeros_like(initial_condition[1][0, :])
    top_boundary = np.zeros_like(initial_condition[1][-1, :])
    left_boundary = np.zeros_like(initial_condition[1][:, 0])
    right_boundary = np.zeros_like(initial_condition[1][:, -1])

    numerical_solution = fdm.solve_poisson_2d(
        initial_condition, 
        config=config)[-1]

    direct_solve_solution = direct_solve(
        config=config,
        initial_condition=initial_condition,
        bottom=bottom_boundary,
        top=top_boundary, 
        right=right_boundary, 
        left=left_boundary,         
    )

    assert abs(numerical_solution - direct_solve_solution).max() < 1e-6


if __name__ == '__main__':

    config = fdm.Poisson2DConfig(
        domain_length_x=2.0,
        domain_length_y=1.0,
        num_grid_points_x=31,
        num_grid_points_y=31,
        max_iterations = 10000,
        pressure_init = 0.0,
        source_terms=[
            fdm.SourceTerm(x=0.25, y=0.25, value=100.0),
            fdm.SourceTerm(x=0.75, y=0.75, value=-100.0),
        ],
        l1_norm_target=1e-4,
    )

    x = np.linspace(0, config.domain_length_x, config.num_grid_points_x)
    y = np.linspace(0, config.domain_length_y, config.num_grid_points_y)
    n = config.num_grid_points_x
    
    initial_condition = fdm.poisson_initial_condition_2d(config)
    phi = direct_solve(
        config,
        initial_condition,
        bottom=np.zeros_like(initial_condition[1][0, :]),
        top=np.zeros_like(initial_condition[1][-1, :]),
        right=np.zeros_like(initial_condition[1][:, 0]),
        left=np.zeros_like(initial_condition[1][:, -1]),
    )

    X, Y = np.meshgrid(x, y)
    Z = phi.reshape(n, n)

    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.show()
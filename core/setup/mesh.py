
import numpy as np
import matplotlib.pyplot as plt

def make_x_grid(lx, nx, base):

    
    return np.logspace(0.0, lx, nx, base=base)/(base**lx/lx)

def make_y_grid(ly, ny, base):

    return np.logspace(0.0, ly, ny, base=base)/(base**ly/ly)

def mesh_grid(x_array, y_array):

    X, Y = np.meshgrid(x_array, y_array)

    return X, Y

def hx(X):

    X_widths = X[2:, 2:] - X[:-2, :-2]

    return X_widths

def hy(Y):

    Y_widths = Y[2:, 2:] - Y[:-2, :-2]

    return Y_widths

def cell_volumes(X_widths, Y_widths):
    Z = X_widths * Y_widths
    return Z



if __name__ == '__main__':

    x_array = make_x_grid(2, 41, 2)
    y_array = make_y_grid(2, 41, 2)

    X, Y = mesh_grid(x_array, y_array)

    x_widths = hx(X)
    y_widths = hy(Y)

    cell_vol = cell_volumes(x_widths, y_widths)

    plt.plot(X, Y, marker='o', color='k', linestyle='none')
    plt.show()

    # plt.scatter(x_array, x_array)
    # plt.show()

    print(cell_vol)

# def generate_mesh(config: object):

#     nx = config.num_grid_points_x
#     ny = config.num_grid_points_y

#     hx = config.cell_x_width_array
#     hy = config.cell_y_width_array

#     cell_centre = 

#     cell_volume = 
#     face_owner = 
#     face_neighbour = 
#     face_Sf = 
#     face_centre = 
#     face_weight =
#     boundary_patch = 
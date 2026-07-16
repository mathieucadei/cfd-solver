
import numpy as np
import matplotlib.pyplot as plt

from ...setup.mesh import build_mesh
from ...setup.initial_conditions import laplace_initial_condition_2d_fvm

from dataclasses import dataclass, field

def solve_laplace_2d_fvm(config: object):

    mesh = build_mesh(config)

    dx = mesh['dist_x']
    dy = mesh['dist_y']
    area_x = mesh['area_x']
    area_y = mesh['area_y']

    ax = area_x / dx
    ay = area_y / dy

    phi = laplace_initial_condition_2d_fvm(config).ravel()
    
    phi_n_x = phi[mesh['neigh_x']]
    phi_p_x = phi[mesh['owner_x']]
    phi_n_y = phi[mesh['neigh_y']]
    phi_p_y = phi[mesh['owner_y']]

    face_fluxes_x = ax * (phi_n_x - phi_p_x)

    face_fluxes_y = ay * (phi_n_y - phi_p_y)

    lap_x = np.concatenate([face_fluxes_x[:,0].reshape(-1,1), 
                            face_fluxes_x[:, 1:] - face_fluxes_x[:, :-1], 
                            face_fluxes_x[:,-1].reshape(-1,1)], axis=1)

    lap_y = np.concatenate([face_fluxes_y[0,:].reshape(-1,1).T, 
                            face_fluxes_y[1:, :] - face_fluxes_y[:-1, :], 
                            face_fluxes_y[-1,:].reshape(-1,1).T], axis=0)

    lap = lap_x + lap_y

    # laplacian = np.zeros((config.ny, config.nx))
    
    # laplacian.flat[mesh['owner_x']] += face_fluxes_x
    # laplacian.flat[mesh['neigh_x']] -= face_fluxes_x
    # laplacian.flat[mesh['owner_y']] += face_fluxes_y
    # laplacian.flat[mesh['neigh_y']] -= face_fluxes_y

    laplacian = lap / mesh['V']

    return laplacian



if __name__ == '__main__':

    @dataclass
    class Mesh:
        nx: int = 6
        ny: int = 6
        lx: float = 1.0
        ly: float = 1.0
        rx: float = 1.
        ry: float = 1.

    mesh = build_mesh(Mesh)

    laplacian = solve_laplace_2d_fvm(Mesh)

    print(laplacian)

    fig, ax = plt.subplots()
    pc = ax.pcolormesh(mesh['xf'], mesh['yf'], laplacian.T, edgecolors='k', linewidth=0.3)
    fig.colorbar(pc, label='laplacian')
    ax.set_aspect('equal')
    plt.show()    
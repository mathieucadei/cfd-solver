
import numpy as np
import matplotlib.pyplot as plt

from ...setup.mesh import build_mesh
from ...setup.initial_conditions import laplace_initial_condition_2d_fvm

from dataclasses import dataclass, field

def solve_laplace_2d_fvm(
        initial_condition: np.ndarray,        
        config: object,
):

    mesh = build_mesh(config)

    dx = mesh['dist_x']
    dy = mesh['dist_y']
    area_x = mesh['area_x']
    area_y = mesh['area_y']

    ax = area_x / dx
    ay = area_y / dy

    l1norm = 1

    phi = initial_condition
    
    phi_n_x = phi[mesh['neigh_x']]
    phi_p_x = phi[mesh['owner_x']]
    phi_n_y = phi[mesh['neigh_y']]
    phi_p_y = phi[mesh['owner_y']]

    phin_n_x = np.empty_like(phi_n_x)
    phin_p_x = np.empty_like(phi_p_x)
    phin_n_y = np.empty_like(phi_n_y)
    phin_p_y = np.empty_like(phi_p_y)

    history = []

    # while l1norm > config.l1_norm_target:

    phin_n_x = np.copy(phi_n_x)
    phin_p_x = np.copy(phi_p_x)
    phin_n_y = np.copy(phi_n_y)
    phin_p_y = np.copy(phi_p_y)

    face_fluxes_x = ax * (phin_n_x - phin_p_x)

    face_fluxes_y = ay * (phin_n_y - phin_p_y)

    # lap_x = np.concatenate([face_fluxes_x[:,0].reshape(-1,1), 
    #                         face_fluxes_x[:, 1:] - face_fluxes_x[:, :-1], 
    #                         - face_fluxes_x[:,-1].reshape(-1,1)], axis=1)

    # lap_y = np.concatenate([face_fluxes_y[0,:].reshape(-1,1).T, 
    #                         face_fluxes_y[1:, :] - face_fluxes_y[:-1, :], 
    #                         - face_fluxes_y[-1,:].reshape(-1,1).T], axis=0)

    # lap = lap_x + lap_y

    laplacian = np.zeros((config.ny, config.nx))
    
    laplacian.flat[mesh['owner_x']] += face_fluxes_x
    laplacian.flat[mesh['neigh_x']] -= face_fluxes_x
    laplacian.flat[mesh['owner_y']] += face_fluxes_y
    laplacian.flat[mesh['neigh_y']] -= face_fluxes_y

        # denominator = np.sum(np.abs(pn))

        # if denominator == 0:
        #     l1norm = np.sum(np.abs(p) - np.abs(pn)) 
        
        # else:
        #     l1norm = (np.sum(np.abs(p) - np.abs(pn))) / denominator
        
        # history.append(p.copy())

    laplacian = laplacian / mesh['V']

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

    initial_condition = laplace_initial_condition_2d_fvm(Mesh)

    # laplacian = solve_laplace_2d_fvm(initial_condition, Mesh)

    print(initial_condition)

    fig, ax = plt.subplots()
    pc = ax.pcolormesh(mesh['CX'], mesh['CY'], initial_condition.T, edgecolors='k', linewidth=0.3)
    fig.colorbar(pc, label='laplacian')
    ax.set_aspect('equal')
    plt.show()    
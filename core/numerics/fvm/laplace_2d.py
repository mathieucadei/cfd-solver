
import numpy as np
import matplotlib.pyplot as plt

from ..boundary_conditions import apply_laplace_boundary_2d_fvm
from ...setup.mesh import build_mesh
from ...setup.initial_conditions import laplace_initial_condition_2d_fvm

from dataclasses import dataclass, field

def solve_laplace_2d_fvm(
    initial_condition: np.ndarray,  
    bottom_boundary: float | np.ndarray,
    top_boundary: float | np.ndarray,
    right_boundary: float | np.ndarray,
    left_boundary: float | np.ndarray,      
    config: object,
):

    mesh = build_mesh(config)

    boundary_faces = apply_laplace_boundary_2d_fvm(
        phi=initial_condition,
        bottom=bottom_boundary,
        top=top_boundary,
        right=right_boundary,
        left=left_boundary,
        config=config,
    )

    # face coefficients
    ax = mesh['area_x'] / mesh['dist_x']
    ay = mesh['area_y'] / mesh['dist_y']
    ax_left   = boundary_faces['left']['area']   / boundary_faces['left']['dist_b']
    ax_right  = boundary_faces['right']['area']  / boundary_faces['right']['dist_b']
    ay_bottom = boundary_faces['bottom']['area'] / boundary_faces['bottom']['dist_b']
    ay_top    = boundary_faces['top']['area']    / boundary_faces['top']['dist_b']

    # D = denominator (sum of coefficients per cell)
    D = np.zeros((config.ny, config.nx))
    D.flat[mesh['owner_x']] += ax
    D.flat[mesh['neigh_x']] += ax
    D.flat[mesh['owner_y']] += ay
    D.flat[mesh['neigh_y']] += ay
    D.flat[boundary_faces['left']['owner']]   += ax_left
    D.flat[boundary_faces['right']['owner']]  += ax_right
    D.flat[boundary_faces['bottom']['owner']] += ay_bottom
    D.flat[boundary_faces['top']['owner']]    += ay_top
    D = D / mesh['V']

    b = np.zeros((config.ny, config.nx))   # source (zero for Laplace)

    # dx = mesh['dist_x']
    # dy = mesh['dist_y']
    # area_x = mesh['area_x']
    # area_y = mesh['area_y']

    # ax = area_x / dx
    # ay = area_y / dy

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

    while l1norm > config.l1_norm_target:

        phin_n_x = np.copy(phi_n_x)
        phin_p_x = np.copy(phi_p_x)
        phin_n_y = np.copy(phi_n_y)
        phin_p_y = np.copy(phi_p_y)

        boundary_fluxes_x_left = ax_left * \
                            (boundary_faces['left']['g'] - f[boundary_faces['left']['owner']])
        
        boundary_fluxes_x_right = ax_right * \
                            (boundary_faces['right']['g'] - f[boundary_faces['right']['owner']])
        
        face_fluxes_x = ax * (f[mesh['neigh_x']] - f[mesh['owner_x']])
        
        boundary_fluxes_y_bottom = ay_bottom * \
                            (boundary_faces['bottom']['g'] - f[boundary_faces['bottom']['owner']])
        
        boundary_fluxes_y_top = ay_top * \
                            (boundary_faces['top']['g'] - f[boundary_faces['top']['owner']])
        
        face_fluxes_y = ay * (f[mesh['neigh_y']] - f[mesh['owner_y']])

        # lap_x = np.concatenate([face_fluxes_x[:,0].reshape(-1,1), 
        #                         face_fluxes_x[:, 1:] - face_fluxes_x[:, :-1], 
        #                         - face_fluxes_x[:,-1].reshape(-1,1)], axis=1)

        # lap_y = np.concatenate([face_fluxes_y[0,:].reshape(-1,1).T, 
        #                         face_fluxes_y[1:, :] - face_fluxes_y[:-1, :], 
        #                         - face_fluxes_y[-1,:].reshape(-1,1).T], axis=0)

        # lap = lap_x + lap_y

            # denominator = np.sum(np.abs(pn))

            # if denominator == 0:
            #     l1norm = np.sum(np.abs(p) - np.abs(pn)) 
            
            # else:
            #     l1norm = (np.sum(np.abs(p) - np.abs(pn))) / denominator
            
            # history.append(p.copy())

        poisson = np.zeros((config.ny, config.nx))
        poisson.flat[boundary_faces['left']['owner']] += boundary_fluxes_x_left
        poisson.flat[boundary_faces['right']['owner']] += boundary_fluxes_x_right
        poisson.flat[mesh['owner_x']] += face_fluxes_x
        poisson.flat[mesh['neigh_x']] -= face_fluxes_x
        poisson.flat[boundary_faces['bottom']['owner']] += boundary_fluxes_y_bottom
        poisson.flat[boundary_faces['top']['owner']] += boundary_fluxes_y_top
        poisson.flat[mesh['owner_y']] += face_fluxes_y
        poisson.flat[mesh['neigh_y']] -= face_fluxes_y
        poisson = poisson / mesh['V']

        phi = phi + (poisson - b) / D
        f = phi.ravel()



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
    
    phi = apply_laplace_boundary_2d_fvm(
        phi=initial_condition,
        bottom=0,
        top=0,
        right=0,
        left=100,
        config=Mesh,
    )

    # laplacian = solve_laplace_2d_fvm(initial_condition, Mesh)

    print(phi)

    # fig, ax = plt.subplots()
    # pc = ax.pcolormesh(mesh['CX'], mesh['CY'], initial_condition.T, edgecolors='k', linewidth=0.3)
    # fig.colorbar(pc, label='laplacian')
    # ax.set_aspect('equal')
    # plt.show()
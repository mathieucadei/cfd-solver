
import numpy as np
import matplotlib.pyplot as plt

from ...setup.mesh import build_mesh

from dataclasses import dataclass, field

def solve_laplace_2d(config: object):

    mesh = build_mesh(config)

    phi = mesh['CX']**2 + mesh['CY']**2

    f = phi.ravel()

    face_fluxes_x = mesh['area_x'] / mesh['dist_x'] * (f[mesh['neigh_x']] - f[mesh['owner_x']])
    face_fluxes_y = mesh['area_y'] / mesh['dist_y'] * (f[mesh['neigh_y']] - f[mesh['owner_y']])

    laplacian = np.zeros((config.ny, config.nx))
    laplacian.flat[mesh['owner_x']] += face_fluxes_x
    laplacian.flat[mesh['neigh_x']] -= face_fluxes_x
    laplacian.flat[mesh['owner_y']] += face_fluxes_y
    laplacian.flat[mesh['neigh_y']] -= face_fluxes_y

    laplacian = laplacian / mesh['V']

    return laplacian



if __name__ == '__main__':

    @dataclass
    class Mesh:
        nx: int = 40
        ny: int = 40
        lx: float = 2.0
        ly: float = 1.0
        rx: float = 1.1
        ry: float = 1.1

    mesh = build_mesh(Mesh)

    laplacian = solve_laplace_2d(Mesh)

    fig, ax = plt.subplots()
    pc = ax.pcolormesh(mesh['xf'], mesh['yf'], laplacian.T, edgecolors='k', linewidth=0.3)
    fig.colorbar(pc, label='laplacian')
    ax.set_aspect('equal')
    plt.show()    
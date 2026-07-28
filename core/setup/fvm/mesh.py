
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass, field

def build_hx_spacing(config: object):

    rx = 1.0 + config.expansion_ratio_x

    raw_hx = rx**np.arange(config.num_cells_x//2)
    raw_hx_sum = np.sum(raw_hx)
    hx = raw_hx / raw_hx_sum * 0.5 * config.domain_length_x
    hx = np.append(hx, hx[::-1])

    return hx

def build_x_face_positions(config: object):

    hx = build_hx_spacing(config)  

    xf = np.cumsum(hx)
    xf =  np.concatenate([[0.0], xf])

    return xf

def build_x_centers(config: object):

    xf = build_x_face_positions(config)

    xc = 0.5 * (xf[:-1] + xf[1:])

    return xc

def build_dist_x(config: object):

    xc = build_x_centers(config)

    dist_x = xc[1:] - xc[:-1]

    return dist_x


def compute_cole_hopf_hx(config: object) -> float:
    """Compute the uniform grid spacing in the x-direction for the Cole-Hopf periodic domain."""

    rx = 1.0 + config.expansion_ratio_x

    raw_hx = rx**np.arange(config.num_cells_x//2)
    raw_hx_sum = np.sum(raw_hx)
    hx = raw_hx / raw_hx_sum * np.pi
    hx = np.append(hx, hx[::-1])

    return hx


def build_spacing(config: object):

    raw_hx = config.rx**np.arange(config.nx//2)
    raw_hx_sum = np.sum(raw_hx)
    hx = raw_hx / raw_hx_sum * 0.5 * config.lx
    hx = np.append(hx, hx[::-1])

    raw_hy = config.ry**np.arange(config.ny//2)
    raw_hy_sum = np.sum(raw_hy)
    hy = raw_hy / raw_hy_sum * 0.5 * config.ly
    hy = np.append(hy, hy[::-1])

    return hx, hy


def build_face_positions(config: object):

    hx, hy = build_spacing(config)  

    xf = np.cumsum(hx)
    xf =  np.concatenate([[0.0], xf])

    yf = np.cumsum(hy)
    yf =  np.concatenate([[0.0], yf])

    return xf, yf


def build_centers(config: object):

    xf, yf = build_face_positions(config)

    xc = 0.5 * (xf[:-1] + xf[1:])
    yc = 0.5 * (yf[:-1] + yf[1:])

    return xc, yc


def build_cell_ids(config: object):

    return np.arange(config.nx*config.ny).reshape((config.ny, config.nx))


def build_faces(config: object):

    ids = build_cell_ids(config)

    hx, hy = build_spacing(config)
    xc, yc = build_centers(config)
    
    owner_x = ids[:, :-1]
    neigh_x = ids[:, 1:]

    owner_y = ids[:-1, :]
    neigh_y = ids[1:, ::]

    dist_x = np.array([xc[1:] - xc[:-1]] * config.ny)
    dist_y = np.array([yc[1:] - yc[:-1]] * config.nx).T

    area_x = np.array([hy[:]] * (config.nx-1)).T
    area_y = np.array([hx[:]] * (config.ny-1))

    return owner_x, neigh_x, dist_x, area_x, owner_y, neigh_y, dist_y, area_y


def build_mesh(config: object):

    hx, hy = build_spacing(config)
    xf, yf = build_face_positions(config)
    xc, yc = build_centers(config)
    ids = build_cell_ids(config)

    CX = np.ones((config.ny,1)) * xc[None,:]
    CY = yc[:,None] * np.ones((1, config.nx))
    V  = hy[:,None] * hx[None,:]

    ox, nx_, dx, ax, oy, ny_, dy, ay = build_faces(config)

    return {
        "hx": hx, "hy": hy, "xf": xf, "yf": yf, "xc": xc, "yc": yc,
        "ids": ids, "CX": CX, "CY": CY, "V": V,
        "owner_x": ox, "neigh_x": nx_, "dist_x": dx, "area_x": ax,
        "owner_y": oy, "neigh_y": ny_, "dist_y": dy, "area_y": ay,
    }



if __name__ == '__main__':

    @dataclass
    class BurgersEquation1DFVMConfig:
        """Configuration parameters for the 1D Burgers' equation."""
        domain_length_x: float = 2.0
        num_cells_x: int = 101
        expansion_ratio_x: float = 0.
        max_iterations: int = 100
        time_step: float = 0.0025
        sigma: float = 0.2
        viscosity: float = 0.07
        grid_type: str = "hat"
        hat_start: float = 0.5
        hat_end: float = 1.0
        u_min: float = 1.0
        u_max: float = 2.0

    config = BurgersEquation1DFVMConfig()
    
    hx = build_hx_spacing(config)

    print(hx[0])

    # @dataclass
    # class Mesh:
    #     nx: int = 40
    #     ny: int = 40
    #     lx: float = 2.0
    #     ly: float = 1.0
    #     rx: float = 1.1
    #     ry: float = 1.1

    # mesh = build_mesh(Mesh)

    # print(mesh['xf'])

    # fig, ax = plt.subplots()
    # for x in mesh['xf']:
    #     ax.axvline(x, color='k', lw=0.5)
    # for y in mesh['yf']:
    #     ax.axhline(y, color='k', lw=0.5)
    # ax.set_xlim(0, mesh['xf'][-1])
    # ax.set_ylim(0, mesh['yf'][-1])
    # ax.set_aspect('equal')
    # plt.show()

    # fig, ax = plt.subplots()
    # pc = ax.pcolormesh(mesh['xf'], mesh['yf'], mesh['V'].T, edgecolors='k', linewidth=0.3)
    # fig.colorbar(pc, label='cell volume')
    # ax.set_aspect('equal')
    # plt.show()

    # print(mesh['hx'])
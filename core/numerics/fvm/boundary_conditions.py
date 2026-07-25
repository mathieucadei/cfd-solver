"""Boundary condition updates for finite-volume solvers."""



import numpy as np



def apply_diffusion_boundary_1d(
    u: np.ndarray,
    un: np.ndarray,
    dt: float,
    hx: np.ndarray,
    dist_x: np.ndarray,
    xc: np.ndarray,
    lx: float,
    nu: float,
) -> None:
    """Apply boundary updates for the 1D diffusion equation."""

    f_e = nu * (u[1] - u[0]) / dist_x[0]

    f_wb = nu * (u[0] - u[-1]) / (xc[0] + lx - xc[-1])

    f_w = nu * (u[-1] - u[-2]) / dist_x[-1]

    f_eb = nu * (u[0] - u[-1]) / (xc[0] + lx - xc[-1])

    u[0] = un[0] + dt / hx[0] * (f_wb - f_e)

    u[-1] = un[-1] + dt / hx[-1] * (f_w - f_eb)
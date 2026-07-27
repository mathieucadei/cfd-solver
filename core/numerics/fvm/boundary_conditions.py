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


def apply_burgers_boundary_1d(
    u: np.ndarray,
    un: np.ndarray,
    dt: float,
    hx: np.ndarray,
    dist_x: np.ndarray,
    xc: np.ndarray,
    lx: float,
    nu: float,
) -> None:
    """Apply boundary updates for the 1D Burgers' equation."""

    e = u**2 / 2

    conv_f_wb = e[-1]

    conv_f_e = e[0]

    conv_f_w = e[-2]

    conv_f_eb = conv_f_wb    

    diff_f_e = nu * (u[1] - u[0]) / dist_x[0]

    diff_f_wb = nu * (u[0] - u[-1]) / (xc[0] + lx - xc[-1])

    diff_f_w = nu * (u[-1] - u[-2]) / dist_x[-1]

    diff_f_eb = nu * (u[0] - u[-1]) / (xc[0] + lx - xc[-1])

    u[0] = un[0] - dt * (conv_f_e - conv_f_wb) / hx[0] \
        + dt * (diff_f_wb - diff_f_e) / hx[0]

    u[-1] = un[-1] - dt * (conv_f_eb - conv_f_w) / hx[-1] \
        + dt * (diff_f_w - diff_f_eb) / hx[-1]
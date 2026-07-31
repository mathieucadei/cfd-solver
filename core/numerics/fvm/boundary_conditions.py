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


def apply_burgers_boundary_1d_fvm(
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

    e = un**2 / 2

    conv_f_wb = e[-1]

    conv_f_e = e[0]

    conv_f_w = e[-2]

    conv_f_eb = conv_f_wb    

    diff_f_e = nu * (un[1] - un[0]) / dist_x[0]

    diff_f_wb = nu * (un[0] - un[-1]) / (xc[0] + lx - xc[-1])

    diff_f_w = nu * (un[-1] - un[-2]) / dist_x[-1]

    diff_f_eb = diff_f_wb

    u[0] = un[0] - dt * (conv_f_e - conv_f_wb) / hx[0] \
        + dt * (diff_f_e - diff_f_wb) / hx[0]

    # u[-1] = u[0]

    u[-1] = un[-1] - dt * (conv_f_eb - conv_f_w) / hx[-1] \
        + dt * (diff_f_eb - diff_f_w) / hx[-1]


def apply_advection_boundary_2d(
    u: np.ndarray,
    c: float,
    u_min: float,
    dt: float,
    hx: np.ndarray,
    hy: np.ndarray,
) -> None:
    """Apply boundary updates for the 2D advection equation."""

    f_w_bottom = c * u[0, :-1]

    f_e_bottom  = c * u[0, 1:]

    f_sb = u_min

    f_n_bottom  = c * u[0, 1:]


    f_w_top = c * u[-1, :-1]

    f_e_top  = c * u[-1, 1:]

    f_s_top = c * u[-1, 1:]

    f_nb = u_min


    f_wb = u_min

    f_e_left  = c * u[1:, 0]

    f_s_left = c * u[1:, 0]

    f_n_left  = c * u[:-1, 0]


    f_w_right = c * u[:-1, -1]

    f_eb = u_min

    f_s_right = c * u[1:, -1]

    f_n_right  = c * u[:-1, -1]



    u[0, :] = dt * (f_e_bottom - f_w_bottom) / hx[1:] + dt * (f_n_bottom - f_sb) / hy[1:]
    u[-1, :] = dt * (f_e[-1, :] - f_w[-1, :]) / hx[1:] + dt * (f_nb - f_s[-1, :]) / hy[1:]
    u[:, 0] = dt * (f_e[:, 0] - f_wb[:, 0]) / hx[1:] + dt * (f_n[:, 0] - f_s[:, 0]) / hy[1:]
    u[:, -1] = dt * (f_eb - f_w[:, -1]) / hx[1:] + dt * (f_n[:, -1] - f_s[:, -1]) / hy[1:]
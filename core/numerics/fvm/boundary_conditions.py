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
    f_sb = c * u_min
    f_n_bottom  = c * u[0, 1:]

    f_wb = c * u_min
    f_e_left  = c * u[1:, 0]
    f_s_left = c * u[1:, 0]
    f_n_left  = c * u[:-1, 0]

    u[0, 1:] = dt * (f_e_bottom - f_w_bottom) / hx[1:] + dt * (f_n_bottom - f_sb) / hy[0]
    u[1:, 0] = dt * (f_e_left - f_wb) / hx[0] + dt * (f_n_left - f_s_left) / hy[1:]
    u[0,0] = dt * (c * u[0,0] - f_wb) / hx[0] + dt * (c * u[0,0] - f_sb) / hy[0]


def apply_convection_boundary_2d(
    u: np.ndarray,
    v: np.ndarray,
    un: np.ndarray,
    vn: np.ndarray,
    u_min: float,
    v_min: float,
    dt: float,
    hx: np.ndarray,
    hy: np.ndarray,
) -> None:
    """Apply boundary updates for the 2D advection equation."""

    e_u = un**2 / 2
    e_v = vn**2 / 2

    f_w_u_bottom = e_u[0, :-1]
    f_e_u_bottom  = e_u[0, 1:]
    f_sb_u = v_min * u_min
    f_n_u_bottom  = vn[0, 1:] * un[0, 1:]

    f_wb_u = u_min**2 / 2
    f_e_u_left  = e_u[1:, 0]
    f_s_u_left = vn[:-1, 0] * un[:-1, 0]
    f_n_u_left  = vn[1:, 0] * un[1:, 0]

    u[0, 1:] = un[0, 1:] + dt * (f_e_u_bottom - f_w_u_bottom) / hx[1:] + dt * (f_n_u_bottom - f_sb_u) / hy[0]
    u[1:, 0] = un[1:, 0] + dt * (f_e_u_left - f_wb_u) / hx[0] + dt * (f_n_u_left - f_s_u_left) / hy[1:]
    u[0, 0] = un[0, 0] + dt * (e_u[0, 0] - f_wb_u) / hx[0] + dt * (vn[0, 0] * un[0, 0] - f_sb_u) / hy[0]   

    f_w_v_bottom = un[0, :-1] * vn[0, :-1]
    f_e_v_bottom  = un[0, 1:] * vn[0, 1:]
    f_sb_v = v_min**2 / 2
    f_n_v_bottom  = e_v[0, 1:]

    f_wb_v = u_min * v_min
    f_e_v_left  = un[1:, 0] * vn[1:, 0]
    f_s_v_left = e_v[:-1, 0]
    f_n_v_left  = e_v[1:, 0]

    v[0, 1:] = vn[0, 1:] + dt * (f_e_v_bottom - f_w_v_bottom) / hx[1:] + dt * (f_n_v_bottom - f_sb_v) / hy[0]
    v[1:, 0] = vn[1:, 0] + dt * (f_e_v_left - f_wb_v) / hx[0] + dt * (f_n_v_left - f_s_v_left) / hy[1:]
    v[0, 0] = vn[0, 0] + dt * (un[0, 0] * vn[0, 0] - f_wb_v) / hx[0] + dt * (e_v[0, 0] - f_sb_v) / hy[0]  
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
    face_areas_x: np.ndarray,
    face_areas_y: np.ndarray,
    cell_volumes: np.ndarray,
) -> None:
    """Apply boundary updates for the 2D advection equation."""

    f_w_bottom = c * u[0, :-1] * face_areas_x[0, :-1]
    f_e_bottom  = c * u[0, 1:] * face_areas_x[0, 1:]
    f_sb = c * u_min * face_areas_y[0, 1:]
    f_n_bottom  = c * u[0, 1:] * face_areas_y[1, 1:]

    f_wb = c * u_min * face_areas_x[1:, 0]
    f_e_left  = c * u[1:, 0] * face_areas_x[1:, 1]
    f_s_left = c * u[1:, 0] * face_areas_y[:-1, 0]
    f_n_left  = c * u[:-1, 0] * face_areas_y[1:, 0]

    u[0, 1:] = dt * (f_e_bottom - f_w_bottom) / cell_volumes[0, 1:] + dt * (f_n_bottom - f_sb) / cell_volumes[0, 1:]
    u[1:, 0] = dt * (f_e_left - f_wb) / cell_volumes[1:, 0]  + dt * (f_n_left - f_s_left) / cell_volumes[1:, 0] 
    u[0,0] = u_min

def apply_convection_boundary_2d(
    u: np.ndarray,
    v: np.ndarray,
    un: np.ndarray,
    vn: np.ndarray,
    u_min: float,
    v_min: float,
    dt: float,
    face_areas_x: np.ndarray,
    face_areas_y: np.ndarray,
    cell_volumes: np.ndarray,
) -> None:
    """Apply boundary updates for the 2D advection equation."""

    e_u = un**2 / 2
    e_v = vn**2 / 2

    f_w_u_bottom = e_u[0, :-1] * face_areas_x[0, :-1]
    f_e_u_bottom  = e_u[0, 1:] * face_areas_x[0, 1:]
    f_sb_u = v_min * u_min * face_areas_y[0, 1:]
    f_n_u_bottom  = vn[0, 1:] * un[0, 1:] * face_areas_y[1, 1:]

    f_wb_u = u_min**2 / 2 * face_areas_x[1:, 0]
    f_e_u_left  = e_u[1:, 0] * face_areas_x[1:, 1]
    f_s_u_left = vn[:-1, 0] * un[:-1, 0] * face_areas_y[:-1, 0]
    f_n_u_left  = vn[1:, 0] * un[1:, 0] * face_areas_y[1:, 0]

    u[0, 1:] = un[0, 1:] + dt * (f_e_u_bottom - f_w_u_bottom) / cell_volumes[0, 1:] + dt * (f_n_u_bottom - f_sb_u) / cell_volumes[0, 1:]
    u[1:, 0] = un[1:, 0] + dt * (f_e_u_left - f_wb_u) / cell_volumes[1:, 0] + dt * (f_n_u_left - f_s_u_left) / cell_volumes[1:, 0]
    u[0, 0] = u_min

    f_w_v_bottom = un[0, :-1] * vn[0, :-1] * face_areas_x[0, :-1]
    f_e_v_bottom  = un[0, 1:] * vn[0, 1:] * face_areas_x[0, 1:]
    f_sb_v = v_min**2 / 2 * face_areas_y[0, 1:]
    f_n_v_bottom  = e_v[0, 1:] * face_areas_y[1, 1:]

    f_wb_v = u_min * v_min * face_areas_x[1:, 0]
    f_e_v_left  = un[1:, 0] * vn[1:, 0] * face_areas_x[1:, 1]
    f_s_v_left = e_v[:-1, 0] * face_areas_y[:-1, 0]
    f_n_v_left  = e_v[1:, 0] * face_areas_y[1:, 0]

    v[0, 1:] = vn[0, 1:] + dt * (f_e_v_bottom - f_w_v_bottom) / cell_volumes[0, 1:] + dt * (f_n_v_bottom - f_sb_v) / cell_volumes[0, 1:]
    v[1:, 0] = vn[1:, 0] + dt * (f_e_v_left - f_wb_v) / cell_volumes[1:, 0] + dt * (f_n_v_left - f_s_v_left) / cell_volumes[1:, 0]
    v[0, 0] = v_min


def apply_diffusion_boundary_2d(
    u: np.ndarray,
    un: np.ndarray,
    dt: float,
    dist_x: np.ndarray,
    dist_y: np.ndarray,
    face_areas_x: np.ndarray,
    face_areas_y: np.ndarray,
    cell_volumes: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    lx: float,
    ly: float,
    nu: float,
) -> None:
    """Apply boundary updates for the 1D diffusion equation."""

    f_e_left = nu * face_areas_x[1:, 1] * (u[1:, 1] - u[1:, 0]) / dist_x[0]
    f_wb = nu * face_areas_x[1:, 0] * (u[1:, 0] - u[1:, -1]) / (xc[0] + lx - xc[-1])
    f_n_left = nu * face_areas_y[2:, 0] * (u[2:, 0] - u[1:-1, 0]) / dist_y[1:]
    f_s_left = nu * face_areas_y[1:-1, 0] * (u[1:-1, 0] - u[:-2, 0]) / dist_y[:-1]

    u[1:, 0] = un[1:, 0] + dt / cell_volumes[1:, 0] * (f_wb - f_e_left) + dt / cell_volumes[1:, 0] * (f_n_left - f_s_left)

    f_eb = nu * face_areas_x[1:, 0] * (u[1:, 0] - u[1:, -1]) / (xc[0] + lx - xc[-1])
    f_w_right = nu * face_areas_x[1:, -1] * (u[1:, -1] - u[1:, -2]) / dist_x[-1]
    f_n_right = nu * face_areas_y[2:, -1] * (u[2:, -1] - u[1:-1, -1]) / dist_y[1:]
    f_s_right = nu * face_areas_y[:-2, -1] * (u[1:-1, -1] - u[:-2, -1]) / dist_y[:-1]

    u[1:, -1] = un[1:, -1] + dt / cell_volumes[1:, -1] * (f_w_right - f_eb) + dt / cell_volumes[1:, -1] * (f_n_right - f_s_right)

    f_e_bottom = nu * face_areas_x[0, 2:] * (u[0, 2:] - u[0, 1:-1]) / dist_x[1:]
    f_w_bottom = nu * face_areas_x[0, 1:-1] * (u[0, 1:-1] - u[0, :-2]) / dist_x[:-1]
    f_n_bottom = nu * face_areas_y[1, 1:] * (u[1, 1:] - u[0, 1:]) / dist_y[0]
    f_sb = nu * face_areas_y[0, 1:] * (u[0, 1:] - u[-1, 1:]) / (yc[0] + ly - yc[-1])

    u[0, 1:] = un[0, 1:] + dt / cell_volumes[0, 1:] * (f_w_bottom - f_e_bottom) + dt / cell_volumes[0, 1:] * (f_n_bottom - f_sb)

    f_e_top = nu * face_areas_x[-1, 2:] * (u[-1, 2:] - u[-1, 1:-1]) / dist_x[1:]
    f_w_top = nu * face_areas_x[-1, 1:-1] * (u[-1, 1:-1] - u[-1, :-2]) / dist_x[:-1]
    f_nb = nu * face_areas_y[0, 1:] * (u[0, 1:] - u[-1, 1:]) / (yc[0] + ly - yc[-1])
    f_s_top = nu * face_areas_y[-1, 1:] * (u[-1, 1:] - u[-2, 1:]) / dist_y[-1]

    u[-1, 1:] = un[-1, 1:] + dt / cell_volumes[-1, 1:] * (f_w_bottom - f_e_bottom) + dt / cell_volumes[-1, 1:] * (f_n_bottom - f_sb)


def apply_laplace_boundary_2d(
    p: np.ndarray,
    bottom: float | np.ndarray,
    top: float | np.ndarray,
    right: float | np.ndarray,
    left: float | np.ndarray,
    dist_x: np.ndarray,
    dist_y: np.ndarray,
    face_areas_x: np.ndarray,
    face_areas_y: np.ndarray,
    cell_volumes: np.ndarray,
    lx: float,
    ly: float,
    xc: np.ndarray, 
    yc: np.ndarray,
) -> None:
    """Apply boundary updates for the 2D Laplace equation."""

    f_w_u_bottom = face_areas_x[0, 1:-1] * (p[0, 1:-1] - p[0, :-2]) / dist_x[:-1]
    f_e_u_bottom  = face_areas_x[0, 2:] * (p[0, 2:] - p[0, 1:-1]) / dist_x[1:]
    f_sb_u = face_areas_y[0, 1:-1] * (p[0, 1:-1] - bottom) / yc[0]
    f_n_u_bottom  = face_areas_y[1, 1:-1] * (p[1, 1:-1] - p[0, 1:-1]) / dist_y[0]

    p[0, 1:-1] = (f_e_u_bottom - f_w_u_bottom) / cell_volumes[0, 1:-1]  +  (f_n_u_bottom - f_sb_u) / cell_volumes[0, 1:-1]

    f_w_u_top = face_areas_x[-1, 1:-1] * (p[-1, 1:-1] - p[-1, :-2]) / dist_x[:-1]
    f_e_u_top  = face_areas_x[-1, 2:] * (p[-1, 2:] - p[-1, 1:-1]) / dist_x[1:]
    f_s_u_top = face_areas_y[-2, 1:-1] * (p[-1, 1:-1] - p[-2, 1:-1]) / dist_y[-1]
    f_nb_u  = face_areas_y[-1, 1:-1] * (top - p[-1, 1:-1]) / (ly - yc[-1])

    p[-1, 1:-1] = (f_e_u_top - f_w_u_top) / cell_volumes[-1, 1:-1]  +  (f_nb_u - f_s_u_top) / cell_volumes[-1, 1:-1]


    f_wb_u= face_areas_x[1:-1, 0] * (p[1:-1, 0] - left) / xc[0]
    f_e_u_left  = face_areas_x[1:-1, 1] * (p[1:-1, 1] - p[1:-1, 0]) / dist_x[0]
    f_s_u_left = face_areas_y[1:-1, 0] * (p[1:-1, 0] - p[:-2, 0]) / dist_y[:-1]
    f_n_u_left  = face_areas_y[2:, 0] * (p[2:, 0] - p[1:-1, 0]) / dist_y[1:]

    
    p[1:-1, 0] = (f_e_u_left - f_wb_u) / cell_volumes[1:-1, 0]  +  (f_n_u_left - f_s_u_left) / cell_volumes[1:-1, 0]

    f_w_u_right= face_areas_x[1:-1, -2] * (p[1:-1, -1] - p[1:-1, -2]) / dist_x[-1]
    f_eb_u  = face_areas_x[1:-1, -1] * (right - p[1:-1, -1]) / (lx - xc[-1])
    f_s_u_right = face_areas_y[1:-1, -1] * (p[1:-1, -1] - p[:-2, -1]) / dist_y[:-1]
    f_n_u_right  = face_areas_y[2:, -1] * (p[2:, -1] - p[1:-1, -1]) / dist_y[1:]
    
    p[1:-1, -1] = (f_eb_u - f_w_u_right) / cell_volumes[1:-1, -1]  +  (f_n_u_right - f_s_u_right) / cell_volumes[1:-1, -1]

    f_e_u_bottom_left = face_areas_x[0, 1] * (p[0, 1] - p[0, 0]) / dist_x[0]
    f_n_u_bottom_left  = face_areas_y[1, 0] * (p[1, 0] - p[0, 0]) / dist_y[0]

    p[0, 0] = (f_e_u_bottom_left - f_wb_u[0]) / cell_volumes[0, 0]  +  (f_n_u_bottom_left - f_sb_u[0]) / cell_volumes[0, 0]

    f_e_u_top_left  = face_areas_x[-1, 1] * (p[-1, 1] - p[-1, 0]) / dist_x[0]
    f_s_u_top_left = face_areas_y[-1, 0] * (p[-1, 0] - p[-2, 0]) / dist_y[-1]

    p[-1, 0] = (f_e_u_top_left - f_wb_u[0]) / cell_volumes[-1, 0]  +  (f_nb_u[0] - f_s_u_top_left) / cell_volumes[-1, 0]
    
    f_w_u_bottom_right = face_areas_x[0, -1] * (p[0, -1] - p[0, -2]) / dist_x[-1]
    f_n_u_bottom_right  = face_areas_y[1, -1] * (p[1, -1] - p[0, -1]) / dist_y[-1]

    p[0, -1] = (f_eb_u[0] - f_w_u_bottom_right) / cell_volumes[0, -1]  +  (f_n_u_bottom_right - f_sb_u[0]) / cell_volumes[0, -1]

    f_w_u_top_right = face_areas_x[-1, -1] * (p[-1, -1] - p[-1, -2]) / dist_x[-1]
    f_s_u_top_right = face_areas_y[-2, -1] * (p[-1, -1] - p[-2, -1]) / dist_y[-1]

    p[-1, -1] = (f_eb_u[0] - f_w_u_top_right) / cell_volumes[-1, -1]  +  (f_nb_u[0] - f_s_u_top_right) / cell_volumes[-1, -1]



    # f_wb_u= face_areas_x[1:-1, 0] * (p[1:-1, 0] - left) / xc[0]
    # f_e_u_left  = face_areas_x[1:-1, 1] * (p[1:-1, 1] - p[1:-1, 0]) / dist_x[1:]
    # f_s_u_left = face_areas_y[1:-1, 0] * (p[1:-1, 0] - p[:-2, 0]) / dist_y[:-1, None]
    # f_n_u_left  = face_areas_y[2:, 0] * (p[2:, 0] - p[1:-1, 0]) / dist_y[1:, None]

    
    # p[1:-1, 0] = (f_e_u_left - f_wb_u) / cell_volumes[1:-1, 0]  +  (f_n_u_left - f_s_u_left) / cell_volumes[1:-1, 0]

    # f_w_u_right= face_areas_x[1:-1, -2] * (p[1:-1, -1] - p[1:-1, -2]) / dist_x[-1]
    # f_eb_u  = face_areas_x[1:-1, -1] * (right - p[1:-1, -1]) / (lx - xc[-1])
    # f_s_u_right = face_areas_y[1:-1, -1] * (p[1:-1, -1] - p[:-2, -1]) / dist_y
    # f_n_u_right  = face_areas_y[2:, -1] * (p[2:, -1] - p[1:-1, -1]) / dist_y[1:]
    
    # p[1:-1, -1] = (f_eb_u - f_w_u_right) / cell_volumes[1:-1, -1]  +  (f_n_u_right - f_s_u_right) / cell_volumes[1:-1, -1]

    # p[0, 0] = (f_e_u_bottom[0, 1] - f_wb_u[0,0]) / cell_volumes[0, 0]  +  (f_n_u_bottom - f_sb_u) / cell_volumes[0, 0]

    # p[-1, 0] = 0

    # p[0, -1] = 0

    # p[-1, -1] = 0

    # p[:, 0] = left  # p = left @ x = 0
    # p[:, -1] = right  # p = right @ x = 2
    # p[0, :] = bottom  # p = bottom @ y = 0
    # p[-1, :] = top  # p = top @ y = 1
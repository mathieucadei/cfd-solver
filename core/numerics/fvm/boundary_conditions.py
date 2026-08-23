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

    u[-1, 1:] = un[-1, 1:] + dt / cell_volumes[-1, 1:] * (f_w_top - f_e_top) + dt / cell_volumes[-1, 1:] * (f_nb - f_s_top)


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

    a_w_bottom = face_areas_x[0, 1:-1] / dist_x[:-1]
    a_e_bottom = face_areas_x[0, 2:] / dist_x[1:]
    a_sb = face_areas_y[0, 1:-1] / yc[0]
    a_n_bottom = face_areas_y[1, 1:-1] / dist_y[0]   

    f_w_bottom = a_w_bottom * p[0, :-2]
    f_e_bottom = a_e_bottom * p[0, 2:]
    f_sb = a_sb * bottom[1:-1]
    f_n_bottom = a_n_bottom * p[1, 1:-1]

    p[0, 1:-1] =(f_e_bottom + f_w_bottom + f_n_bottom + f_sb) / (a_w_bottom + a_e_bottom + a_sb + a_n_bottom)


    a_w_top = face_areas_x[-1, 1:-1] / dist_x[:-1]
    a_e_top = face_areas_x[-1, 2:] / dist_x[1:]
    a_s_top = face_areas_y[-2, 1:-1] / dist_y[-1]
    a_nb = face_areas_y[-1, 1:-1] / (ly - yc[-1])
  
    f_w_top = a_w_top * p[-1, :-2]
    f_e_top = a_e_top * p[-1, 2:]
    f_s_top = a_s_top * p[-2, 1:-1]
    f_nb = a_nb * top[1:-1]

    p[-1, 1:-1] =(f_e_top + f_w_top + f_nb + f_s_top) / (a_w_top + a_e_top + a_s_top + a_nb)


    a_wb = face_areas_x[1:-1, 0] / xc[0]
    a_e_left = face_areas_x[1:-1, 1] / dist_x[0]
    a_s_left = face_areas_y[1:-1, 0] / dist_y[:-1]
    a_n_left = face_areas_y[2:, 0] / dist_y[1:]
  
    f_wb = a_wb * left[1:-1]
    f_e_left= a_e_left * p[1:-1, 1]
    f_s_left = a_s_left * p[:-2, 0]
    f_n_left = a_n_left * p[2:, 0]

    p[1:-1, 0] =(f_e_left + f_wb + f_n_left + f_s_left) / (a_wb + a_e_left + a_s_left + a_n_left)


    a_w_right = face_areas_x[1:-1, -2] / dist_x[-1]
    a_eb = face_areas_x[1:-1, -1] / (lx - xc[-1])
    a_s_right = face_areas_y[1:-1, -1] / dist_y[:-1]
    a_n_right = face_areas_y[2:, -1] / dist_y[1:]
  
    f_w_right = a_w_right * p[1:-1, -2]
    f_eb = a_eb * right[1:-1]
    f_s_right = a_s_right * p[:-2, -1]
    f_n_right = a_n_right * p[2:, -1]

    p[1:-1, -1] =(f_eb + f_w_right + f_n_right + f_s_right) / (a_w_right + a_eb + a_s_right + a_n_right)

    a_wb_bottom = face_areas_x[0, 0] / xc[0]
    a_e_bottom_left = face_areas_x[0, 1] / dist_x[0]
    a_sb_left = face_areas_y[0, 0] / yc[0]
    a_n_bottom_left = face_areas_y[1, 0] / dist_y[0]   

    f_wb_bottom = a_wb_bottom * left[0]
    f_e_bottom_left = a_e_bottom_left * p[0, 1]
    f_sb_left = a_sb_left * bottom[0]
    f_n_bottom_left = a_n_bottom_left * p[1, 0]

    p[0, 0] =(f_e_bottom_left + f_wb_bottom + f_n_bottom_left + f_sb_left) / (a_wb_bottom + a_e_bottom_left + a_sb_left + a_n_bottom_left)

    a_wb_top= face_areas_x[-1, 0] / xc[0]
    a_e_top_left = face_areas_x[-1, 1] / dist_x[0]
    a_s_top_left = face_areas_y[-2, 0] / dist_y[-1]
    a_nb_left = face_areas_y[-1, 0] / (ly - yc[-1]) 

    f_wb_top = a_wb_top * left[-1]
    f_e_top_left = a_e_top_left * p[-1, 1]
    f_s_top_left = a_s_top_left * p[-2, 0]
    f_nb_left = a_nb_left * top[-1]

    p[-1, 0] =(f_e_top_left + f_wb_top + f_nb_left + f_s_top_left) / (a_wb_top + a_e_top_left + a_s_top_left + a_nb_left)


    a_w_bottom_right = face_areas_x[0, -2] / dist_x[-1]
    a_eb_bottom = face_areas_x[0, -1] / (lx - xc[-1])
    a_sb_right = face_areas_y[0, -1] / yc[0]
    a_n_bottom_right = face_areas_y[1, -1] / dist_y[0]   

    f_w_bottom_right = a_w_bottom_right * p[0, -2]
    f_eb_bottom = a_eb_bottom * right[0]
    f_sb_right = a_sb_right * bottom[-1]
    f_n_bottom_right = a_n_bottom_right * p[1, -1]

    p[0, -1] =(f_eb_bottom + f_w_bottom_right + f_n_bottom_right + f_sb_right) / (a_w_bottom_right + a_eb_bottom + a_sb_right + a_n_bottom_right)


    a_w_top_right = face_areas_x[-1, -2] / dist_x[-1]
    a_eb_top = face_areas_x[-1, -1] / (lx - xc[-1])
    a_s_top_right = face_areas_y[-2, -1] / dist_y[-1]
    a_nb_right = face_areas_y[-1, -1] / (ly - yc[-1])  

    f_w_top_right = a_w_top_right * p[-1, -2]
    f_eb_top = a_eb_top * right[-1]
    f_s_top_right = a_s_top_right * p[-2, -1]
    f_nb_right = a_nb_right * top[-1]

    p[-1, -1] =(f_eb_top + f_w_top_right + f_nb_right + f_s_top_right) / (a_w_top_right + a_eb_top + a_s_top_right + a_nb_right)


def apply_source_term_boundary_2d(
    b: np.ndarray,
    rho: float,
    dt: float,
    u: np.ndarray,
    v: np.ndarray,
    u_lid: float,
    face_areas_x: np.ndarray,
    face_areas_y: np.ndarray,
    cell_volumes: np.ndarray,
) -> np.ndarray:
    """Compute the 2D source term for the Poisson equation in the 2D Navier-Stokes solver."""

    # left wall

    f_wb_u = 0
    f_e_u_left = face_areas_x[1:-1, 1] * (u[1:-1, 0] + u[1:-1, 1]) / 2
    f_s_u_left = face_areas_y[1:-1, 0] * (u[:-2, 0] + u[1:-1, 0]) / 2
    f_n_u_left = face_areas_y[2:, 0] * (u[1:-1, 0] + u[2:, 0]) / 2

    f_wb_v = 0
    f_e_v_left = face_areas_x[1:-1, 1] * (v[1:-1, 0] + v[1:-1, 1]) / 2
    f_s_v_left = face_areas_y[1:-1, 0] * (v[:-2, 0] + v[1:-1, 0]) / 2
    f_n_v_left = face_areas_y[2:, 0] * (v[1:-1, 0] + v[2:, 0]) / 2

    b[1:-1, 0] = (
                        rho * 
                            (1 / dt *   (
                                            (f_e_u_left - f_wb_u) / cell_volumes[1:-1, 0] +
                                            (f_n_v_left - f_s_v_left) / cell_volumes[1:-1, 0]
                                        ) -
                                        (
                                            (f_e_u_left - f_wb_u) / cell_volumes[1:-1, 0])**2 -
                                    2 * (
                                            (f_n_u_left - f_s_u_left) / cell_volumes[1:-1, 0] *
                                            (f_e_v_left - f_wb_v) / cell_volumes[1:-1, 0]
                                        ) -
                                        (
                                            (f_n_v_left - f_s_v_left) / cell_volumes[1:-1, 0])**2
                            )
                    )

    # right wall

    f_w_u_right = face_areas_x[1:-1, -2] * (u[1:-1, -2] + u[1:-1, -1]) / 2
    f_eb_u = 0
    f_s_u_right = face_areas_y[1:-1, 0] * (u[:-2, -1] + u[1:-1, -1]) / 2
    f_n_u_right = face_areas_y[2:, 0] * (u[1:-1, -1] + u[2:, -1]) / 2

    f_w_v_right = face_areas_x[1:-1, -2] * (v[1:-1, -2] + v[1:-1, -1]) / 2
    f_eb_v = 0
    f_s_v_right = face_areas_y[1:-1, 0] * (v[:-2, -1] + v[1:-1, -1]) / 2
    f_n_v_right = face_areas_y[2:, 0] * (v[1:-1, -1] + v[2:, -1]) / 2

    b[1:-1, -1] = (
                        rho * 
                            (1 / dt *   (
                                            (f_eb_u - f_w_u_right) / cell_volumes[1:-1, -1] +
                                            (f_n_v_right - f_s_v_right) / cell_volumes[1:-1, -1]
                                        ) -
                                        (
                                            (f_eb_u - f_w_u_right) / cell_volumes[1:-1, -1])**2 -
                                    2 * (
                                            (f_n_u_right - f_s_u_right) / cell_volumes[1:-1, -1] *
                                            (f_eb_v - f_w_v_right) / cell_volumes[1:-1, -1]
                                        ) -
                                        (
                                            (f_n_v_right - f_s_v_right) / cell_volumes[1:-1, -1])**2
                            )
                    )


    # bottom wall

    f_w_u_bottom = face_areas_x[0, 1:-1] * (u[0, :-2] + u[0, 1:-1]) / 2
    f_e_u_bottom = face_areas_x[0, 2:] * (u[0, 1:-1] + u[0, 2:]) / 2
    f_sb_u = 0
    f_n_u_bottom = face_areas_y[1, 1:-1] * (u[0, 1:-1] + u[1, 1:-1]) / 2

    f_w_v_bottom = face_areas_x[0, 1:-1] * (v[0, :-2] + v[0, 1:-1]) / 2
    f_e_v_bottom = face_areas_x[0, 2:] * (v[0, 1:-1] + v[0, 2:]) / 2
    f_sb_v = 0
    f_n_v_bottom = face_areas_y[1, 1:-1] * (v[0, 1:-1] + v[1, 1:-1]) / 2

    b[0, 1:-1] = (
                        rho * 
                            (1 / dt *   (
                                            (f_e_u_bottom - f_w_u_bottom) / cell_volumes[0, 1:-1] +
                                            (f_n_v_bottom - f_sb_v) / cell_volumes[0, 1:-1]
                                        ) -
                                        (
                                            (f_e_u_bottom - f_w_u_bottom) / cell_volumes[0, 1:-1])**2 -
                                    2 * (
                                            (f_n_u_bottom - f_sb_u) / cell_volumes[0, 1:-1] *
                                            (f_e_v_bottom - f_w_v_bottom) / cell_volumes[0, 1:-1]
                                        ) -
                                        (
                                            (f_n_v_bottom - f_sb_v) / cell_volumes[0, 1:-1])**2
                            )
                    )

    # top wall

    f_w_u_top = face_areas_x[-1, 1:-1] * (u[-1, :-2] + u[-1, 1:-1]) / 2
    f_e_u_top = face_areas_x[-1, 2:] * (u[-1, 1:-1] + u[-1, 2:]) / 2
    f_s_u_top = face_areas_y[-2, 1:-1] * (u[-2, 1:-1] + u[-1, 1:-1]) / 2
    f_nb_u = face_areas_y[-1, 1:-1] * u_lid

    f_w_v_top = face_areas_x[-1, 1:-1] * (v[-1, :-2] + v[-1, 1:-1]) / 2
    f_e_v_top = face_areas_x[-1, 2:] * (v[-1, 1:-1] + v[-1, 2:]) / 2
    f_s_v_top = face_areas_y[-2, 1:-1] * (v[-2, 1:-1] + v[-1, 1:-1]) / 2
    f_nb_v = 0

    b[-1, 1:-1] = (
                        rho * 
                            (1 / dt *   (
                                            (f_e_u_top - f_w_u_top) / cell_volumes[-1, 1:-1] +
                                            (f_nb_v - f_s_v_top) / cell_volumes[-1, 1:-1]
                                        ) -
                                        (
                                            (f_e_u_top - f_w_u_top) / cell_volumes[-1, 1:-1])**2 -
                                    2 * (
                                            (f_nb_u - f_s_u_top) / cell_volumes[-1, 1:-1] *
                                            (f_e_v_top - f_w_v_top) / cell_volumes[-1, 1:-1]
                                        ) -
                                        (
                                            (f_nb_v - f_s_v_top) / cell_volumes[-1, 1:-1])**2
                            )
                    )


    # bottom left corner

    f_e_u_bottom_left = face_areas_x[0, 1] * (u[0, 0] + u[0, 1]) / 2
    f_n_u_bottom_left = face_areas_y[1, 0] * (u[0, 0] + u[1, 0]) / 2

    f_e_v_bottom_left = face_areas_x[0, 1] * (v[0, 0] + v[0, 1]) / 2
    f_n_v_bottom_left = face_areas_y[1, 0] * (v[0, 0] + v[1, 0]) / 2

    b[0, 0] = (
                        rho * 
                            (1 / dt *   (
                                            (f_e_u_bottom_left - f_wb_u) / cell_volumes[0, 0] +
                                            (f_n_v_bottom_left - f_sb_u) / cell_volumes[0, 0]
                                        ) -
                                        (
                                            (f_e_u_bottom_left - f_wb_u) / cell_volumes[0, 0])**2 -
                                    2 * (
                                            (f_n_u_bottom_left - f_sb_u) / cell_volumes[0, 0] *
                                            (f_e_v_bottom_left - f_wb_v) / cell_volumes[0, 0]
                                        ) -
                                        (
                                            (f_n_v_bottom_left - f_sb_v) / cell_volumes[0, 0])**2
                            )
                    )

    # bottom right corner

    f_w_u_bottom_right = face_areas_x[0, -2] * (u[0, -2] + u[0, -1]) / 2
    f_n_u_bottom_right = face_areas_y[1, -1] * (u[0, -1] + u[1, -1]) / 2

    f_w_v_bottom_right = face_areas_x[0, -2] * (v[0, -2] + v[0, -1]) / 2
    f_n_v_bottom_right = face_areas_y[1, -1] * (v[0, -1] + v[1, -1]) / 2

    b[0, -1] = (
                        rho * 
                            (1 / dt *   (
                                            (f_eb_u - f_w_u_bottom_right) / cell_volumes[0, -1] +
                                            (f_n_v_bottom_right - f_sb_u) / cell_volumes[0, -1]
                                        ) -
                                        (
                                            (f_eb_u - f_w_u_bottom_right) / cell_volumes[0, -1])**2 -
                                    2 * (
                                            (f_n_u_bottom_right - f_sb_u) / cell_volumes[0, -1] *
                                            (f_eb_v - f_w_v_bottom_right) / cell_volumes[0, -1]
                                        ) -
                                        (
                                            (f_n_v_bottom_right - f_sb_v) / cell_volumes[0, -1])**2
                            )
                    )


    # top left corner

    f_e_u_top_left = face_areas_x[-1, 1] * (u[-1, 0] + u[-1, 1]) / 2
    f_s_u_top_left = face_areas_y[-2, 0] * (u[-2, 0] + u[-1, 0]) / 2
    f_nb_u_left = face_areas_y[-1, 0] * u_lid

    f_e_v_top_left = face_areas_x[-1, 1] * (v[-1, 0] + v[-1, 1]) / 2
    f_s_v_top_left = face_areas_y[-2, 0] * (v[-2, 0] + v[-1, 0]) / 2

    b[-1, 0] = (
                        rho * 
                            (1 / dt *   (
                                            (f_e_u_top_left - f_wb_u) / cell_volumes[-1, 0] +
                                            (f_nb_u_left - f_s_u_top_left) / cell_volumes[-1, -1]
                                        ) -
                                        (
                                            (f_e_u_top_left - f_wb_u) / cell_volumes[-1, 0])**2 -
                                    2 * (
                                            (f_nb_u_left - f_s_u_top_left) / cell_volumes[-1, 0] *
                                            (f_e_v_top_left - f_wb_v) / cell_volumes[-1, 0]
                                        ) -
                                        (
                                            (f_nb_v - f_s_v_top_left) / cell_volumes[-1, 0])**2
                            )
                    )


    # top right corner

    f_w_u_top_right = face_areas_x[-1, -1] * (u[-1, -2] + u[-1, -1]) / 2
    f_s_u_top_right = face_areas_y[-2, -1] * (u[-2, -1] + u[-1, -1]) / 2
    f_nb_u_right = face_areas_y[-1, -1] * u_lid

    f_w_v_top_right = face_areas_x[-1, -1] * (v[-1, -2] + v[-1, -1]) / 2
    f_s_v_top_right = face_areas_y[-2, -1] * (v[-2, -1] + v[-1, -1]) / 2

    b[-1, -1] = (
                        rho * 
                            (1 / dt *   (
                                            (f_eb_u - f_w_u_top_right) / cell_volumes[-1, -1] +
                                            (f_nb_u_right - f_s_u_top_right) / cell_volumes[-1, -1]
                                        ) -
                                        (
                                            (f_eb_u - f_w_u_top_right) / cell_volumes[-1, -1])**2 -
                                    2 * (
                                            (f_nb_u_right - f_s_u_top_right) / cell_volumes[-1, -1] *
                                            (f_eb_v - f_w_v_top_right) / cell_volumes[-1, -1]
                                        ) -
                                        (
                                            (f_nb_v - f_s_v_top_right) / cell_volumes[-1, -1])**2
                            )
                    )



def apply_pressure_poisson_term_boundary(
    p: np.ndarray,
    b: np.ndarray,
    dist_x: np.ndarray,
    dist_y: np.ndarray,
    face_areas_x: np.ndarray,
    face_areas_y: np.ndarray,
    cell_volumes: np.ndarray,
    lx: float,
    ly: float,
    xc: np.ndarray, 
    yc: np.ndarray,
) -> np.ndarray:
    """Iteratively solve the Poisson equation for pressure correction in the 2D Navier-Stokes solver."""

    bottom = p[0, :]
    top = 0
    right = p[:, -1]
    left = p[:, 0]

    a_w_bottom = face_areas_x[0, 1:-1] / dist_x[:-1]
    a_e_bottom = face_areas_x[0, 2:] / dist_x[1:]
    a_sb = face_areas_y[0, 1:-1] / yc[0]
    a_n_bottom = face_areas_y[1, 1:-1] / dist_y[0]   

    f_w_bottom = a_w_bottom * p[0, :-2]
    f_e_bottom = a_e_bottom * p[0, 2:]
    f_sb = a_sb * bottom[1:-1]
    f_n_bottom = a_n_bottom * p[1, 1:-1]

    p[0, 1:-1] =(f_e_bottom + f_w_bottom + f_n_bottom + f_sb - b[0, 1:-1] * cell_volumes[0, 1:-1]) / (a_w_bottom + a_e_bottom + a_sb + a_n_bottom)


    a_w_top = face_areas_x[-1, 1:-1] / dist_x[:-1]
    a_e_top = face_areas_x[-1, 2:] / dist_x[1:]
    a_s_top = face_areas_y[-2, 1:-1] / dist_y[-1]
    a_nb = face_areas_y[-1, 1:-1] / (ly - yc[-1])
  
    f_w_top = a_w_top * p[-1, :-2]
    f_e_top = a_e_top * p[-1, 2:]
    f_s_top = a_s_top * p[-2, 1:-1]
    f_nb = a_nb * top

    p[-1, 1:-1] =(f_e_top + f_w_top + f_nb + f_s_top - b[-1, 1:-1] * cell_volumes[-1, 1:-1]) / (a_w_top + a_e_top + a_s_top + a_nb)


    a_wb = face_areas_x[1:-1, 0] / xc[0]
    a_e_left = face_areas_x[1:-1, 1] / dist_x[0]
    a_s_left = face_areas_y[1:-1, 0] / dist_y[:-1]
    a_n_left = face_areas_y[2:, 0] / dist_y[1:]
  
    f_wb = a_wb * left[1:-1]
    f_e_left= a_e_left * p[1:-1, 1]
    f_s_left = a_s_left * p[:-2, 0]
    f_n_left = a_n_left * p[2:, 0]

    p[1:-1, 0] =(f_e_left + f_wb + f_n_left + f_s_left - b[1:-1, 0] * cell_volumes[1:-1, 0]) / (a_wb + a_e_left + a_s_left + a_n_left)


    a_w_right = face_areas_x[1:-1, -2] / dist_x[-1]
    a_eb = face_areas_x[1:-1, -1] / (lx - xc[-1])
    a_s_right = face_areas_y[1:-1, -1] / dist_y[:-1]
    a_n_right = face_areas_y[2:, -1] / dist_y[1:]
  
    f_w_right = a_w_right * p[1:-1, -2]
    f_eb = a_eb * right[1:-1]
    f_s_right = a_s_right * p[:-2, -1]
    f_n_right = a_n_right * p[2:, -1]

    p[1:-1, -1] =(f_eb + f_w_right + f_n_right + f_s_right - b[1:-1, -1] * cell_volumes[1:-1, -1]) / (a_w_right + a_eb + a_s_right + a_n_right)

    a_wb_bottom = face_areas_x[0, 0] / xc[0]
    a_e_bottom_left = face_areas_x[0, 1] / dist_x[0]
    a_sb_left = face_areas_y[0, 0] / yc[0]
    a_n_bottom_left = face_areas_y[1, 0] / dist_y[0]   

    f_wb_bottom = a_wb_bottom * left[0]
    f_e_bottom_left = a_e_bottom_left * p[0, 1]
    f_sb_left = a_sb_left * bottom[0]
    f_n_bottom_left = a_n_bottom_left * p[1, 0]

    p[0, 0] =(f_e_bottom_left + f_wb_bottom + f_n_bottom_left + f_sb_left - b[0, 0] * cell_volumes[0, 0]) / (a_wb_bottom + a_e_bottom_left + a_sb_left + a_n_bottom_left)

    a_wb_top= face_areas_x[-1, 0] / xc[0]
    a_e_top_left = face_areas_x[-1, 1] / dist_x[0]
    a_s_top_left = face_areas_y[-2, 0] / dist_y[-1]
    a_nb_left = face_areas_y[-1, 0] / (ly - yc[-1]) 

    f_wb_top = a_wb_top * left[-1]
    f_e_top_left = a_e_top_left * p[-1, 1]
    f_s_top_left = a_s_top_left * p[-2, 0]
    f_nb_left = a_nb_left * top

    p[-1, 0] =(f_e_top_left + f_wb_top + f_nb_left + f_s_top_left - b[-1, 0] * cell_volumes[-1, 0]) / (a_wb_top + a_e_top_left + a_s_top_left + a_nb_left)


    a_w_bottom_right = face_areas_x[0, -2] / dist_x[-1]
    a_eb_bottom = face_areas_x[0, -1] / (lx - xc[-1])
    a_sb_right = face_areas_y[0, -1] / yc[0]
    a_n_bottom_right = face_areas_y[1, -1] / dist_y[0]   

    f_w_bottom_right = a_w_bottom_right * p[0, -2]
    f_eb_bottom = a_eb_bottom * right[0]
    f_sb_right = a_sb_right * bottom[-1]
    f_n_bottom_right = a_n_bottom_right * p[1, -1]

    p[0, -1] =(f_eb_bottom + f_w_bottom_right + f_n_bottom_right + f_sb_right - b[0, -1] * cell_volumes[0, -1]) / (a_w_bottom_right + a_eb_bottom + a_sb_right + a_n_bottom_right)


    a_w_top_right = face_areas_x[-1, -2] / dist_x[-1]
    a_eb_top = face_areas_x[-1, -1] / (lx - xc[-1])
    a_s_top_right = face_areas_y[-2, -1] / dist_y[-1]
    a_nb_right = face_areas_y[-1, -1] / (ly - yc[-1])  

    f_w_top_right = a_w_top_right * p[-1, -2]
    f_eb_top = a_eb_top * right[-1]
    f_s_top_right = a_s_top_right * p[-2, -1]
    f_nb_right = a_nb_right * top

    p[-1, -1] =(f_eb_top + f_w_top_right + f_nb_right + f_s_top_right - b[-1, -1] * cell_volumes[-1, -1]) / (a_w_top_right + a_eb_top + a_s_top_right + a_nb_right)


def apply_cavity_flow_boundary_2d(
    u: np.ndarray,
    v: np.ndarray,
    un: np.ndarray,
    vn: np.ndarray,
    p: np.ndarray,
    u_lid: float,
    dt: float,
    rho: float,
    nu: float,
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
    """Apply boundary updates for the 2D cavity flow equation."""

    bottom = p[0, :]
    top = 0
    right = p[:, -1]
    left = p[:, 0]

    # left wall

    ## convection
    
    u_wb = 0
    u_e_left = (u[1:-1, 0] + u[1:-1, 1]) / 2
    v_s_left = (v[:-2, 0] + v[1:-1, 0]) / 2
    v_n_left = (v[1:-1, 0] + v[2:, 0]) / 2   

    f_wb = u_wb * face_areas_x[1:-1, 0]
    f_e_left = u_e_left * face_areas_x[1:-1, 1]
    f_s_left = v_s_left * face_areas_y[1:-1, 0]
    f_n_left = v_n_left * face_areas_y[2:, 0] 

    u_wb = 0
    u_e_left = np.where(f_e_left>0, u[1:-1, 0], u[1:-1, 1])
    u_s_left = np.where(f_s_left>0, u[:-2, 0], u[1:-1, 0])
    u_n_left = np.where(f_n_left>0, u[1:-1, 0], u[2:, 0])

    v_wb = 0
    v_e_left = np.where(f_e_left>0, v[1:-1, 0], v[1:-1, 1])
    v_s_left = np.where(f_s_left>0, v[:-2, 0], v[1:-1, 0])
    v_n_left = np.where(f_n_left>0, v[1:-1, 0], v[2:, 0])

    ## diffusion

    a_e_left = face_areas_x[1:-1, 1] / dist_x[0]
    a_wb = face_areas_x[1:-1, 0] / xc[0]
    a_n_left = face_areas_y[2:, 0] / dist_y[1:]
    a_s_left = face_areas_y[1:-1, 0] / dist_y[:-1]

    ## pressure

    pf_wb_p = face_areas_x[1:-1, 0] * left[1:-1]
    pf_e_p_left = face_areas_x[1:-1, 1] * (p[1:-1, 0] + p[1:-1, 1]) / 2
    pf_s_p_left = face_areas_y[1:-1, 0] * (p[:-2, 0] + p[1:-1, 0]) / 2
    pf_n_p_left = face_areas_y[2:, 0] * (p[1:-1, 0] + p[2:, 0]) / 2


    ## update

    u[1:-1, 0] = un[1:-1, 0] + dt / cell_volumes[1:-1, 0] * (
        (f_e_left * u_e_left - f_wb * u_wb + f_n_left * u_n_left - f_s_left * u_s_left) + 
        nu * (a_e_left * (u[1:-1, 1] - u[1:-1, 0]) - a_wb * (u[1:-1, 0] - u_wb) + a_n_left * (u[2:, 0] - u[1:-1, 0]) - a_s_left * (u[1:-1, 0] - u[:-2, 0]))
        ) - (dt / rho * (pf_e_p_left - pf_wb_p) / cell_volumes[1:-1, 0])
    
    v[1:-1, 0] = vn[1:-1, 0] + dt / cell_volumes[1:-1, 0] * (
        (f_e_left * v_e_left - f_wb * v_wb + f_n_left * v_n_left - f_s_left * v_s_left) + 
        nu * (a_e_left * (v[1:-1, 1] - v[1:-1, 0]) - a_wb * (v[1:-1, 0] - v_wb) + a_n_left * (v[2:, 0] - v[1:-1, 0]) - a_s_left * (v[1:-1, 0] - v[:-2, 0]))
        ) - (dt / rho * (pf_n_p_left - pf_s_p_left) / cell_volumes[1:-1, 0])

    # right wall

    ## convection

    u_w_right = (u[1:-1, -2] + u[1:-1, -1]) / 2
    u_eb = 0
    v_s_right = (v[:-2, -1] + v[1:-1, -1]) / 2
    v_n_right = (v[1:-1, -1] + v[2:, -1]) / 2   

    f_w_right = u_w_right * face_areas_x[1:-1, -2]
    f_eb = u_eb * face_areas_x[1:-1, -1]
    f_s_right = v_s_right * face_areas_y[1:-1, -1]
    f_n_right = v_n_right * face_areas_y[2:, -1] 

    u_w_right = np.where(f_w_right>0, u[1:-1, -2], u[1:-1, -1])
    u_eb = 0
    u_s_right = np.where(f_s_right>0, u[:-2, -1], u[1:-1, -1])
    u_n_right = np.where(f_n_right>0, u[1:-1, -1], u[2:, -1])

    v_w_right = np.where(f_w_right>0, v[1:-1, -2], v[1:-1, -1])
    v_eb = 0
    v_s_right = np.where(f_s_right>0, v[:-2, -1], v[1:-1, -1])
    v_n_right = np.where(f_n_right>0, v[1:-1, -1], v[2:, -1])

    ## diffusion

    a_eb = face_areas_x[1:-1, -1] / (lx - xc[-1])
    a_w_right = face_areas_x[1:-1, -1] / dist_x[-1]
    a_n_right = face_areas_y[2:, -1] / dist_y[1:]
    a_s_right = face_areas_y[1:-1, -1] / dist_y[:-1]

    ## pressure

    pf_w_p_right = face_areas_x[1:-1, -2] * (p[1:-1, -2] + p[1:-1, -1]) / 2 
    pf_eb_p = face_areas_x[1:-1, -1] * right[1:-1]
    pf_s_p_right = face_areas_y[1:-1, -1] * (p[:-2, -1] + p[1:-1, -1]) / 2
    pf_n_p_right = face_areas_y[2:, -1] * (p[1:-1, -1] + p[2:, -1]) / 2

    ## update

    u[1:-1, -1] = un[1:-1, -1] + dt / cell_volumes[1:-1, -1] * (
        (f_eb * u_eb - f_w_right * u_w_right + f_n_right * u_n_right - f_s_right * u_s_right) +
        nu * (a_eb * (u_eb - u[1:-1, -1]) - a_w_right * (u[1:-1, -1] - u[1:-1, -2]) + a_n_right * (u[2:, -1] - u[1:-1, -1]) - a_s_right * (u[1:-1, -1] - u[:-2, -1]))
        ) - (dt / rho * (pf_eb_p - pf_w_p_right) / cell_volumes[1:-1, -1])


    v[1:-1, -1] = vn[1:-1, -1] + dt / cell_volumes[1:-1, -1] * (
        (f_eb * v_eb - f_w_right * v_w_right + f_n_right * v_n_right - f_s_right * v_s_right) +
        nu * (a_eb * (v_eb - v[1:-1, -1]) - a_w_right * (v[1:-1, -1] - v[1:-1, -2]) + a_n_right * (v[2:, -1] - v[1:-1, -1]) - a_s_right * (v[1:-1, -1] - v[:-2, -1]))
        ) - (dt / rho * (pf_n_p_right - pf_s_p_right) / cell_volumes[1:-1, -1])


    # bottom wall

    ## convection

    u_w_bottom = (u[0, :-2] + u[0, 1:-1]) / 2
    u_e_bottom = (u[0, 1:-1] + u[0, 2:]) / 2
    v_sb = 0
    v_n_bottom = (v[0, 1:-1] + v[1, 1:-1]) / 2   

    f_w_bottom = u_w_bottom * face_areas_x[0, 1:-1]
    f_e_bottom = u_e_bottom * face_areas_x[0, 2:]
    f_sb = v_sb * face_areas_y[0, 1:-1]
    f_n_bottom = v_n_bottom * face_areas_y[1, 1:-1] 

    u_w_bottom = np.where(f_w_bottom>0, u[0, :-2], u[0, 1:-1])
    u_e_bottom = np.where(f_e_bottom>0, u[0, 1:-1], u[0, 2:])
    u_sb = 0
    u_n_bottom = np.where(f_n_bottom>0, u[0, 1:-1], u[1, 1:-1])

    v_w_bottom = np.where(f_w_bottom>0, v[0, :-2], v[0, 1:-1])
    v_e_bottom = np.where(f_e_bottom>0, v[0, 1:-1], v[0, 2:])
    v_sb = 0
    v_n_bottom = np.where(f_n_bottom>0, v[0, 1:-1], v[1, 1:-1])

    ## diffusion

    a_e_bottom = face_areas_x[0, 2:] / dist_x[1:]
    a_w_bottom = face_areas_x[0, 1:-1] / dist_x[:-1]
    a_n_bottom = face_areas_y[1, 1:-1] / dist_y[0]
    a_sb = face_areas_y[0, 1:-1] / yc[0]

    ## pressure

    pf_w_p_bottom = face_areas_x[0, 1:-1] * (p[0, :-2] + p[0, 1:-1]) / 2 
    pf_e_p_bottom = face_areas_x[0, 2:] * (p[0, 1:-1] + p[0, 2:]) / 2 
    pf_sb_p = face_areas_y[0, 1:-1] * bottom[1:-1]
    pf_n_p_bottom = face_areas_y[1, 1:-1] * (p[0, 1:-1] + p[1, 1:-1]) / 2

    ## update

    u[0, 1:-1] = un[0, 1:-1] + dt / cell_volumes[0, 1:-1] * (
        (f_e_bottom * u_e_bottom - f_w_bottom * u_w_bottom + f_n_bottom * u_n_bottom - f_sb * u_sb) +
        nu * (a_e_bottom * (u[0, 2:] - u[0, 1:-1]) - a_w_bottom * (u[0, 1:-1] - u[0, :-2]) + a_n_bottom * (u[1, 1:-1] - u[0, 1:-1]) - a_sb * (u[0, 1:-1] - u_sb))
        ) - (dt / rho * (pf_e_p_bottom - pf_w_p_bottom) / cell_volumes[0, 1:-1])

    v[0, 1:-1] = vn[0, 1:-1] + dt / cell_volumes[0, 1:-1] * (
            (f_e_bottom * v_e_bottom - f_w_bottom * v_w_bottom + f_n_bottom * v_n_bottom - f_sb * v_sb) +
            nu * (a_e_bottom * (v[0, 2:] - v[0, 1:-1]) - a_w_bottom * (v[0, 1:-1] - v[0, :-2]) + a_n_bottom * (v[1, 1:-1] - v[0, 1:-1]) - a_sb * (v[0, 1:-1] - v_sb))
        ) - (dt / rho * (pf_n_p_bottom - pf_sb_p) / cell_volumes[0, 1:-1])


    # top wall

    ## convection

    u_w_top = (u[-1, :-2] + u[-1, 1:-1]) / 2
    u_e_top = (u[-1, 1:-1] + u[-1, 2:]) / 2
    v_s_top = (v[-2, 1:-1] + v[-1, 1:-1]) / 2 
    v_nb = 0 

    f_w_top = u_w_top * face_areas_x[-1, 1:-1]
    f_e_top = u_e_top * face_areas_x[-1, 2:]
    f_s_top = v_s_top * face_areas_y[-2, 1:-1]
    f_nb = v_nb * face_areas_y[-1, 1:-1] 

    u_w_top = np.where(f_w_top>0, u[-1, :-2], u[-1, 1:-1])
    u_e_top = np.where(f_e_top>0, u[-1, 1:-1], u[-1, 2:])
    u_s_top = np.where(f_s_top>0, u[-2, 1:-1], u[-1, 1:-1])
    u_nb = u_lid

    v_w_top = np.where(f_w_top>0, v[-1, :-2], v[-1, 1:-1])
    v_e_top = np.where(f_e_top>0, v[-1, 1:-1], v[-1, 2:])
    v_s_top = np.where(f_s_top>0, v[-2, 1:-1], v[-1, 1:-1])
    v_nb = 0

    ## diffusion

    a_e_top = face_areas_x[-1, 2:] / dist_x[1:]
    a_w_top = face_areas_x[-1, 1:-1] / dist_x[:-1]
    a_nb = face_areas_y[-1, 1:-1] / (ly - yc[-1])
    a_s_top = face_areas_y[-1, 1:-1] / dist_y[-1]

    ## pressure

    pf_w_p_top = face_areas_x[-1, 1:-1] * (p[-1, :-2] + p[-1, 1:-1]) / 2 
    pf_e_p_top = face_areas_x[-1, 2:] * (p[-1, 1:-1] + p[-1, 2:]) / 2 
    pf_s_p_top = face_areas_y[-2, 1:-1] * (p[-2, 1:-1] + p[-1, 1:-1]) / 2
    pf_nb_p = face_areas_y[-1, 1:-1] * top

    ## update

    u[-1, 1:-1] = un[-1, 1:-1] + dt / cell_volumes[0, 1:-1] * (
        (f_e_top * u_e_top - f_w_top * u_w_top + f_nb * u_nb - f_s_top * u_s_top) +
        nu * (a_e_top * (u[-1, 2:] - u[-1, 1:-1]) - a_w_top * (u[-1, 1:-1] - u[-1, :-2]) + a_nb * (u_nb - u[-1, 1:-1]) - a_s_top * (u[-1, 1:-1] - u[-2, 1:-1]))
        ) - (dt / rho * (pf_e_p_top - pf_w_p_top) / cell_volumes[-1, 1:-1])

    v[-1, 1:-1] = vn[-1, 1:-1] + dt / cell_volumes[0, 1:-1] * (
        (f_e_top * v_e_top - f_w_top * v_w_top + f_nb * v_nb - f_s_top * v_s_top) +
        nu * (a_e_top * (v[-1, 2:] - v[-1, 1:-1]) - a_w_top * (v[-1, 1:-1] - v[-1, :-2]) + a_nb * (v_nb - v[-1, 1:-1]) - a_s_top * (v[-1, 1:-1] - v[-2, 1:-1]))
        ) - (dt / rho * (pf_nb_p - pf_s_p_top) / cell_volumes[-1, 1:-1])


    # bottom left wall

    ## convection
    
    u_e_bottom_left = (u[0, 0] + u[0, 1]) / 2
    v_n_bottom_left = (v[0, 0] + v[1, 0]) / 2   

    f_e_bottom_left = u_e_bottom_left * face_areas_x[0, 1]
    f_wb_bottom = u_wb * face_areas_x[0, 0]
    f_n_bottom_left = v_n_bottom_left * face_areas_y[1, 0]
    f_sb_left = v_sb * face_areas_y[0, 0]

    u_e_bottom_left = np.where(f_e_bottom_left>0, u[0, 0], u[0, 1])
    u_n_bottom_left = np.where(f_n_bottom_left>0, u[0, 0], u[1, 0])

    v_e_bottom_left = np.where(f_e_bottom_left>0, v[0, 0], v[0, 1])
    v_n_bottom_left = np.where(f_n_bottom_left>0, v[0, 0], v[1, 0])

    ## diffusion

    a_e_bottom_left = face_areas_x[0, 1] / dist_x[0]
    a_wb_bottom = face_areas_x[0, 0] / xc[0]
    a_n_bottom_left = face_areas_y[1, 0] / dist_y[0]
    a_sb_left = face_areas_y[0, 0] / yc[0]

    ## pressure

    pf_e_p_bottom_left = face_areas_x[0, 1] * (p[0, 0] + p[0, 1]) / 2
    pf_wb_p_bottom = face_areas_x[0, 0] * left[0]
    pf_n_p_bottom_left = face_areas_y[1, 0] * (p[0, 0] + p[1, 0]) / 2
    pf_sb_p_left = face_areas_y[0, 0] * bottom[0]


    ## update

    u[0, 0] = un[0, 0] + dt / cell_volumes[0, 0] * (
        (f_e_bottom_left * u_e_bottom_left - f_wb_bottom * u_wb + f_n_bottom_left * u_n_bottom_left - f_sb_left * u_sb) + 
        nu * (a_e_bottom_left * (u[0, 1] - u[0, 0]) - a_wb_bottom * (u[0, 0] - u_wb) + a_n_bottom_left * (u[1, 0] - u[0, 0]) - a_sb_left * (u[0, 0] - u_sb))
        ) - (dt / rho * (pf_e_p_bottom_left - pf_wb_p_bottom) / cell_volumes[0, 0])
    
    v[0, 0] = vn[0, 0] + dt / cell_volumes[0, 0] * (
        (f_e_bottom_left * v_e_bottom_left - f_wb_bottom * v_wb + f_n_bottom_left * v_n_bottom_left - f_sb_left * v_sb) + 
        nu * (a_e_bottom_left * (v[0, 1] - v[0, 0]) - a_wb_bottom * (v[0, 0] - v_wb) + a_n_bottom_left * (v[1, 0] - v[0, 0]) - a_sb_left * (v[0, 0] - v_sb))
        ) - (dt / rho * (pf_n_p_bottom_left - pf_sb_p_left) / cell_volumes[0, 0])


    # bottom right wall

    ## convection
    
    u_w_bottom_right = (u[0, -2] + u[0, -1]) / 2
    v_n_bottom_right = (v[0, -1] + v[1, -1]) / 2   

    f_eb_bottom = u_eb * face_areas_x[0, -1]
    f_w_bottom_right = u_w_bottom_right * face_areas_x[0, -2]
    f_n_bottom_right = v_n_bottom_right * face_areas_y[1, -1] 
    f_sb_right = v_sb * face_areas_y[0, -1]

    u_w_bottom_right = np.where(f_w_bottom_right>0, u[0, -2], u[0, -1])
    u_n_bottom_right = np.where(f_n_bottom_right>0, u[0, -1], u[1, -1])

    v_w_bottom_right = np.where(f_w_bottom_right>0, v[0, -2], v[0, -1])
    v_n_bottom_right = np.where(f_n_bottom_right>0, v[0, -1], v[1, -1])

    ## diffusion
    
    a_eb_bottom = face_areas_x[0, -1] / (lx - xc[-1])
    a_w_bottom_right = face_areas_x[0, -2] / dist_x[-1]
    a_n_bottom_right = face_areas_y[1, -1] / dist_y[0]
    a_sb_right = face_areas_y[0, -1] / yc[0]

    ## pressure

    pf_w_p_bottom_right = face_areas_x[0, -2] * (p[0, -2] + p[0, -1]) / 2
    pf_eb_p_bottom = face_areas_x[0, -1] * right[0]
    pf_n_p_bottom_right = face_areas_y[1, -1] * (p[0, -1] + p[1, -1]) / 2
    pf_sb_p_right = face_areas_y[0, -1] * bottom[-1]


    ## update

    u[0, -1] = un[0, -1] + dt / cell_volumes[0, -1] * (
        (f_eb_bottom * u_eb - f_w_bottom_right * u_w_bottom_right + f_n_bottom_right * u_n_bottom_right - f_sb_right * u_sb) + 
        nu * (a_eb_bottom * (u_eb - u[0, -1]) - a_w_bottom_right * (u[0, -1] - u[0, -2]) + a_n_bottom_right * (u[1, -1] - u[0, -1]) - a_sb_right * (u[0, -1] - u_sb))
        ) - (dt / rho * (pf_eb_p_bottom - pf_w_p_bottom_right) / cell_volumes[0, -1])
    
    v[0, -1] = vn[0, -1] + dt / cell_volumes[0, -1] * (
        (f_eb_bottom * v_eb - f_w_bottom_right * v_w_bottom_right + f_n_bottom_right * v_n_bottom_right - f_sb_right * v_sb) + 
        nu * (a_eb_bottom * (v_eb - v[0, -1]) - a_w_bottom_right * (v[0, -1] - v[0, -2]) + a_n_bottom_right * (v[1, -1] - v[0, -1]) - a_sb_right * (v[0, -1] - v_sb))
        ) - (dt / rho * (pf_n_p_bottom_right - pf_sb_p_right) / cell_volumes[0, -1])


    # top left wall

    ## convection
    
    u_e_top_left = (u[-1, 0] + u[-1, 1]) / 2
    v_s_top_left = (v[-2, 0] + v[-1, 0]) / 2   

    f_e_top_left = u_e_top_left * face_areas_x[-1, 1]
    f_wb_top = u_wb * face_areas_x[-1, 0]
    f_nb_left = v_nb * face_areas_y[-1, 0] 
    f_s_top_left = v_s_top_left * face_areas_y[-2, 0] 

    u_e_top_left = np.where(f_e_top_left>0, u[-1, 0], u[-1, 1])
    u_s_top_left = np.where(f_s_top_left>0, u[-2, 0], u[-1, 0])

    v_e_top_left = np.where(f_e_top_left>0, v[-1, 0], v[-1, 1])
    v_s_top_left = np.where(f_s_top_left>0, v[-2, 0], v[-1, 0])

    ## diffusion

    a_e_top_left = face_areas_x[-1, 1] / dist_x[0]
    a_wb_top = face_areas_x[-1, 0] / xc[0]
    a_s_top_left = face_areas_y[-2, 0] / dist_y[-1]
    a_nb_left = face_areas_y[-1, 0] / (ly - yc[-1])

    ## pressure

    pf_e_p_top_left = face_areas_x[-1, 1] * (p[-1, 0] + p[-1, 1]) / 2
    pf_wb_p_top = face_areas_x[-1, 0] * left[-1]
    pf_nb_p_left = face_areas_y[-1, 0] * top
    pf_s_p_top_left = face_areas_y[-2, 0] * (p[-2, 0] + p[-1, 0]) / 2


    ## update

    u[-1, 0] = un[-1, 0] + dt / cell_volumes[-1, 0] * (
        (f_e_top_left * u_e_top_left - f_wb_top * u_wb + f_nb_left * u_nb - f_s_top_left * u_s_top_left) + 
        nu * (a_e_top_left * (u[-1, 1] - u[-1, 0]) - a_wb_top * (u[-1, 0] - u_wb) + a_nb_left * (u_nb - u[-1, 0]) - a_s_top_left * (u[-1, 0] - u[-2, 0]))
        ) - (dt / rho * (pf_e_p_top_left - pf_wb_p_top) / cell_volumes[-1, 0])
    
    v[-1, 0] = vn[-1, 0] + dt / cell_volumes[-1, 0] * (
        (f_e_top_left * v_e_top_left - f_wb_top * v_wb + f_nb_left * v_nb - f_s_top_left * v_s_top_left) + 
        nu * (a_e_top_left * (v[-1, 1] - v[-1, 0]) - a_wb_top * (v[-1, 0] - v_wb) + a_nb_left * (v_nb - v[-1, 0]) - a_s_top_left * (v[-1, 0] - v[-2, 0]))
        ) - (dt / rho * (pf_nb_p_left - pf_s_p_top_left) / cell_volumes[-1, 0])


    # top right wall

    ## convection
    
    u_w_top_right = (u[-1, -2] + u[-1, -1]) / 2
    v_s_top_right = (v[-2, -1] + v[-1, -1]) / 2   

    f_eb_top = u_eb * face_areas_x[-1, -1]
    f_w_top_right = u_w_top_right * face_areas_x[-1, -2]
    f_nb_right = v_nb * face_areas_y[-1, -1] 
    f_s_top_right = v_s_top_right * face_areas_y[-2, -1] 

    u_w_top_right = np.where(f_w_top_right>0, u[-1, -2], u[-1, -1])
    u_s_top_right = np.where(f_s_top_right>0, u[-2, -1], u[-1, -1])

    v_w_top_right = np.where(f_w_top_right>0, v[-1, -2], v[-1, -1])
    v_s_top_right = np.where(f_s_top_right>0, v[-2, -1], v[-1, -1])

    ## diffusion
    a_eb_top = face_areas_x[-1, -1] / (lx - xc[-1])
    a_w_top_right = face_areas_x[-1, -2] / dist_x[-1]
    a_nb_right = face_areas_y[-1, -1] / (ly - yc[-1])
    a_s_top_right = face_areas_y[-2, -1] / dist_y[-1]

    ## pressure

    pf_eb_p_top = face_areas_x[-1, -1] * right[-1]
    pf_w_p_top_right = face_areas_x[-1, -2] * (p[-1, -2] + p[-1, -1]) / 2
    pf_nb_p_right = face_areas_y[-1, -1] * top
    pf_s_p_top_right = face_areas_y[-2, -1] * (p[-2, -1] + p[-1, -1]) / 2


    ## update

    u[-1, -1] = un[-1, -1] + dt / cell_volumes[-1, -1] * (
        (f_eb_top * u_eb - f_w_top_right * u_w_top_right + f_nb_right * u_nb - f_s_top_right * u_s_top_right) + 
        nu * (a_eb_top * (u_eb - u[-1, -1]) - a_w_top_right * (u[-1, -2] - u[-1, -1]) + a_nb_right * (u_nb - u[-1, -1]) - a_s_top_right * (u[-1, -1] - u[-2, -1]))
        ) - (dt / rho * (pf_eb_p_top - pf_w_p_top_right) / cell_volumes[-1, -1])
    
    v[-1, -1] = vn[-1, -1] + dt / cell_volumes[-1, -1] * (
        (f_eb_top * v_eb - f_w_top_right * v_w_top_right + f_nb_right * v_nb - f_s_top_right * v_s_top_right) + 
        nu * (a_eb_top * (v_eb - v[-1, -1]) - a_w_top_right * (v[-1, -2] - v[-1, -1]) + a_nb_right * (v_nb - v[-1, -1]) - a_s_top_right * (v[-1, -1] - v[-2, -1]))
        ) - (dt / rho * (pf_nb_p_right - pf_s_p_top_right) / cell_volumes[-1, -1])
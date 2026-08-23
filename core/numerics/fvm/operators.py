"""Reusable finite-difference operators for 1D & 2D transport equations."""



import numpy as np



def compute_advection_1d_term(
    u: np.ndarray,
    c: float,
    hx: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute the 1D upwind advection term for a constant wave speed."""

    term = np.zeros_like(u)

    f_w = c * u[:-1]

    f_e = c * u[1:]

    term[1:] = dt * (f_e - f_w) / hx[1:]

    return term


def compute_convection_1d_term(
    u: np.ndarray,
    hx: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute the 1D upwind advection term for a constant wave speed."""

    e = u**2 / 2

    term = np.zeros_like(u)

    f_w = e[:-1]

    f_e = e[1:]

    term[1:] = dt * (f_e - f_w) / hx[1:]

    return term


def compute_diffusion_1d_term(
    u: np.ndarray,
    hx: np.ndarray,
    dist_x: np.ndarray,
    dt: float,
    nu: float,
) -> np.ndarray:
    """Compute the 1D central-difference diffusion term."""

    term = np.zeros_like(u)

    f_w = nu * (u[1:] - u[:-1]) / dist_x

    f_e = nu * (u[2:] - u[1:-1]) / dist_x[1:]

    term[1:-1] = dt / hx[1:-1] * (f_e - f_w[:-1])

    return term


def compute_advection_2d_term(
    u: np.ndarray,
    c: float,
    face_areas_x: np.ndarray,
    face_areas_y: np.ndarray,
    cell_volumes: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute the 2D upwind advection term for a constant wave speed."""

    term = np.zeros_like(u)

    f_w = c * u[1:, :-1] * face_areas_x[1:, :-1]

    f_e = c * u[1:, 1:] * face_areas_x[1:, 1:]

    f_s = c * u[:-1, 1:] * face_areas_y[:-1, 1:]

    f_n = c * u[1:, 1:] * face_areas_y[1:, 1:]   

    term[1:, 1:] = dt * (f_e - f_w) / cell_volumes[1:, 1:] + dt * (f_n - f_s) / cell_volumes[1:, 1:]

    return term


def compute_convection_2d_term(
    u: np.ndarray,
    v: np.ndarray,      
    face_areas_x: np.ndarray,
    face_areas_y: np.ndarray,
    cell_volumes: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute the 2D upwind convection u & v terms"""

    u_w = (u[1:-1, :-2] + u[1:-1, 1:-1]) / 2
    u_e = (u[1:-1, 1:-1] + u[1:-1, 2:]) / 2
    v_s = (v[:-2, 1:-1] + v[1:-1, 1:-1]) / 2
    v_n = (v[1:-1 , 1:-1] + v[2:, 1:-1]) / 2   

    f_w = u_w * face_areas_x[1:-1, 1:-1]
    f_e = u_e * face_areas_x[1:-1, 2:]
    f_s = v_s * face_areas_y[1:-1, 1:-1]
    f_n = v_n * face_areas_y[2:, 1:-1] 

    u_w = np.where(f_w>0, u[1:-1, :-2], u[1:-1, 1:-1])
    u_e = np.where(f_e>0, u[1:-1, 1:-1], u[1:-1, 2:])
    u_s = np.where(f_s>0, u[:-2, 1:-1], u[1:-1, 1:-1])
    u_n = np.where(f_n>0, u[1:-1, 1:-1], u[2:, 1:-1])

    v_w = np.where(f_w>0, v[1:-1, :-2], v[1:-1, 1:-1])
    v_e = np.where(f_e>0, v[1:-1, 1:-1], v[1:-1, 2:])
    v_s = np.where(f_s>0, v[:-2, 1:-1], v[1:-1, 1:-1])
    v_n = np.where(f_n>0, v[1:-1, 1:-1], v[2:, 1:-1])

    # e_u = u**2 / 2
    # e_v = v**2 / 2

    u_term = np.zeros_like(u)
    v_term = np.zeros_like(v)

    u_term[1:-1, 1:-1] = dt / cell_volumes[1:-1, 1:-1] * (f_e * u_e - f_w * u_w + f_n * u_n - f_s * u_s)
    v_term[1:-1, 1:-1] = dt / cell_volumes[1:-1, 1:-1] * (f_e * v_e - f_w * v_w + f_n * v_n - f_s * v_s)

    # f_w_u = e_u[1:, :-1] * face_areas_x[1:, :-1]

    # f_e_u = e_u[1:, 1:] * face_areas_x[1:, 1:]

    # f_s_u = v[:-1, 1:] * u[:-1, 1:] * face_areas_y[:-1, 1:]

    # f_n_u = v[1:, 1:] * u[1:, 1:] * face_areas_y[1:, 1:]

    # f_w_v = u[1:, :-1] * v[1:, :-1] * face_areas_x[1:, :-1]

    # f_e_v = u[1:, 1:] * v[1:, 1:] * face_areas_x[1:, 1:]

    # f_s_v = e_v[:-1, 1:] * face_areas_y[:-1, 1:]

    # f_n_v = e_v[1:, 1:] * face_areas_y[1:, 1:]

    # u_term[1:, 1:] = dt * (f_e_u - f_w_u) / cell_volumes[1:, 1:] + dt * (f_n_u - f_s_u) / cell_volumes[1:, 1:]

    # v_term[1:, 1:] = dt * (f_e_v - f_w_v) / cell_volumes[1:, 1:] + dt * (f_n_v - f_s_v) / cell_volumes[1:, 1:]

    return u_term, v_term


def compute_diffusion_2d_term(
    u: np.ndarray,
    dist_x: np.ndarray,
    dist_y: np.ndarray, 
    face_areas_x: np.ndarray,
    face_areas_y: np.ndarray,
    cell_volumes: np.ndarray,
    dt: float,
    nu: float,
) -> np.ndarray:
    """Compute the 2D central-difference diffusion term."""

    term = np.zeros_like(u)

    f_w = nu * face_areas_x[1:, 1:] * (u[1:, 1:] - u[1:, :-1]) / dist_x

    f_e = nu * face_areas_x[1:, 2:] * (u[1:, 2:] - u[1:, 1:-1]) / dist_x[1:]

    f_s = nu * face_areas_y[:-1, 1:] * (u[1:, 1:] - u[:-1, 1:]) / dist_y[:, None]

    f_n = nu * face_areas_y[2:, 1:] * (u[2:, 1:] - u[1:-1, 1:]) / dist_y[1:, None]

    term[1:-1, 1:-1] = dt / cell_volumes[1:-1, 1:-1] * (f_e[:-1, :] - f_w[:-1, :-1]) + dt / cell_volumes[1:-1, 1:-1] * (f_n[:, :-1] - f_s[:-1, :-1])

    return term


def compute_source_term_2d(
    b: np.ndarray,
    rho: float,
    dt: float,
    u: np.ndarray,
    v: np.ndarray,
    dist_x: np.ndarray,
    dist_y: np.ndarray,
    face_areas_x: np.ndarray,
    face_areas_y: np.ndarray,
    cell_volumes: np.ndarray,
) -> np.ndarray:
    """Compute the 2D source term for the Poisson equation in the 2D Navier-Stokes solver."""

    f_w_u = face_areas_x[1:, 1:] * (u[1:, 1:] + u[1:, :-1]) / 2

    f_e_u = face_areas_x[1:, 2:] * (u[1:, 2:] + u[1:, 1:-1]) / 2

    f_s_u = face_areas_y[:-1, 1:] * (u[1:, 1:] + u[:-1, 1:]) / 2

    f_n_u = face_areas_y[2:, 1:] * (u[2:, 1:] + u[1:-1, 1:]) / 2

    f_w_v = face_areas_x[1:, 1:] * (v[1:, 1:] + v[1:, :-1]) / 2

    f_e_v = face_areas_x[1:, 2:] * (v[1:, 2:] + v[1:, 1:-1]) / 2

    f_s_v = face_areas_y[:-1, 1:] * (v[1:, 1:] + v[:-1, 1:]) / 2

    f_n_v = face_areas_y[2:, 1:] * (v[2:, 1:] + v[1:-1, 1:]) / 2

    b[1:-1, 1:-1] = (
                        rho * 
                            (1 / dt *   (
                                            (f_e_u[:-1, :] - f_w_u[:-1, :-1]) / cell_volumes[1:-1, 1:-1] +
                                            (f_n_v[:, :-1] - f_s_v[:-1, :-1]) / cell_volumes[1:-1, 1:-1]
                                        ) -
                                        (
                                            (f_e_u[:-1, :] - f_w_u[:-1, :-1]) / cell_volumes[1:-1, 1:-1])**2 -
                                    2 * (
                                            (f_n_u[:, :-1] - f_s_u[:-1, :-1]) / cell_volumes[1:-1, 1:-1] *
                                            (f_e_v[:-1, :] - f_w_v[:-1, :-1]) / cell_volumes[1:-1, 1:-1]
                                        ) -
                                        (
                                            (f_n_v[:, :-1] - f_s_v[:-1, :-1]) / cell_volumes[1:-1, 1:-1])**2
                            )
                    )

    return b


def compute_pressure_poisson_term(
    p: np.ndarray,
    b: np.ndarray,
    nit: int, 
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

    pn = p.copy()

    a_w = face_areas_x[1:-1, 1:-1] / dist_x[:-1]
    a_e = face_areas_x[1:-1, 2:] / dist_x[1:]
    a_s = face_areas_y[1:-1, 1:-1] / dist_y[:-1, None]
    a_n = face_areas_y[2:, 1:-1] / dist_y[1:, None]

    bottom = pn[0, :]
    top = 0
    right = pn[:, -1]
    left = pn[:, 0]
    
    for q in range(nit):
        
        pn = p.copy()

        f_w = a_w * pn[1:-1, :-2]
        f_e = a_e * pn[1:-1, 2:]
        f_s = a_s * pn[:-2, 1:-1]
        f_n = a_n * pn[2:, 1:-1]

        p[1:-1, 1:-1] =(f_e + f_w + f_n + f_s - b[1:-1, 1:-1] * cell_volumes[1:-1, 1:-1]) / (a_w + a_e + a_s + a_n)

        a_w_bottom = face_areas_x[0, 1:-1] / dist_x[:-1]
        a_e_bottom = face_areas_x[0, 2:] / dist_x[1:]
        a_sb = face_areas_y[0, 1:-1] / yc[0]
        a_n_bottom = face_areas_y[1, 1:-1] / dist_y[0]   

        f_w_bottom = a_w_bottom * pn[0, :-2]
        f_e_bottom = a_e_bottom * pn[0, 2:]
        f_sb = a_sb * bottom[1:-1]
        f_n_bottom = a_n_bottom * pn[1, 1:-1]

        p[0, 1:-1] =(f_e_bottom + f_w_bottom + f_n_bottom + f_sb - b[0, 1:-1] * cell_volumes[0, 1:-1]) / (a_w_bottom + a_e_bottom + a_sb + a_n_bottom)


        a_w_top = face_areas_x[-1, 1:-1] / dist_x[:-1]
        a_e_top = face_areas_x[-1, 2:] / dist_x[1:]
        a_s_top = face_areas_y[-2, 1:-1] / dist_y[-1]
        a_nb = face_areas_y[-1, 1:-1] / (ly - yc[-1])
    
        f_w_top = a_w_top * pn[-1, :-2]
        f_e_top = a_e_top * pn[-1, 2:]
        f_s_top = a_s_top * pn[-2, 1:-1]
        f_nb = a_nb * top

        p[-1, 1:-1] =(f_e_top + f_w_top + f_nb + f_s_top - b[-1, 1:-1] * cell_volumes[-1, 1:-1]) / (a_w_top + a_e_top + a_s_top + a_nb)


        a_wb = face_areas_x[1:-1, 0] / xc[0]
        a_e_left = face_areas_x[1:-1, 1] / dist_x[0]
        a_s_left = face_areas_y[1:-1, 0] / dist_y[:-1]
        a_n_left = face_areas_y[2:, 0] / dist_y[1:]
    
        f_wb = a_wb * left[1:-1]
        f_e_left= a_e_left * pn[1:-1, 1]
        f_s_left = a_s_left * pn[:-2, 0]
        f_n_left = a_n_left * pn[2:, 0]

        p[1:-1, 0] =(f_e_left + f_wb + f_n_left + f_s_left - b[1:-1, 0] * cell_volumes[1:-1, 0]) / (a_wb + a_e_left + a_s_left + a_n_left)


        a_w_right = face_areas_x[1:-1, -2] / dist_x[-1]
        a_eb = face_areas_x[1:-1, -1] / (lx - xc[-1])
        a_s_right = face_areas_y[1:-1, -1] / dist_y[:-1]
        a_n_right = face_areas_y[2:, -1] / dist_y[1:]
    
        f_w_right = a_w_right * pn[1:-1, -2]
        f_eb = a_eb * right[1:-1]
        f_s_right = a_s_right * pn[:-2, -1]
        f_n_right = a_n_right * pn[2:, -1]

        p[1:-1, -1] =(f_eb + f_w_right + f_n_right + f_s_right - b[1:-1, -1] * cell_volumes[1:-1, -1]) / (a_w_right + a_eb + a_s_right + a_n_right)

        a_wb_bottom = face_areas_x[0, 0] / xc[0]
        a_e_bottom_left = face_areas_x[0, 1] / dist_x[0]
        a_sb_left = face_areas_y[0, 0] / yc[0]
        a_n_bottom_left = face_areas_y[1, 0] / dist_y[0]   

        f_wb_bottom = a_wb_bottom * left[0]
        f_e_bottom_left = a_e_bottom_left * pn[0, 1]
        f_sb_left = a_sb_left * bottom[0]
        f_n_bottom_left = a_n_bottom_left * pn[1, 0]

        p[0, 0] =(f_e_bottom_left + f_wb_bottom + f_n_bottom_left + f_sb_left - b[0, 0] * cell_volumes[0, 0]) / (a_wb_bottom + a_e_bottom_left + a_sb_left + a_n_bottom_left)

        a_wb_top= face_areas_x[-1, 0] / xc[0]
        a_e_top_left = face_areas_x[-1, 1] / dist_x[0]
        a_s_top_left = face_areas_y[-2, 0] / dist_y[-1]
        a_nb_left = face_areas_y[-1, 0] / (ly - yc[-1]) 

        f_wb_top = a_wb_top * left[-1]
        f_e_top_left = a_e_top_left * pn[-1, 1]
        f_s_top_left = a_s_top_left * pn[-2, 0]
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

        f_w_top_right = a_w_top_right * pn[-1, -2]
        f_eb_top = a_eb_top * right[-1]
        f_s_top_right = a_s_top_right * pn[-2, -1]
        f_nb_right = a_nb_right * top

        p[-1, -1] =(f_eb_top + f_w_top_right + f_nb_right + f_s_top_right - b[-1, -1] * cell_volumes[-1, -1]) / (a_w_top_right + a_eb_top + a_s_top_right + a_nb_right)

        
    return p, pn
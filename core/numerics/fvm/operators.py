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

    e_u = u**2 / 2
    e_v = v**2 / 2

    u_term = np.zeros_like(u)
    v_term = np.zeros_like(v)

    f_w_u = e_u[1:, :-1] * face_areas_x[1:, :-1]

    f_e_u = e_u[1:, 1:] * face_areas_x[1:, 1:]

    f_s_u = v[:-1, 1:] * u[:-1, 1:] * face_areas_y[:-1, 1:]

    f_n_u = v[1:, 1:] * u[1:, 1:] * face_areas_y[1:, 1:]

    f_w_v = u[1:, :-1] * v[1:, :-1] * face_areas_x[1:, :-1]

    f_e_v = u[1:, 1:] * v[1:, 1:] * face_areas_x[1:, 1:]

    f_s_v = e_v[:-1, 1:] * face_areas_y[:-1, 1:]

    f_n_v = e_v[1:, 1:] * face_areas_y[1:, 1:]

    u_term[1:, 1:] = dt * (f_e_u - f_w_u) / cell_volumes[1:, 1:] + dt * (f_n_u - f_s_u) / cell_volumes[1:, 1:]

    v_term[1:, 1:] = dt * (f_e_v - f_w_v) / cell_volumes[1:, 1:] + dt * (f_n_v - f_s_v) / cell_volumes[1:, 1:]

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
) -> np.ndarray:
    """Iteratively solve the Poisson equation for pressure correction in the 2D Navier-Stokes solver."""

    pn = p.copy()

    a_w = face_areas_x[1:-1, 1:-1] / dist_x[:-1]
    a_e = face_areas_x[1:-1, 2:] / dist_x[1:]
    a_s = face_areas_y[1:-1, 1:-1] / dist_y[:-1, None]
    a_n = face_areas_y[2:, 1:-1] / dist_y[1:, None]
    
    for q in range(nit):
        
        pn = p.copy()

        f_w = a_w * pn[1:-1, :-2]
        f_e = a_e * pn[1:-1, 2:]
        f_s = a_s * pn[:-2, 1:-1]
        f_n = a_n * pn[2:, 1:-1]

        p[1:-1, 1:-1] =(f_e + f_w + f_n + f_s - b[1:-1, 1:-1]) / (a_w + a_e + a_s + a_n)

        # p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
        # p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        # p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        # p[-1, :] = 0        # p = 0 at y = 2
        
    return p, pn
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
    hx: float,
    hy: float,
    dt: float,
) -> np.ndarray:
    """Compute the 2D upwind advection term for a constant wave speed."""

    term = np.zeros_like(u)

    f_w = c * u[1:, :-1]

    f_e = c * u[1:, 1:]

    f_s = c * u[:-1, 1:]

    f_n = c * u[1:, 1:]    

    term[1:, 1:] = dt * (f_e - f_w) / hx[1:] + dt * (f_n - f_s) / hy[1:]

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

    f_s_u = v[:-1, 1:] * u[:-1, 1:] * face_areas_x[:-1, 1:]

    f_n_u = v[1:, 1:] * u[1:, 1:] * face_areas_x[1:, 1:]

    f_w_v = u[1:, :-1] * v[1:, :-1] * face_areas_y[1:, :-1]

    f_e_v = u[1:, 1:] * v[1:, 1:] * face_areas_y[1:, 1:]

    f_s_v = e_v[:-1, 1:] * face_areas_y[:-1, 1:]

    f_n_v = e_v[1:, 1:] * face_areas_y[1:, 1:]

    u_term[1:, 1:] = dt * (f_e_u - f_w_u) / cell_volumes[1:, 1:] + dt * (f_n_u - f_s_u) / cell_volumes[1:, 1:]

    v_term[1:, 1:] = dt * (f_e_v - f_w_v) / cell_volumes[1:, 1:] + dt * (f_n_v - f_s_v) / cell_volumes[1:, 1:]

    return u_term, v_term
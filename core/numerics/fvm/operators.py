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

    f_w = c * u[1:] / hx[1:]

    f_e = c * u[:-1] / hx[1:]

    term[1:] = dt * (f_w - f_e)

    return term


def compute_convection_1d_term(
    u: np.ndarray,
    hx: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute the 1D upwind advection term for a constant wave speed."""

    e = u**2 / 2

    term = np.zeros_like(u)

    f_w = e[1:] / hx[1:]

    f_e = e[:-1] / hx[1:]

    term[1:] = dt * (f_w - f_e)

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
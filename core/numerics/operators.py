import numpy as np


def compute_advection_1d_term(
    u: np.ndarray,
    c: float,
    dx: float,
    dt: float,
) -> np.ndarray:

    term = np.zeros_like(u)

    term[1:] = c * dt / dx * (u[1:] - u[:-1])

    return term


def compute_convection_1d_term(
    u: np.ndarray,
    dx: float,
    dt: float,
) -> np.ndarray:

    term = np.zeros_like(u)

    term[1:] = u[1:] * dt / dx * (u[1:] - u[:-1])

    return term


def compute_diffusion_1d_term(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
) -> np.ndarray:

    term = np.zeros_like(u)

    term[1:-1] = nu * dt / dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2])

    return term
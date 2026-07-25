"""Boundary condition updates for finite-difference solvers."""



import numpy as np



def apply_diffusion_boundary_1d(
    u: np.ndarray,
    un: np.ndarray,
    dt: float,
    dx: float,
    nu: float,
) -> None:
    """Apply boundary updates for the 1D diffusion equation."""

    u[0] = un[0] + nu * dt / dx**2 * (un[1] - 2 * un[0] + un[-1])
    u[-1] = un[-1] + nu * dt / dx**2 * (un[0] - 2 * un[-1] + un[-2])
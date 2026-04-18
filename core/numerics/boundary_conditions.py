"""Boundary condition updates for finite-difference solvers."""



import numpy as np



def apply_periodic_diffusion_boundary_1d(
    u: np.ndarray,
    un: np.ndarray,
    dt: float,
    dx: float,
    nu: float,
) -> None:
    """Apply periodic boundary updates for the 1D diffusion equation."""

    u[0] = un[0] + nu * dt / dx**2 * (un[1] - 2 * un[0] + un[-1])
    u[-1] = un[-1] + nu * dt / dx**2 * (un[0] - 2 * un[-1] + un[-2])

    
def apply_periodic_burgers_boundary_1d(
    u: np.ndarray,
    un: np.ndarray,
    dt: float,
    dx: float,
    nu: float,
) -> None:
    """Apply periodic boundary updates for the 1D Burgers' equation."""

    u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) \
        + nu * dt / dx**2 * (un[1] - 2 * un[0] + un[-2])
    
    # Treat the last grid point as the periodic duplicate of the first,
    # so un[-2] is the last distinct neighbor and u[-1] is reset to u[0].
    u[-1] = un[0]


def apply_periodic_advection_boundary_2d(
    u: np.ndarray,
    u_min: float,
) -> None:
    """Apply periodic boundary updates for the 2D advection equation."""

    u[0, :] = u_min
    u[-1, :] = u_min
    u[:, 0] = u_min
    u[:, -1] = u_min


def apply_periodic_convection_boundary_2d(
    u: np.ndarray,
    v: np.ndarray,
    u_min: float,
    v_min: float,
) -> None:
    """Apply periodic boundary updates for the 2D convection equation."""

    u[0, :] = u_min
    u[-1, :] = u_min
    u[:, 0] = u_min
    u[:, -1] = u_min

    v[0, :] = v_min
    v[-1, :] = v_min
    v[:, 0] = v_min
    v[:, -1] = v_min


def apply_periodic_diffusion_boundary_2d(
    u: np.ndarray,
    u_min: float,
) -> None:
    """Apply periodic boundary updates for the 2D diffusion equation."""

    u[0, :] = u_min
    u[-1, :] = u_min
    u[:, 0] = u_min
    u[:, -1] = u_min
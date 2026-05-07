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

    
def apply_burgers_boundary_1d(
    u: np.ndarray,
    un: np.ndarray,
    dt: float,
    dx: float,
    nu: float,
) -> None:
    """Apply boundary updates for the 1D Burgers' equation."""

    u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) \
        + nu * dt / dx**2 * (un[1] - 2 * un[0] + un[-2])
    
    # Treat the last grid point as the periodic duplicate of the first,
    # so un[-2] is the last distinct neighbor and u[-1] is reset to u[0].
    u[-1] = un[0]


def apply_advection_boundary_2d(
    u: np.ndarray,
    u_min: float,
) -> None:
    """Apply boundary updates for the 2D advection equation."""

    u[0, :] = u_min
    u[-1, :] = u_min
    u[:, 0] = u_min
    u[:, -1] = u_min


def apply_convection_boundary_2d(
    u: np.ndarray,
    v: np.ndarray,
    u_min: float,
    v_min: float,
) -> None:
    """Apply boundary updates for the 2D convection equation."""

    u[0, :] = u_min
    u[-1, :] = u_min
    u[:, 0] = u_min
    u[:, -1] = u_min

    v[0, :] = v_min
    v[-1, :] = v_min
    v[:, 0] = v_min
    v[:, -1] = v_min


def apply_diffusion_boundary_2d(
    u: np.ndarray,
    u_min: float,
) -> None:
    """Apply boundary updates for the 2D diffusion equation."""

    u[0, :] = u_min
    u[-1, :] = u_min
    u[:, 0] = u_min
    u[:, -1] = u_min


def apply_laplace_boundary_2d(
    p: np.ndarray,
    bottom: float | np.ndarray,
    top: float | np.ndarray,
    right: float | np.ndarray,
    left: float | np.ndarray,
) -> None:
    """Apply boundary updates for the 2D Laplace equation."""

    p[:, 0] = left  # p = left @ x = 0
    p[:, -1] = right  # p = right @ x = 2
    p[0, :] = bottom  # p = bottom @ y = 0
    p[-1, :] = top  # p = top @ y = 1


def apply_poisson_boundary_2d(
    p: np.ndarray,
) -> None:
    """Apply boundary updates for the 2D Laplace equation."""

    p[0, :] = 0
    p[1, :] = 0
    p[:, 0] = 0
    p[:, -1] = 0


def apply_cavity_flow_boundary_2d(
    u: np.ndarray,
    v: np.ndarray,
    u_lid: float,
) -> None:
    """Apply boundary updates for the 2D cavity flow equation."""

    u[0, :]  = 0
    u[:, 0]  = 0
    u[:, -1] = 0
    u[-1, :] = u_lid

    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0
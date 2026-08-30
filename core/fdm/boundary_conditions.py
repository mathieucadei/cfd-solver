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


def apply_periodic_source_boundary_2d(
    b: np.ndarray,
    rho: float,
    dt: float,
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute the 2D source term for the Poisson equation in the 2D Navier-Stokes solver."""

    # Periodic BC Pressure @ x = outlet
    b[1:-1, -1] = (rho * (1 / dt * ((u[1:-1, 0] - u[1:-1,-2]) / (2 * dx) +
                                    (v[2:, -1] - v[0:-2, -1]) / (2 * dy)) -
                          ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 -
                          2 * ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) *
                               (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx)) -
                          ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2))
    
    # Periodic BC Pressure @ x = inlet
    b[1:-1, 0] = (rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) +
                                   (v[2:, 0] - v[0:-2, 0]) / (2 * dy)) -
                         ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 -
                         2 * ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) *
                              (v[1:-1, 1] - v[1:-1, -1]) / (2 * dx))-
                         ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2))   

    return b


def apply_periodic_pressure_poisson_boundary_2d(
    b: np.ndarray,
    p: np.ndarray,
    pn: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute the 2D source term for the Poisson equation in the 2D Navier-Stokes solver."""

    # Periodic BC Pressure @ x = outlet
    p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2])* dy**2 +
                    (pn[2:, -1] + pn[0:-2, -1]) * dx**2) /
                    (2 * (dx**2 + dy**2)) -
                    dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, -1])

    # Periodic BC Pressure @ x = inlet
    p[1:-1, 0] = (((pn[1:-1, 1] + pn[1:-1, -1])* dy**2 +
                    (pn[2:, 0] + pn[0:-2, 0]) * dx**2) /
                    (2 * (dx**2 + dy**2)) -
                    dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0])
    
    # Wall boundary conditions, pressure
    p[-1, :] =p[-2, :]  # dp/dy = 0 at y = 2
    p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
    
    return p


def apply_periodic_channel_flow_boundary_2d(
        u: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        un: np.ndarray,
        vn: np.ndarray,
        nu: np.ndarray,
        rho: np.ndarray,
        source: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        dt: np.ndarray,
) -> np.ndarray:
    
    # Periodic BC u @ x = 2     
    u[1:-1, -1] = (un[1:-1, -1] - un[1:-1, -1] * dt / dx * 
                (un[1:-1, -1] - un[1:-1, -2]) -
                vn[1:-1, -1] * dt / dy * 
                (un[1:-1, -1] - un[0:-2, -1]) -
                dt / (2 * rho * dx) *
                (p[1:-1, 0] - p[1:-1, -2]) + 
                nu * (dt / dx**2 * 
                (un[1:-1, 0] - 2 * un[1:-1,-1] + un[1:-1, -2]) +
                dt / dy**2 * 
                (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + source * dt)
    
    # Periodic BC u @ x = 0
    u[1:-1, 0] = (un[1:-1, 0] - un[1:-1, 0] * dt / dx *
                (un[1:-1, 0] - un[1:-1, -1]) -
                vn[1:-1, 0] * dt / dy * 
                (un[1:-1, 0] - un[0:-2, 0]) - 
                dt / (2 * rho * dx) * 
                (p[1:-1, 1] - p[1:-1, -1]) + 
                nu * (dt / dx**2 * 
                (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                dt / dy**2 *
                (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + source * dt)

    # Periodic BC v @ x = 2
    v[1:-1, -1] = (vn[1:-1, -1] - un[1:-1, -1] * dt / dx *
                (vn[1:-1, -1] - vn[1:-1, -2]) - 
                vn[1:-1, -1] * dt / dy *
                (vn[1:-1, -1] - vn[0:-2, -1]) -
                dt / (2 * rho * dy) * 
                (p[2:, -1] - p[0:-2, -1]) +
                nu * (dt / dx**2 *
                (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
                dt / dy**2 *
                (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))

    # Periodic BC v @ x = 0
    v[1:-1, 0] = (vn[1:-1, 0] - un[1:-1, 0] * dt / dx *
                (vn[1:-1, 0] - vn[1:-1, -1]) -
                vn[1:-1, 0] * dt / dy *
                (vn[1:-1, 0] - vn[0:-2, 0]) -
                dt / (2 * rho * dy) * 
                (p[2:, 0] - p[0:-2, 0]) +
                nu * (dt / dx**2 * 
                (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
                dt / dy**2 * 
                (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))   

    # Wall BC: u,v = 0 @ y = 0,2
    u[0, :] = 0
    u[-1, :] = 0
    v[0, :] = 0
    v[-1, :] = 0
"""Reusable finite-difference operators for 1D & 2D transport equations."""



import numpy as np



def compute_advection_1d_term(
    u: np.ndarray,
    c: float,
    dx: float,
    dt: float,
) -> np.ndarray:
    """Compute the 1D upwind advection term for a constant wave speed."""

    term = np.zeros_like(u)

    term[1:] = c * dt / dx * (u[1:] - u[:-1])

    return term


def compute_convection_1d_term(
    u: np.ndarray,
    dx: float,
    dt: float,
) -> np.ndarray:
    """Compute the 1D upwind convection term."""
    term = np.zeros_like(u)

    term[1:] = u[1:] * dt / dx * (u[1:] - u[:-1])

    return term


def compute_diffusion_1d_term(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
) -> np.ndarray:
    """Compute the 1D central-difference diffusion term."""
    term = np.zeros_like(u)

    term[1:-1] = nu * dt / dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2])

    return term


def compute_advection_2d_term(
    u: np.ndarray,
    c: float,
    dx: float,
    dy: float,
    dt: float,
) -> np.ndarray:
    """Compute the 2D upwind advection term for a constant wave speed."""

    term = np.zeros_like(u)

    term[1:, 1:] = c * dt / dx * (u[1:, 1:] - u[1:, :-1]) + c * dt / dy * (u[1:, 1:] - u[:-1, 1:])

    return term

def compute_convection_2d_term(
    u: np.ndarray,
    v: float,
    dx: float,
    dy: float,
    dt: float,
) -> np.ndarray:
    """Compute the 2D upwind convection u & v terms"""

    u_term = np.zeros_like(u)
    v_term = np.zeros_like(v)

    u_term[1:, 1:] = u[1:, 1:] * dt / dx * (u[1:, 1:] - u[1:, :-1]) + v[1:, 1:] * dt / dy * (u[1:, 1:] - u[:-1, 1:])
    v_term[1:, 1:] = u[1:, 1:] * dt / dx * (v[1:, 1:] - v[1:, :-1]) + v[1:, 1:] * dt / dy * (v[1:, 1:] - v[:-1, 1:])

    return u_term, v_term


def compute_diffusion_2d_term(
    u: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    nu: float,
) -> np.ndarray:
    """Compute the 1D central-difference diffusion term."""

    term = np.zeros_like(u)

    term[1:-1, 1:-1] = nu * dt / dx**2 * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2]) + nu * dt / dy**2 * (u[2:,1: -1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1])

    return term


def compute_laplace_2d_term(
    p: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute the 2D central-difference Laplace term."""

    term = np.zeros_like(p)

    term[1:-1, 1:-1] = (p[1:-1, 2:] + p[1:-1, 0:-2]) * dy**2 + (p[2:, 1:-1] + p[0:-2, 1:-1]) * dx**2 / (2 * (dx**2 + dy**2))

    return term


def compute_source_term_2d(
    b: np.ndarray,
    rho: float,
    dt: float,
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute the 2D source term for the Poisson equation in the 2D Navier-Stokes solver."""

    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b


def compute_pressure_poisson_term(
        p: np.ndarray,
        b: np.ndarray,
        nit: int, 
        dx: float, 
        dy: float, 
) -> np.ndarray:
    """Iteratively solve the Poisson equation for pressure correction in the 2D Navier-Stokes solver."""

    pn = np.empty_like(p)
    pn = p.copy()
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])

        p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        p[-1, :] = 0        # p = 0 at y = 2
        
    return p
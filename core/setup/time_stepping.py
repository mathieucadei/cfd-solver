"""Time-step utilities for 1D numerical and analytical solvers."""



from .grids import compute_cole_hopf_dx, compute_dx, compute_dx_2d



def compute_convective_dt(config: object) -> float:
    """Compute the time step for 1D advection and convection problems."""

    dx = compute_dx(config)
    
    return config.sigma * dx


def compute_diffusive_dt(config: object) -> float:
    """Compute the time step for 1D diffusion-dominated problems."""

    dx = compute_dx(config)
    
    return config.sigma * dx**2 / config.viscosity


def compute_cole_hopf_dt(config: object) -> float:
    """Compute the time step for the 1D Cole-Hopf analytical solution."""

    dx = compute_cole_hopf_dx(config)
    
    return dx * config.viscosity


def compute_convective_dt_2d(config: object) -> float:
    """Compute the time step for 1D advection and convection problems."""

    dx = compute_dx_2d(config)
    
    return config.sigma * dx
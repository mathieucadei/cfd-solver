import numpy as np

from .grids import compute_dx, compute_cole_hopf_dx

def compute_dt(config: object) -> float:
    """Computes the time step based on the provided configuration."""

    dx = compute_dx(config)
    
    return config.sigma * dx**2 / config.viscosity

def compute_cole_hopf_dt(config: object) -> float:
    """Computes the time step for the Cole-Hopf equation based on the provided configuration."""

    dx = compute_cole_hopf_dx(config)
    
    return dx * config.viscosity
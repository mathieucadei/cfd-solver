"""Time-step utilities for 1D & 2D numerical and analytical solvers."""


import numpy as np

from ..grids import compute_cole_hopf_dx, compute_dx, compute_dy
from .mesh import build_mesh



def compute_advective_dt_1d(config: object) -> float:
    """Compute the time step for 1D advection problem."""

    mesh = build_mesh(config)

    hx = mesh['hx']

    hx_min = np.min(hx)
    
    return config.sigma * hx_min / config.wavespeed
"""Time-step utilities for 1D & 2D numerical and analytical solvers."""


import numpy as np

from ..grids import compute_cole_hopf_dx, compute_dx, compute_dy
from .mesh import build_hx_spacing, build_x_centers, build_h_spacing



def compute_advective_dt_1d_fvm(config: object) -> float:
    """Compute the time step for 1D advection problem."""

    hx = build_hx_spacing(config)

    hx_min = np.min(hx)
    
    return config.sigma * hx_min / config.wavespeed


def compute_convection_dt_1d_fvm(config: object) -> float:
    """Compute the time step for 1D convection problem."""

    hx = build_hx_spacing(config)

    hx_min = np.min(hx)
    
    return config.sigma * hx_min / config.u_max


def compute_diffusive_dt_1d_fvm(config: object) -> float:
    """Compute the time step for 1D diffusion-dominated problems."""

    hx = build_hx_spacing(config)

    hx_min = np.min(hx)
    
    return config.sigma * hx_min**2 / config.viscosity


def compute_cole_hopf_dt_1d_fvm(config: object) -> float:
    """Compute the time step for the 1D Cole-Hopf analytical solution."""

    hx = build_hx_spacing(config)

    hx_min = np.min(hx)
    
    return hx_min * config.viscosity


def compute_advective_dt_2d_fvm(config: object) -> float:
    """Compute the time step for 2D advection problem."""

    hx, hy = build_h_spacing(config)

    hx_min = np.min(hx)
    hy_min = np.min(hy)
    
    return config.sigma * min(hx_min, hy_min) / config.wavespeed
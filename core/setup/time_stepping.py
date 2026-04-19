"""Time-step utilities for 1D & 2D numerical and analytical solvers."""


import numpy as np

from .grids import compute_cole_hopf_dx, compute_dx, compute_dy



def compute_advective_dt_1d(config: object) -> float:
    """Compute the time step for 1D advection problem."""

    dx = compute_dx(config)
    
    return config.sigma * dx / config.wavespeed


def compute_convective_dt_1d(config: object) -> float:
    """Compute the time step for 1D convection problem."""

    dx = compute_dx(config)
    
    return config.sigma * dx / config.u_max


def compute_diffusive_dt_1d(config: object) -> float:
    """Compute the time step for 1D diffusion-dominated problems."""

    dx = compute_dx(config)
    
    return config.sigma * dx**2 / config.viscosity


def compute_cole_hopf_dt_1d(config: object) -> float:
    """Compute the time step for the 1D Cole-Hopf analytical solution."""

    dx = compute_cole_hopf_dx(config)
    
    return dx * config.viscosity


def compute_advective_dt_2d(config: object) -> float:
    """Compute the time step for 2D advection problem."""

    dx = compute_dx(config)
    
    return config.sigma * dx / config.wavespeed


def compute_convective_dt_2d(config: object) -> float:
    """Compute the time step for 2D convection problem."""

    dx = compute_dx(config)
    
    return config.sigma * dx / np.max(config.u_max, config.v_max)


def compute_diffusive_dt_2d(config: object) -> float:
    """Compute the time step for 2D diffusion-dominated problems."""

    dx = compute_dx(config)
    dy = compute_dy(config)
    
    return config.sigma * dx * dy / config.viscosity
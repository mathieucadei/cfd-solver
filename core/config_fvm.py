"""Configuration dataclasses for 1D & 2D numerical and analytical simulations."""


import numpy as np

from dataclasses import dataclass, field



@dataclass
class Advection1DFVMConfig:
    domain_length_x: float = 2.0
    num_cells_x: int = 100
    expansion_ratio_x: float = 1.
    max_iterations: int = 40
    sigma: float = 1.0
    wavespeed: float = 1.0
    hat_start: float = 0.5
    hat_end: float = 1.0
    u_min: float = 1.0
    u_max: float = 2.0


@dataclass
class Convection1DFVMConfig:
    domain_length_x: float = 2.0
    num_cells_x: int = 100
    expansion_ratio_x: float = 1.
    max_iterations: int = 40
    sigma: float = 1.0
    hat_start: float = 0.5
    hat_end: float = 1.0
    u_min: float = 1.0
    u_max: float = 2.0


@dataclass
class Diffusion1DFVMConfig:
    """Configuration parameters for the 1D diffusion equation."""
    domain_length_x: float = 2.0
    num_grid_points_x: int = 41
    max_iterations: int = 41
    sigma: float = 0.2
    viscosity: float = 0.3
    hat_start: float = 0.5
    hat_end: float = 1.0
    u_min: float = 1.0
    u_max: float = 2.0
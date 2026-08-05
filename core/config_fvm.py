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
    num_cells_x: int = 41
    expansion_ratio_x: float = 0.
    max_iterations: int = 41
    sigma: float = 0.2
    viscosity: float = 0.3
    hat_start: float = 0.5
    hat_end: float = 1.0
    u_min: float = 1.0
    u_max: float = 2.0


@dataclass
class BurgersEquation1DFVMConfig:
    """Configuration parameters for the 1D Burgers' equation."""
    domain_length_x: float = 2.0
    num_cells_x: int = 101
    expansion_ratio_x: float = 0.
    max_iterations: int = 100
    time_step: float = 0.0025
    sigma: float = 0.2
    viscosity: float = 0.07
    grid_type: str = "hat"
    hat_start: float = 0.5
    hat_end: float = 1.0
    u_min: float = 1.0
    u_max: float = 2.0


@dataclass
class Advection2DFVMConfig:
    """Configuration parameters for the 2D linear advection equation."""
    domain_length_x: float = 2.0
    domain_length_y: float = 2.0
    num_cells_x: int = 81
    num_cells_y: int = 81
    expansion_ratio_x: float = 0.
    expansion_ratio_y: float = 0.
    max_iterations: int = 25
    sigma: float = 1
    wavespeed: float = 1.0
    hat_start_x: float = 0.5
    hat_start_y: float = 0.5
    hat_end_x: float = 1.0
    hat_end_y: float = 1.0
    u_min: float = 1.0
    u_max: float = 2.0


@dataclass
class Convection2DFVMConfig:
    """Configuration parameters for the 2D linear advection equation."""
    domain_length_x: float = 2.0
    domain_length_y: float = 2.0
    num_cells_x: int = 81
    num_cells_y: int = 81
    expansion_ratio_x: float = 0.
    expansion_ratio_y: float = 0.
    max_iterations: int = 25
    sigma: float = 1
    hat_start_x: float = 0.5
    hat_start_y: float = 0.5
    hat_end_x: float = 1.0
    hat_end_y: float = 1.0
    u_min: float = 1.0
    u_max: float = 2.0
    v_min: float = 1.0
    v_max: float = 2.0


@dataclass
class Diffusion2DFVMConfig:
    """Configuration parameters for the 2D diffusion equation."""
    domain_length_x: float = 2.0
    domain_length_y: float = 2.0
    num_cells_x: int = 30
    num_cells_y: int = 30
    expansion_ratio_x: float = 0.
    expansion_ratio_y: float = 0.
    max_iterations: int = 41
    sigma: float = 0.25
    viscosity: float = 0.05
    hat_start_x: float = 0.5
    hat_start_y: float = 0.5
    hat_end_x: float = 1.0
    hat_end_y: float = 1.0
    u_min: float = 1.0
    u_max: float = 2.0


@dataclass
class BurgersEquation2DFVMConfig:
    """Configuration parameters for the 2D Burgers equation."""
    domain_length_x: float = 2.0
    domain_length_y: float = 2.0
    num_cells_x: int = 30
    num_cells_y: int = 30
    expansion_ratio_x: float = 0.
    expansion_ratio_y: float = 0.
    max_iterations: int = 120
    sigma: float = 0.0009
    viscosity: float = 0.01
    hat_start_x: float = 0.5
    hat_start_y: float = 0.5
    hat_end_x: float = 1.0
    hat_end_y: float = 1.0
    u_min: float = 1.0
    u_max: float = 2.0
    v_min: float = 1.0
    v_max: float = 2.0
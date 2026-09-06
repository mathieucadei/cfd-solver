"""Configuration dataclasses for 1D & 2D numerical and analytical simulations."""


import numpy as np

from dataclasses import dataclass, field



@dataclass
class Advection1DConfig:
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
class Convection1DConfig:
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
class Diffusion1DConfig:
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
class BurgersEquation1DConfig:
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
class Advection2DConfig:
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
class Convection2DConfig:
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
class Diffusion2DConfig:
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
class BurgersEquation2DConfig:
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


@dataclass
class Laplace2DConfig:
    """Configuration parameters for the 2D Laplace equation."""
    domain_length_x: float = 2.0
    domain_length_y: float = 1.0
    num_cells_x: int = 30
    num_cells_y: int = 30
    expansion_ratio_x: float = 0.
    expansion_ratio_y: float = 0.
    l1_norm_target: float = 1e-4


@dataclass
class SourceTerm:
    x: float
    y: float
    value: float


@dataclass
class Poisson2DConfig:
    domain_length_x: float = 2.0
    domain_length_y: float = 1.0
    num_cells_x: int = 31
    num_cells_y: int = 31
    expansion_ratio_x: float = 0.
    expansion_ratio_y: float = 0.
    max_iterations: int = 100
    pressure_init: float = 0.0
    source_terms: list[SourceTerm] = field(default_factory=lambda: [
        SourceTerm(x=0.3, y=0.5, value=10.0),
        SourceTerm(x=0.6, y=0.2, value=-5.0),
    ])
    l1_norm_target: float = 1e-4


@dataclass
class CavityFlowConfig:
    domain_length_x: float = 2.0
    domain_length_y: float = 1.0
    num_cells_x: int = 40
    num_cells_y: int = 40
    expansion_ratio_x: float = 0.
    expansion_ratio_y: float = 0.
    max_iterations: int = 500
    max_pseudo_iterations: int = 50
    time_step: float = 0.001
    u_lid: float = 1.0
    density: float = 1.0
    viscosity: float = 0.1


@dataclass
class ChannelFlowConfig:
    domain_length_x: float = 2.0
    domain_length_y: float = 1.0
    num_cells_x: int = 40
    num_cells_y: int = 40
    expansion_ratio_x: float = 0.
    expansion_ratio_y: float = 0.
    max_iterations: int = 10
    max_pseudo_iterations: int = 50
    time_step: float = 0.001
    source: float = 1.0
    density: float = 1.0
    viscosity: float = 0.1
    u_l1_norm_target: float = 0.001
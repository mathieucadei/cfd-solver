"""Configuration dataclasses for 1D numerical and analytical simulations."""



from dataclasses import dataclass



@dataclass
class Advection1DConfig:
    """Configuration parameters for the 1D linear advection equation."""
    domain_length: float = 2.0
    num_grid_points: int = 81
    max_iterations: int = 25
    sigma: float = 1
    wavespeed: float = 1.0
    hat_start: float = 0.5
    hat_end: float = 1.0
    u_min: float = 1.0
    u_max: float = 2.0


@dataclass
class Convection1DConfig:
    """Configuration parameters for the 1D nonlinear convection equation."""
    domain_length: float = 2.0
    num_grid_points: int = 101
    max_iterations: int = 100
    sigma: float = 0.2
    hat_start: float = 0.5
    hat_end: float = 1.0
    u_min: float = 1.0
    u_max: float = 2.0


@dataclass
class Diffusion1DConfig:
    """Configuration parameters for the 1D diffusion equation."""
    domain_length: float = 2.0
    num_grid_points: int = 41
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
    domain_length: float = 2.0
    num_grid_points: int = 101
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
    """Configuration parameters for the 1D linear advection equation."""
    domain_length_x: float = 2.0
    domain_length_y: float = 2.0
    num_grid_points_x: int = 81
    num_grid_points_y: int = 81
    max_iterations: int = 25
    sigma: float = 1
    wavespeed: float = 1.0
    hat_start_x: float = 0.5
    hat_start_y: float = 0.5
    hat_end_x: float = 1.0
    hat_end_y: float = 1.0
    u_min: float = 1.0
    u_max: float = 2.0
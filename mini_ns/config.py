from dataclasses import dataclass

@dataclass
class Advection1DConfig:
    """Configuration for the Navier-Stokes solver."""
    domain_length: float = 2.0
    num_grid_points: int = 81
    max_iterations: int = 25
    time_step: float = 0.025
    wavespeed: float = 1.0
    hat_start: float = 0.5
    hat_end: float = 1.0
    u_min: float = 1.0
    u_max: float = 2.0
    # viscosity: float = 0.01

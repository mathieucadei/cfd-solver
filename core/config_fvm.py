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
from .config import Advection1DConfig, Convection1DConfig, Diffusion1DConfig
from .grids import make_1d_grid, compute_1d_grid_points_spacing
from .time_step import compute_time_step
from .initial_conditions import hat_initial_condition
from .num_mod import solve_advection_1d, solve_convection_1d, solve_diffusion_1d
from .ana_mod import solve_heat_equation_1d
from .signal_processing import (
    generate_mode_indices,
    compute_coefficients,
    compute_series_terms,
    compute_series
)

from .plotting import plot_snapshots, plot_animation
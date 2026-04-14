from .config import Advection1DConfig, Convection1DConfig, Diffusion1DConfig, BurgersEquation1DConfig
from .setup.grids import make_1d_grid, make_cole_hopf_1d_grid, compute_dx
from .setup.time_stepping import compute_dt, compute_cole_hopf_dt
from .setup.initial_conditions import hat_initial_condition, cole_hopf_initial_condition
from .numerics import solve_advection_1d, solve_convection_1d, solve_diffusion_1d, solve_burgers_equation_1d
from .analytical import solve_heat_equation_1d, cole_hopf_1d_ufunc, solve_cole_hopf_1d
from .signal_processing import (
    generate_mode_indices,
    compute_coefficients,
    compute_series_terms,
    compute_series
)
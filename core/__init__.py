from .config import Advection1DConfig, Advection2DConfig, Convection1DConfig, Convection2DConfig, Diffusion1DConfig, Diffusion2DConfig, BurgersEquation1DConfig
from .setup.grids import make_1d_grid, make_cole_hopf_1d_grid, compute_dx, compute_dx_2d, compute_dy_2d
from .setup.time_stepping import compute_convective_dt, compute_diffusive_dt, compute_advective_dt_2d, compute_convective_dt_2d, compute_cole_hopf_dt
from .setup.initial_conditions import hat_initial_condition_1d, hat_initial_condition_2d, hat_convective_initial_condition_2d, cole_hopf_initial_condition
from .numerics import solve_advection_1d, solve_advection_2d, solve_convection_1d, solve_convection_2d, solve_diffusion_1d, solve_diffusion_2d, solve_burgers_equation_1d
from .analytical import solve_heat_equation_1d, cole_hopf_1d_ufunc, solve_cole_hopf_1d
from .signal_processing import (
    generate_mode_indices,
    compute_coefficients,
    compute_series_terms,
    compute_series
)
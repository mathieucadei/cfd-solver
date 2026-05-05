from .config import (
    Advection1DConfig, 
    Advection2DConfig, 
    Convection1DConfig, 
    Convection2DConfig, 
    Diffusion1DConfig, 
    Diffusion2DConfig, 
    BurgersEquation1DConfig, 
    BurgersEquation2DConfig,
    Laplace2DConfig,
)

from .setup.grids import (
    compute_cole_hopf_dx, 
    compute_dx, 
    make_cole_hopf_x_grid, 
    make_x_grid,
    make_y_grid,  
)

from .setup.time_stepping import (
    compute_advective_dt_1d,
    compute_advective_dt_2d,
    compute_cole_hopf_dt_1d,  
    compute_convective_dt_1d, 
    compute_convective_dt_2d, 
    compute_diffusive_dt_1d, 
    compute_diffusive_dt_2d, 
)

from .setup.initial_conditions import (
    cole_hopf_initial_condition_1d,
    hat_convective_initial_condition_2d, 
    hat_initial_condition_1d, 
    hat_initial_condition_2d, 
    laplace_initial_condition_2d,
)

from .numerics import (
    solve_advection_1d, 
    solve_advection_2d,
    solve_burgers_equation_1d, 
    solve_burgers_equation_2d, 
    solve_convection_1d, 
    solve_convection_2d, 
    solve_diffusion_1d, 
    solve_diffusion_2d,
    solve_laplace_2d,
)


from .signal_processing import (
    compute_coefficients,
    compute_series_terms,
    compute_series,
    generate_mode_indices,
)

from .analytical import cole_hopf_1d_ufunc, solve_cole_hopf_1d, solve_heat_equation_1d
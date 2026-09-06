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
    SourceTerm,
    Poisson2DConfig,
    CavityFlowConfig,
    ChannelFlowConfig,
)

from .mesh import (
    build_hx_spacing,
    build_x_face_positions,
    build_x_centers,
    build_cole_hopf_hx_spacing,
    build_cole_hopf_x_face_positions,
    build_cole_hopf_x_centers,
    build_cole_hopf_dist_x,
    build_h_spacing,
    build_face_positions,
    build_centers,
)

from .time_stepping import (
    compute_advective_dt_1d,
    compute_advective_dt_2d,
    compute_cole_hopf_dt_1d,  
    compute_convection_dt_1d, 
    compute_convective_dt_2d, 
    compute_diffusive_dt_1d, 
    compute_diffusive_dt_2d, 
)
              
from .initial_conditions import (
    cole_hopf_initial_condition_1d,
    hat_convective_initial_condition_2d, 
    hat_initial_condition_1d,
    hat_initial_condition_2d,
    laplace_initial_condition_2d,
    poisson_initial_condition_2d,
    cavity_flow_initial_condition,
    channel_flow_initial_condition,
)

from .solvers import (
    solve_advection_1d,
    solve_advection_2d,
    solve_convection_1d,
    solve_convection_2d,
    solve_diffusion_1d,
    solve_diffusion_2d,
    solve_burgers_equation_1d,
    solve_burgers_equation_2d,
    solve_laplace_2d,
    solve_poisson_2d,
    solve_cavity_flow,
    solve_channel_flow,
)
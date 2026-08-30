"""Numerical solver for the 2D diffusion equation."""



import numpy as np

from ..operators import compute_convection_2d_term, compute_diffusion_2d_term, compute_source_term_2d, compute_pressure_poisson_term
from ..boundary_conditions import apply_cavity_flow_boundary_2d

from ..config import CavityFlowConfig
from ..grids import compute_dx, compute_dy



def solve_cavity_flow(
    initial_condition: np.ndarray,
    config: CavityFlowConfig,
) -> np.ndarray:
    """Solve the 2D cavity flow equation with an explicit central finite-difference scheme."""

    nu = config.viscosity
    rho = config.density

    dx = compute_dx(config)
    dy = compute_dy(config)
    dt = config.time_step

    u = initial_condition[0]
    v = initial_condition[1]
    p = initial_condition[2]
    b = initial_condition[3]

    un = np.empty_like(u)
    vn = np.empty_like(v)
    pn = np.empty_like(p)
    bn = np.empty_like(b)

    u_history = np.zeros((config.max_iterations + 1, config.num_grid_points_y, config.num_grid_points_x))
    v_history = np.zeros((config.max_iterations + 1, config.num_grid_points_y, config.num_grid_points_x))
    p_history = np.zeros((config.max_iterations + 1, config.num_grid_points_y, config.num_grid_points_x))

    u_history[0] = initial_condition[0]
    v_history[0] = initial_condition[1]
    p_history[0] = initial_condition[2]

    for n in range(1, config.max_iterations + 1):

        un = u.copy()
        vn = v.copy()
        pn = p.copy()
        bn = b.copy()


        convection_u_term, convection_v_term = compute_convection_2d_term(un, vn, dx, dy, config.time_step)
        diffusion_u_term = compute_diffusion_2d_term(un, dx, dy, config.time_step, config.viscosity)
        diffusion_v_term = compute_diffusion_2d_term(vn, dx, dy, config.time_step, config.viscosity)
        b = compute_source_term_2d(bn, config.density, config.time_step, un, vn, dx, dy)
        p = compute_pressure_poisson_term(pn, b, config.max_pseudo_iterations, dx, dy)[0]

        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         convection_u_term[1:-1, 1:-1] -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) + 
                         diffusion_u_term[1:-1, 1:-1])

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        convection_v_term[1:-1, 1:-1] -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         diffusion_v_term[1:-1, 1:-1])
        
        apply_cavity_flow_boundary_2d(u, v, config.u_lid)
        
        u_history[n] = u
        v_history[n] = v
        p_history[n] = p
    
    return u_history, v_history, p_history
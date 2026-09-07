"""Numerical solver for the 2D diffusion equation."""



import numpy as np

from ..operators import compute_convection_2d_term, compute_diffusion_2d_term, compute_source_term_2d, compute_pressure_poisson_term
from ..boundary_conditions import apply_periodic_source_boundary_2d, apply_periodic_pressure_poisson_boundary_2d, apply_periodic_channel_flow_boundary_2d

from ..config import ChannelFlowConfig
from ..grids import compute_dx, compute_dy



def solve_channel_flow(
    initial_condition: np.ndarray,
    config: ChannelFlowConfig,
) -> np.ndarray:
    """Solve the 2D channel flow equation with an explicit central finite-difference scheme."""

    nu = config.viscosity
    rho = config.density

    dx = compute_dx(config)
    dy = compute_dy(config)
    dt = config.time_step

    u_l1norm = 1

    u = initial_condition[0]
    v = initial_condition[1]
    p = initial_condition[2]
    b = initial_condition[3]

    un = np.empty_like(u)
    vn = np.empty_like(v)
    pn = np.empty_like(p)
    bn = np.empty_like(b)

    u_history = []
    v_history = []
    p_history = []

    while u_l1norm > config.u_l1_norm_target:

        un = u.copy()
        vn = v.copy()
        pn = p.copy()
        bn = b.copy()


        convection_u_term, convection_v_term = compute_convection_2d_term(un, vn, dx, dy, config.time_step)
        diffusion_u_term = compute_diffusion_2d_term(un, dx, dy, config.time_step, config.viscosity)
        diffusion_v_term = compute_diffusion_2d_term(vn, dx, dy, config.time_step, config.viscosity)
        b_term = compute_source_term_2d(bn, rho, dt, un, vn, dx, dy)
        p_term, pn_term = compute_pressure_poisson_term(pn, b_term, config.max_pseudo_iterations, dx, dy)

        b = apply_periodic_source_boundary_2d(b_term, rho, dt, un, vn, dx, dy)
        p = apply_periodic_pressure_poisson_boundary_2d(b, p_term, pn_term, dx, dy)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] - 
                        convection_u_term[1:-1, 1:-1] -
                        dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                        diffusion_u_term[1:-1, 1:-1] + 
                        config.source * dt)

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] - 
                        convection_v_term[1:-1, 1:-1] - 
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) + 
                        diffusion_v_term[1:-1, 1:-1])
        
        apply_periodic_channel_flow_boundary_2d(u, v, p, un, vn, nu, rho, config.source, dx, dy, dt)

        denominator = np.sum(np.abs(un))

        if denominator == 0:
            u_l1norm = np.sum(np.abs(u - un))
        
        else:
            u_l1norm = np.sum(np.abs(u - un)) / denominator
        
        u_history.append(u.copy())
        v_history.append(v.copy())
        p_history.append(p.copy())
    
    u_history_array = np.stack(u_history, axis=0)
    v_history_array = np.stack(v_history, axis=0)
    p_history_array = np.stack(p_history, axis=0)
    
    return u_history_array, v_history_array, p_history_array
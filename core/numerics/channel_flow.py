"""Numerical solver for the 2D diffusion equation."""



import numpy as np

from .operators import compute_convection_2d_term, compute_diffusion_2d_term, compute_source_term_2d, compute_pressure_poisson_term
from .boundary_conditions import apply_cavity_flow_boundary_2d, apply_periodic_source_boundary_2d, apply_periodic_pressure_poisson_boundary_2d

from ..config import ChannelFlowConfig
from ..setup.grids import compute_dx, compute_dy



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

    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = initial_condition[3]

    u_history = []
    v_history = []
    p_history = []

    while u_l1norm > config.u_l1_norm_target:

        un = u.copy()
        vn = v.copy()

        b_term = compute_source_term_2d(b, rho, dt, un, vn, dx, dy)
        p_term = compute_pressure_poisson_term(initial_condition[2], b, config.max_pseudo_iterations, dx, dy)[0]
        pn_term = compute_pressure_poisson_term(initial_condition[2], b, config.max_pseudo_iterations, dx, dy)[1]

        b = apply_periodic_source_boundary_2d(b_term, rho, dt, un, vn, dx, dy)
        p = apply_periodic_pressure_poisson_boundary_2d(b, p_term, pn_term, dx, dy)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) +
                        config.source * dt)

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
        
        
        # Periodic BC u @ x = 2     
        u[1:-1, -1] = (un[1:-1, -1] - un[1:-1, -1] * dt / dx * 
                    (un[1:-1, -1] - un[1:-1, -2]) -
                    vn[1:-1, -1] * dt / dy * 
                    (un[1:-1, -1] - un[0:-2, -1]) -
                    dt / (2 * rho * dx) *
                    (p[1:-1, 0] - p[1:-1, -2]) + 
                    nu * (dt / dx**2 * 
                    (un[1:-1, 0] - 2 * un[1:-1,-1] + un[1:-1, -2]) +
                    dt / dy**2 * 
                    (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + config.source * dt)

        # Periodic BC u @ x = 0
        u[1:-1, 0] = (un[1:-1, 0] - un[1:-1, 0] * dt / dx *
                    (un[1:-1, 0] - un[1:-1, -1]) -
                    vn[1:-1, 0] * dt / dy * 
                    (un[1:-1, 0] - un[0:-2, 0]) - 
                    dt / (2 * rho * dx) * 
                    (p[1:-1, 1] - p[1:-1, -1]) + 
                    nu * (dt / dx**2 * 
                    (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                    dt / dy**2 *
                    (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + config.source * dt)

        # Periodic BC v @ x = 2
        v[1:-1, -1] = (vn[1:-1, -1] - un[1:-1, -1] * dt / dx *
                    (vn[1:-1, -1] - vn[1:-1, -2]) - 
                    vn[1:-1, -1] * dt / dy *
                    (vn[1:-1, -1] - vn[0:-2, -1]) -
                    dt / (2 * rho * dy) * 
                    (p[2:, -1] - p[0:-2, -1]) +
                    nu * (dt / dx**2 *
                    (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
                    dt / dy**2 *
                    (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))

        # Periodic BC v @ x = 0
        v[1:-1, 0] = (vn[1:-1, 0] - un[1:-1, 0] * dt / dx *
                    (vn[1:-1, 0] - vn[1:-1, -1]) -
                    vn[1:-1, 0] * dt / dy *
                    (vn[1:-1, 0] - vn[0:-2, 0]) -
                    dt / (2 * rho * dy) * 
                    (p[2:, 0] - p[0:-2, 0]) +
                    nu * (dt / dx**2 * 
                    (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
                    dt / dy**2 * 
                    (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))   

        # Wall BC: u,v = 0 @ y = 0,2
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0

        denominator = np.sum(np.abs(un))

        if denominator == 0:
            u_l1norm = np.sum(np.abs(u) - np.abs(un)) 
        
        else:
            u_l1norm = (np.sum(np.abs(u) - np.abs(un))) / denominator
        
        u_history.append(u.copy())
        v_history.append(v.copy())
        p_history.append(p.copy())
    
    u_history_array = np.stack(u_history, axis=0)
    v_history_array = np.stack(v_history, axis=0)
    p_history_array = np.stack(p_history, axis=0)
    
    return u_history_array, v_history_array, p_history_array
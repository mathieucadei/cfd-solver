"""Numerical solver for the 2D diffusion equation."""



import numpy as np

from .operators import compute_convection_2d_term, compute_diffusion_2d_term, compute_source_term_2d, compute_pressure_poisson_term
from .boundary_conditions import apply_cavity_flow_boundary_2d

from ...setup.fvm.mesh import build_mesh, build_h_spacing, build_dist, build_face_positions, build_centers, build_face_areas, compute_cell_volumes




def solve_cavity_flow(
    initial_condition: np.ndarray,
    config: object,
) -> np.ndarray:
    """Solve the 2D cavity flow equation with an explicit central finite-difference scheme."""

    nu = config.viscosity
    rho = config.density

    dist_x, dist_y = build_dist(config)
    face_areas_x, face_areas_y = build_face_areas(config)
    cell_volumes = compute_cell_volumes(config)   
    xc, yc = build_centers(config)

    dt = config.time_step

    u, v, p, b = initial_condition

    un = np.empty_like(u)
    vn = np.empty_like(v)
    pn = np.empty_like(p)
    bn = np.empty_like(b)

    u_history = np.zeros((config.max_iterations + 1, config.num_cells_y, config.num_cells_x))
    v_history = np.zeros((config.max_iterations + 1, config.num_cells_y, config.num_cells_x))
    p_history = np.zeros((config.max_iterations + 1, config.num_cells_y, config.num_cells_x))

    u_history[0], v_history[0], p_history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()
        vn = v.copy()
        pn = p.copy()
        bn = b.copy()

        convection_u_term, convection_v_term = compute_convection_2d_term(
                                                    un, 
                                                    vn, 
                                                    face_areas_x, 
                                                    face_areas_y, 
                                                    cell_volumes, 
                                                    dt
                                                )
        
        diffusion_u_term = compute_diffusion_2d_term(
                            un,
                            dist_x,
                            dist_y,
                            face_areas_x, 
                            face_areas_y, 
                            cell_volumes,                             
                            dt, 
                            config.viscosity
                        )

        diffusion_v_term = compute_diffusion_2d_term(
                            vn,
                            dist_x,
                            dist_y,
                            face_areas_x, 
                            face_areas_y, 
                            cell_volumes,                             
                            dt, 
                            config.viscosity
                        )
        
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
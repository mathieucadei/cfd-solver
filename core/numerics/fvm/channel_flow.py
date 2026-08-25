"""Numerical solver for the 2D diffusion equation."""



import numpy as np

from .operators import compute_convection_2d_term, compute_diffusion_2d_term, compute_source_term_2d, compute_periodic_pressure_poisson_term
from .boundary_conditions import apply_periodic_source_term_boundary_2d, apply_pressure_poisson_term_boundary, apply_cavity_flow_boundary_2d

from ...setup.fvm.mesh import build_mesh, build_h_spacing, build_dist, build_face_positions, build_centers, build_face_areas, compute_cell_volumes



def solve_channel_flow_fvm(
    initial_condition: np.ndarray,
    config: object,
) -> np.ndarray:
    """Solve the 2D channel flow equation with an explicit central finite-difference scheme."""

    nu = config.viscosity
    rho = config.density

    dist_x, dist_y = build_dist(config)
    face_areas_x, face_areas_y = build_face_areas(config)
    cell_volumes = compute_cell_volumes(config)   
    xc, yc = build_centers(config)

    dt = config.time_step

    u_l1norm = 1

    u, v, p, b = initial_condition

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
        
        b = compute_source_term_2d(
                bn, 
                config.density, 
                config.time_step, 
                un, 
                vn,
                dist_x,
                dist_y,                           
                face_areas_x,
                face_areas_y, 
                cell_volumes, 
            )

        apply_periodic_source_term_boundary_2d(
                b,
                config.density, 
                config.time_step, 
                un, 
                vn,
                face_areas_x,
                face_areas_y, 
                cell_volumes, 
        )
        
        p = compute_periodic_pressure_poisson_term(
                pn, 
                b, 
                config.max_pseudo_iterations, 
                dist_x,
                dist_y,                           
                face_areas_x,
                face_areas_y,
                cell_volumes,
                lx=config.domain_length_x,
                ly=config.domain_length_y,
                xc=xc,
                yc=yc,
            )[0]

        # p = apply_periodic_pressure_poisson_boundary_2d(b, p_term, pn_term, dx, dy)

        f_w_p = face_areas_x[1:, 1:] * (p[1:, 1:] + p[1:, :-1]) / 2
        f_e_p = face_areas_x[1:, 2:] * (p[1:, 2:] + p[1:, 1:-1]) / 2
        f_s_p = face_areas_y[:-1, 1:] * (p[1:, 1:] + p[:-1, 1:]) / 2
        f_n_p = face_areas_y[2:, 1:] * (p[2:, 1:] + p[1:-1, 1:]) / 2

        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         convection_u_term[1:-1, 1:-1] -
                         dt / rho * (f_e_p[:-1, :] - f_w_p[:-1, :-1]) / cell_volumes[1:-1, 1:-1] + 
                         diffusion_u_term[1:-1, 1:-1] + 
                        config.source * dt)

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        convection_v_term[1:-1, 1:-1] -
                        dt / rho * (f_n_p[:, :-1] - f_s_p[:-1, :-1]) / cell_volumes[1:-1, 1:-1] +
                         diffusion_v_term[1:-1, 1:-1])
        
        apply_cavity_flow_boundary_2d(
            u, 
            v,
            un,
            vn,
            p, 
            config.source,
            config.time_step,
            config.density,
            config.viscosity,            
            dist_x=dist_x,
            dist_y=dist_y,
            face_areas_x=face_areas_x, 
            face_areas_y=face_areas_y,
            cell_volumes=cell_volumes, 
            lx=config.domain_length_x,
            ly=config.domain_length_y,
            xc=xc,
            yc=yc,
        )

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
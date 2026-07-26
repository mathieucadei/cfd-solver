"""Numerical solver for the 1D Burgers' equation."""



import numpy as np

from .operators import compute_convection_1d_term, compute_diffusion_1d_term
from .boundary_conditions import apply_burgers_boundary_1d

from ...config_fvm import BurgersEquation1DFVMConfig
from ...setup.fvm.mesh import build_mesh, build_hx_spacing, build_x_face_positions, build_x_centers, build_dist_x, make_cole_hopf_x_mesh
from ...setup.fvm.time_stepping import compute_diffusive_dt_1d_fvm, compute_cole_hopf_dt_1d_fvm


def solve_burgers_equation_1d_fvm(
    initial_condition: np.ndarray,
    config: BurgersEquation1DFVMConfig,
) -> np.ndarray:
    """Solve the 1D Burgers' equation with an explicit finite-difference scheme."""

    if config.grid_type == "hat":
        hx = build_hx_spacing(config)
        dt = compute_diffusive_dt_1d_fvm(config)
    
    elif config.grid_type == "cole_hopf":

        hx = make_cole_hopf_x_mesh(config)
        dt = compute_cole_hopf_dt_1d_fvm(config)
    
    else:
        raise ValueError("grid_type must be 'hat' or 'cole_hopf'")

    dist_x = build_dist_x(config)
    xc = build_x_centers(config)

    u = initial_condition.copy()

    history = np.zeros((config.max_iterations + 1, config.num_cells_x))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        convection_term = compute_convection_1d_term(un, hx, dt)
        diffusion_term = compute_diffusion_1d_term(un, hx, dist_x, dt, config.viscosity)

        u[1:-1] = un[1:-1] - convection_term[1:-1] \
            + diffusion_term[1:-1]
        
        apply_burgers_boundary_1d(
            u=u,
            un=un,
            dt=dt,
            hx=hx,
            dist_x=dist_x,
            xc=xc,
            lx=config.domain_length_x,
            nu=config.viscosity,
        )
        
        history[n] = u
    
    return history
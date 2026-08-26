"""Numerical solver for the 1D advection equation."""



import numpy as np
import matplotlib.pyplot as plt

from .operators import compute_advection_1d_term
from ...setup.fvm.time_stepping import compute_advective_dt_1d_fvm
from ...setup.fvm.mesh import build_mesh, build_hx_spacing, build_x_face_positions, build_x_centers
from ...setup.fvm.initial_conditions import hat_initial_condition_1d_fvm

from dataclasses import dataclass


def solve_advection_1d_fvm(
    initial_condition: np.ndarray,
    config: object,
) -> np.ndarray:
    """Solve the 1D advection equation with an explicit upwind finite-volume scheme."""

    dt = compute_advective_dt_1d_fvm(config)

    hx = build_hx_spacing(config)

    u = initial_condition.copy()

    history = np.zeros((config.max_iterations + 1, config.num_cells_x))

    history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        advection_term = compute_advection_1d_term(un, config.wavespeed, hx, dt)

        u[1:] = un[1:] - advection_term[1:]

        history[n] = u

    return history


if __name__ == '__main__':

    @dataclass
    class Advection1DFVMConfig:
        nx: int = 100
        ny: int = 1
        lx: float = 2.0
        rx: float = 1.
        sigma: float = 1.0
        wavespeed: float = 1.0
        max_iterations: int = 40
        hat_start: float = 0.5
        hat_end: float = 1.0
        u_min: float = 1.0
        u_max: float = 2.0
    
    hx = build_hx_spacing(Advection1DFVMConfig)

    initial_condition = hat_initial_condition_1d_fvm(hx, Advection1DFVMConfig)
    
    history = solve_advection_1d_fvm(
        initial_condition=initial_condition,
        config=Advection1DFVMConfig
    )

    u = history[-1]

    xf = build_x_face_positions(Advection1DFVMConfig)

    xc = build_x_centers(Advection1DFVMConfig)

    fig, ax = plt.subplots(3,1, figsize=(12,12), constrained_layout=True)
    ax[0].plot(xc, u)
    pc = ax[1].pcolormesh(xf, [0, 1], u[None, :], edgecolors='k', linewidth=0.3)
    ax[2].quiver(xc[::4], np.zeros_like(xc)[::4], u[::4], np.zeros_like(u)[::4])
    fig.colorbar(pc, label='u')

    plt.show()
    

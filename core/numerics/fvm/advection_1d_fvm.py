"""Numerical solver for the 1D advection equation."""



import numpy as np
import matplotlib.pyplot as plt

from ...setup.fvm.time_stepping import compute_advective_dt_1d
from ...setup.fvm.mesh import build_mesh
from ...setup.fvm.initial_conditions import hat_initial_condition_1d

from dataclasses import dataclass, field


def solve_advection_1d_fvm(
    initial_condition: np.ndarray,
    config: object,
) -> np.ndarray:
    """Solve the 1D advection equation with an explicit upwind finite-difference scheme."""

    dt = compute_advective_dt_1d(config)
    mesh = build_mesh(config)
    hx = mesh['hx']

    u = initial_condition.copy()

    # u = np.ones(config.nx)      #numpy function ones()

    # u[int(len(hx)*0.25):int(len(hx)*0.5)] = 2  #setting u = 2 between 0.5 and 1 as per our I.C.s

    # history = np.zeros((config.max_iterations + 1, config.nx))

    # history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        f_w = config.wavespeed * u[1:] / hx[1:]

        f_e = config.wavespeed * u[:-1] / hx[1:]

        u[1:] = un[1:] - dt * (f_w - f_e)

        # history[n] = u

    return u


if __name__ == '__main__':

    @dataclass
    class Advection1DFVMConfig:
        nx: int = 40
        ny: int = 1
        lx: float = 2.0
        ly: float = 1.0
        rx: float = 1.1
        ry: float = 1.
        sigma: float = 1.0
        wavespeed: float = 1.0
        max_iterations: int = 50
        hat_start: float = 0.5
        hat_end: float = 1.0
        u_min: float = 1.0
        u_max: float = 2.0
    
    mesh = build_mesh(Advection1DFVMConfig)

    initial_condition = hat_initial_condition_1d(mesh['hx'], Advection1DFVMConfig)
    
    u = solve_advection_1d_fvm(
        initial_condition=initial_condition,
        config=Advection1DFVMConfig
    )

    print(u)

    fig, ax = plt.subplots(3,1, figsize=(12,12), constrained_layout=True)
    ax[0].plot(mesh['xc'], u)
    pc = ax[1].pcolormesh(mesh['xf'], [0, 1], u[None, :], edgecolors='k', linewidth=0.3)
    ax[2].quiver(mesh['xc'][::4], np.zeros_like(mesh['xc'])[::4], u[::4], np.zeros_like(u)[::4])
    fig.colorbar(pc, label='u')

    plt.show()
    

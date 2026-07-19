"""Numerical solver for the 1D advection equation."""



import numpy as np
import matplotlib.pyplot as plt
from .operators import compute_advection_1d_term

from ..setup.mesh import build_mesh
from ..setup.initial_conditions import hat_initial_condition_1d

from dataclasses import dataclass, field


def solve_advection_1d_fvm(
    config: object,
    mesh: object,
) -> np.ndarray:
    """Solve the 1D advection equation with an explicit upwind finite-difference scheme."""

    dt = 0.025

    u = np.ones(config.nx)      #numpy function ones()

    u[int(len(mesh['xc'])*0.25):int(len(mesh['xc'])*0.5)] = 2  #setting u = 2 between 0.5 and 1 as per our I.C.s

    # history = np.zeros((config.max_iterations + 1, config.nx))

    # history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        f_w = config.c * u[1:]

        f_e = config.c * u[:-1]

        u[1:] = un[1:] - config.c * dt / mesh['hx'][1:] * (un[1:] - un[:-1])

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
        c: float = 1.0
        max_iterations: int = 25
    
    mesh = build_mesh(Advection1DFVMConfig)
    
    u = solve_advection_1d_fvm(
        config=Advection1DFVMConfig,
        mesh=mesh,
    )

    print(u)

    fig, ax = plt.subplots(3,1, figsize=(12,12), constrained_layout=True)
    ax[0].plot(mesh['xc'], u)
    pc = ax[1].pcolormesh(mesh['xf'], [0, 1], u[None, :], edgecolors='k', linewidth=0.3)
    ax[2].quiver(mesh['xc'][::4], np.zeros_like(mesh['xc'])[::4], u[::4], np.zeros_like(u)[::4])
    fig.colorbar(pc, label='u')

    plt.show()
    

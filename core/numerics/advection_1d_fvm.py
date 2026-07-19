"""Numerical solver for the 1D advection equation."""



import numpy as np
import matplotlib.pyplot as plt
from .operators import compute_advection_1d_term

from ..setup.mesh import build_mesh
from ..setup.initial_conditions import hat_initial_condition_1d

from dataclasses import dataclass, field


def solve_advection_1d_fvm(
    initial_condition: np.ndarray,
    config: object,
    mesh: object,
) -> np.ndarray:
    """Solve the 1D advection equation with an explicit upwind finite-difference scheme."""

    dt = 0.05

    u = initial_condition.copy()

    # history = np.zeros((config.max_iterations + 1, config.nx))

    # history[0] = initial_condition

    for n in range(1, config.max_iterations + 1):

        un = u.copy()

        f_w = config.c * u[1:]

        f_e = config.c * u[:-1]

        u[1:] = un[1:] - dt / mesh['hx'] * (f_w - f_e)

        # history[n] = u

    return u


if __name__ == '__main__':

    @dataclass
    class Mesh:
        nx: int = 40
        ny: int = 40
        lx: float = 2.0
        ly: float = 1.0
        rx: float = 1.1
        ry: float = 1.1
        c: float = 1.0
        sigma: float = 1
        wavespeed: float = 1.0
        u_min: float = 1.0
        u_max: float = 2.0
        hat_start: float = 0.5
        hat_end: float = 1.0
        max_iterations: int = 25
    
    mesh = build_mesh(Mesh)
    
    initial_condition = hat_initial_condition_1d(
        x_array = mesh['CX'],
        config=Mesh
    )
    
    history = solve_advection_1d_fvm(
        initial_condition=initial_condition,
        config=Mesh,
        mesh=mesh,
    )

    print(history)

    plt.plot(initial_condition)
    plt.plot(history[0])
    plt.plot(history[-1])
    plt.show()
    

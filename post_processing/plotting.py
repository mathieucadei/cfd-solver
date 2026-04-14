import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation


def plot_line(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str='x',
    y_label: str='y',
    equation: str=None,
    title: bool=False,
    step_stride: int=5, 
    save_fig: bool=False,
) -> None:
    
    _, ax = plt.subplots()
    
    for n in range(0, y_values.shape[0], step_stride):

        ax.plot(x_values, y_values[n], color=cm.plasma(n/(y_values.shape[0] - 1)), label=f'Time step: {n}')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, rotation=0)
    ax.legend()
    ax.set_title(f'{equation.title()} Solution Snapshots' if title else None)

    if save_fig:
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig(f'results/figures/{equation.replace(" ", "_")}_solution_snapshots.png')

    plt.show()


def plot_line_cuts(
    x_values: np.ndarray,
    y_matrix: np.ndarray,
    axis: int = 0,
    x_label: str='x',
    y_label: str='y',
    cut_label: str='Step',
    equation: str=None,
    title: bool=False,
    step_stride: int=5, 
    save_fig: bool=False,
) -> None:
    
    _, ax = plt.subplots()

    num_cuts = y_matrix.shape[axis]

    for n in range(0, num_cuts, step_stride):
        if axis == 0:
            y_cut = y_matrix[n, :]
        elif axis == 1:
            y_cut = y_matrix[:, n]
        else:
            raise ValueError("axis must be 0 or 1")

        ax.plot(x_values, y_cut, color=cm.plasma(n/(num_cuts - 1)), label=f'{cut_label}: {n}')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, rotation=0)
    ax.legend()
    ax.set_title(f'{equation.title()} Solution Snapshots' if title else None)

    if save_fig:
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig(f'results/figures/{equation.replace(" ", "_")}_solution_snapshots.png')

    plt.show()


def plot_snapshots(
    x_array: np.ndarray,
    time_array: np.ndarray, 
    history_num: np.ndarray,
    history_ana: np.ndarray=None, 
    equation: str="equation", 
    step_stride: int=5, 
    save_fig: bool=False
) -> None:
    """Plots snapshots of the solution at specified time steps."""

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])


    x_grid, t_grid = np.meshgrid(x_array, time_array)

    surf = ax1.plot_surface(
        x_grid,
        t_grid,
        history_num,
        cmap=cm.plasma,
    )

    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_zlabel("u")
    ax1.set_box_aspect((2.0, 2.0, 1.2))


    # fig.colorbar(surf, ax=ax1)


    ctr = ax2.contourf(
        x_grid,
        t_grid,
        history_num,
        cmap=cm.plasma,
    )

    ax2.set_xlabel("x")
    ax2.set_ylabel("t", rotation=0)
    fig.colorbar(ctr, ax=ax2, label="u", fraction=0.046, pad=0.04)

    for n in range(0, history_num.shape[0], step_stride):

        label_num = f'Numerical (Time step: {n})' if history_ana is not None else f'Time step: {n}'
        ax3.plot(x_array, history_num[n], color=cm.plasma(n/(history_num.shape[0] - 1)), label=label_num)

    if history_ana is not None:

        for n in range(0, history_ana.shape[0], step_stride):

            ax3.plot(x_array, history_ana[n], color=cm.plasma(n/(history_num.shape[0] - 1)), linestyle='--', label=f'Analytical (Time step: {n})')
    
    ax3.set_xlabel("x")
    ax3.set_ylabel("u", rotation=0)
    ax3.legend()

    for x in range(0, history_num.shape[1], step_stride):

        label_num = f'Numerical (x: {x})' if history_ana is not None else f'x: {x}'
        ax4.plot(time_array, history_num[:, x], color=cm.plasma(x/(history_num.shape[1] - 1)), label=label_num)

    if history_ana is not None:

        for x in range(0, history_ana.shape[1], step_stride):

            ax4.plot(time_array, history_ana[:, x], color=cm.plasma(x/(history_num.shape[1] - 1)), linestyle='--', label=f'Analytical (x: {x})')
    
    ax4.set_xlabel("t")
    ax4.set_ylabel("u", rotation=0)
    ax4.legend()

    fig.suptitle(f"{equation.title()} Solution Snapshots")

    if save_fig:
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig(f'results/figures/{equation.replace(" ", "_")}_solution_snapshots.png')

    plt.show()


def plot_animation(
    x_array: np.ndarray,
    history_num: np.ndarray,
    history_ana: np.ndarray=None,
    equation: str="equation", 
    save_fig: bool=False
) -> None:
    """Creates an animation of the solution evolving over time."""
    
    fig, ax = plt.subplots()
    num_line, = ax.plot(x_array, history_num[0], lw=2,  label='Numerical')

    if history_ana is not None:

        ana_line, = ax.plot(x_array, history_ana[0], '--', lw=2, label='Analytical')

    ax.set_xlabel("x")
    ax.set_ylabel("u", rotation=0)

    if history_ana is not None:
        ax.legend()
    
    ax.set_title(f"{equation.title()} Solution Animation")

    def update(frame):

        num_line.set_ydata(history_num[frame])

        if history_ana is not None:

            ana_line.set_ydata(history_ana[frame])

        ax.set_title(f"{equation.title()} Solution Animation (Time step: {frame})")

        return num_line, ana_line if history_ana is not None else num_line,

    if history_ana is not None:
        frames = history_ana.shape[0]
    else:
        frames = history_num.shape[0]

    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

    if save_fig:
        os.makedirs('results/animations', exist_ok=True)
        ani.save(f'results/animations/{equation.replace(" ", "_")}_solution_animation.mp4', writer="ffmpeg")

    plt.show()
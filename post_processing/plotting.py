import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_snapshots(
    x_array: np.ndarray, 
    history_num: np.ndarray,
    history_ana: np.ndarray=None, 
    equation: str="equation", 
    step_stride: int=5, 
    save_fig: bool=False
) -> None:
    """Plots snapshots of the solution at specified time steps."""
    
    _, ax = plt.subplots()

    for n in range(0, history_num.shape[0], step_stride):

        ax.plot(x_array, history_num[n], label=f'Time step: {n}')

    if history_ana is not None:

        for n in range(0, history_ana.shape[0], step_stride):

            ax.plot(x_array, history_ana[n], '--', label=f'Analytical (Time step: {n})')
    
    ax.set_xlabel("x")
    ax.set_ylabel("u", rotation=0)
    ax.legend()

    ax.set_title(f"{equation.title()} Solution Snapshots")

    if save_fig:
        os.makedirs('post_processing/figures', exist_ok=True)
        plt.savefig(f'post_processing/figures/{equation.replace(" ", "_")}_solution_snapshots.png')

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
    num_line, = ax.plot(x_array, history_num[0], lw=2)

    if history_ana is not None:

        ana_line, = ax.plot(x_array, history_ana[0], '--', lw=2, label='Analytical')

    ax.set_xlabel("x")
    ax.set_ylabel("u", rotation=0)
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
        os.makedirs('post_processing/animations', exist_ok=True)
        ani.save(f'post_processing/animations/{equation.replace(" ", "_")}_solution_animation.mp4', writer="ffmpeg")

    plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_snapshots(
    x: np.ndarray, 
    history: np.ndarray, 
    equation: str="equation", 
    step_stride: int=5, 
    save_fig: bool=False
) -> None:
    """Plots snapshots of the solution at specified time steps."""
    
    fig, ax = plt.subplots()

    for n in range(0, history.shape[0], step_stride):

        ax.plot(x, history[n], label=f'Time step: {n}')
    
    ax.set_xlabel("x")
    ax.set_ylabel("u", rotation=0)
    ax.legend()

    ax.set_title(f"{equation.title()} Solution Snapshots")

    if save_fig:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{equation.replace(" ", "_")}_solution_snapshots.png')

    plt.show()

def plot_animation(
    x: np.ndarray, 
    history: np.ndarray, 
    equation: str="equation", 
    save_fig: bool=False
) -> None:
    """Creates an animation of the solution evolving over time."""
    
    fig, ax = plt.subplots()
    line, = ax.plot(x, history[0], lw=2)

    ax.set_xlabel("x")
    ax.set_ylabel("u", rotation=0)
    ax.set_title(f"{equation.title()} Solution Animation")

    def update(frame):

        line.set_ydata(history[frame])

        ax.set_title(f"{equation.title()} Solution Animation (Time step: {frame})")

        return line,

    ani = FuncAnimation(fig, update, frames=history.shape[0], interval=100, blit=False)

    if save_fig:
        os.makedirs('animations', exist_ok=True)
        ani.save(f'animations/{equation.replace(" ", "_")}_solution_animation.mp4', writer="ffmpeg")

    plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_solution_traces(
    ax: Axes,
    x_values: np.ndarray,
    num_solution_matrix: np.ndarray,
    cut_values: np.ndarray,
    step_stride: int = 5,
    ana_solution_matrix: np.ndarray = None,
    x_label: str = 'x',
    y_label: str = 'u',
    cut_label: str = 't',
    equation_name: str = None,
    title: bool = False,
) -> None:

    n_cuts = cut_values.shape[0]

    for n in range(0, n_cuts, step_stride):

        if n_cuts == num_solution_matrix.shape[0]:
            y_cut = num_solution_matrix[n, :]
        else:
            y_cut = num_solution_matrix[:, n]
        
        num_label = f'Numerical ({cut_label}: {np.max(cut_values)/n_cuts*n:.3g})' if ana_solution_matrix is not None else f'{cut_label}: {np.max(cut_values)/n_cuts*n:.3g}'
        
        ax.plot(x_values, y_cut, color=cm.plasma(n/(n_cuts - 1)), label=num_label)
    
    if ana_solution_matrix is not None:
        for n in range(0, n_cuts, step_stride):

            if n_cuts == ana_solution_matrix.shape[0]:
                y_cut = ana_solution_matrix[n, :]
            else:
                y_cut = ana_solution_matrix[:, n]
        
            ana_label = f'Analytical ({cut_label}: {n})'
        
            ax.plot(x_values, y_cut, color=cm.plasma(n/(n_cuts - 1)), linestyle='--', label=ana_label)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, rotation=0)
    ax.legend()

    if title:
        ax.set_title(f'{equation_name} Solution')


def plot_solution_contour(
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    solution_matrix: np.ndarray,
    cmap: Colormap = cm.plasma,
    x_label: str = 'x',
    y_label: str = 't',
    z_label: str = 'u',
    equation_name: str = None,
    title: bool = False,
):
    
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    contour = ax.contourf(x_grid, y_grid, solution_matrix, cmap=cmap)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, rotation=0)

    if title:
        ax.set_title(f'{equation_name} Solution')
    
    return contour


def plot_solution_surface(
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    solution_matrix: np.ndarray,
    cmap: Colormap = cm.plasma,
    x_label: str = 'x',
    y_label: str = 't',
    z_label: str = 'u',
    equation_name: str = None,
    title: bool = False,
) -> None:
    
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    ax.plot_surface(x_grid, y_grid, solution_matrix, cmap=cmap)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    if title:
        ax.set_title(f'{equation_name} Solution')


def show_solution_traces(
    x_values: np.ndarray,
    num_solution_matrix: np.ndarray,
    cut_values: np.ndarray,
    ana_solution_matrix: np.ndarray = None,
    cut_label: str = 't',
    x_label: str = 'x',
    y_label: str = 'u',
    equation_name: str = None,
    title: bool = False,
    step_stride: int = 5,
    save: bool = False,       
) -> None:
    
    fig, ax = plt.subplots()

    plot_solution_traces(
        ax=ax,
        x_values=x_values,
        num_solution_matrix=num_solution_matrix,
        cut_values=cut_values,
        ana_solution_matrix=ana_solution_matrix,
        cut_label=cut_label,
        x_label=x_label,
        y_label=y_label,
        equation_name=equation_name,
        title=title,
        step_stride=step_stride,
    )

    if save:
        _save_fig(fig=fig, equation_name=equation_name, fig_type='traces')

    plt.show()


def show_solution_contour(
    x_values: np.ndarray,
    y_values: np.ndarray,
    solution_matrix: np.ndarray,
    cmap: Colormap = cm.plasma,
    x_label: str = 'x',
    y_label: str = 't',
    z_label: str = 'u',
    equation_name: str = None,
    title: bool = False,
    save: bool = False,     
) -> None:
    
    fig = plt.figure()
    ax = fig.add_subplot()

    contour = plot_solution_contour(
        ax=ax,
        x_values=x_values,
        y_values=y_values,
        solution_matrix=solution_matrix,
        cmap=cmap,
        x_label=x_label,
        y_label=y_label,
        z_label=z_label,
        equation_name=equation_name,
        title=title,   
    )

    fig.colorbar(contour, ax=ax)

    if save:
        _save_fig(fig=fig, equation_name=equation_name, fig_type='contour')

    plt.show()


def show_solution_surface(
    x_values: np.ndarray,
    y_values: np.ndarray,
    solution_matrix: np.ndarray,
    cmap: Colormap = cm.plasma,
    x_label: str = 'x',
    y_label: str = 't',
    z_label: str = 'u',
    equation_name: str = None,
    title: bool = False,
    save: bool = False,     
) -> None:
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    plot_solution_surface(
        ax=ax,
        x_values=x_values,
        y_values=y_values,
        solution_matrix=solution_matrix,
        cmap=cmap,
        x_label=x_label,
        y_label=y_label,
        z_label=z_label,
        equation_name=equation_name,
        title=title,   
    )

    if save:
        _save_fig(fig=fig, equation_name=equation_name, fig_type='surface')

    plt.show()


def show_solution_overview(
    x_values: np.ndarray,
    y_values: np.ndarray, 
    num_solution_matrix: np.ndarray,
    ana_solution_matrix: np.ndarray = None,
    cmap: Colormap = cm.plasma,
    x_label: str = 'x',
    y_label: str = 't',
    z_label: str = 'u',
    equation_name: str = None,
    step_stride: int=5,
    title: bool = False, 
    save: bool=False,
) -> None:
    """Plots snapshots of the solution at specified time steps."""

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])


    plot_solution_surface(
            ax=ax1,
            x_values=x_values,
            y_values=y_values,
            solution_matrix=num_solution_matrix,
            cmap=cmap,
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
            equation_name=equation_name,  
        )

    ax1.set_box_aspect((2.0, 2.0, 1.2))


    contour = plot_solution_contour(
        ax=ax2,
        x_values=x_values,
        y_values=y_values,
        solution_matrix=num_solution_matrix,
        cmap=cmap,
        x_label=x_label,
        y_label=y_label,
        z_label=z_label,
        equation_name=equation_name,  
    )

    fig.colorbar(contour, ax=ax2, label=z_label, fraction=0.046, pad=0.04)

    plot_solution_traces(
        ax=ax3,
        x_values=x_values,
        num_solution_matrix=num_solution_matrix,
        cut_values=y_values,
        ana_solution_matrix=ana_solution_matrix,
        cut_label=y_label,
        x_label=x_label,
        y_label=z_label,
        equation_name=equation_name,
        step_stride=step_stride,
    )

    plot_solution_traces(
        ax=ax4,
        x_values=y_values,
        num_solution_matrix=num_solution_matrix,
        cut_values=x_values,
        ana_solution_matrix=ana_solution_matrix,
        cut_label=x_label,
        x_label=y_label,
        y_label=z_label,
        equation_name=equation_name,
        step_stride=step_stride,
    )

    if title:
        fig.suptitle(f"{equation_name.title()} Solution Overview")

    if save:
        _save_fig(fig=fig, equation_name=equation_name, fig_type='overview')

    plt.show()


def show_solution_1d_animation(
    x_values: np.ndarray,
    num_solution_matrix: np.ndarray,
    ana_solution_matrix: np.ndarray = None,
    equation_name: str = 'equation',
    save: bool = False
) -> None:
    """Creates an animation of the solution evolving over time."""
    
    fig, ax = plt.subplots()
    num_line, = ax.plot(x_values, num_solution_matrix[0], lw=2,  label='Numerical')

    if ana_solution_matrix is not None:

        ana_line, = ax.plot(x_values, ana_solution_matrix[0], '--', lw=2, label='Analytical')

    ax.set_xlabel("x")
    ax.set_ylabel("u", rotation=0)

    if ana_solution_matrix is not None:
        ax.legend()
    
    ax.set_title(f'{equation_name.title()} Solution Animation')

    def update(frame):

        num_line.set_ydata(num_solution_matrix[frame])

        if ana_solution_matrix is not None:

            ana_line.set_ydata(ana_solution_matrix[frame])

        ax.set_title(f'{equation_name.title()} Solution Animation (Time step: {frame})')

        return num_line, ana_line if ana_solution_matrix is not None else num_line,

    if ana_solution_matrix is not None:
        frames = ana_solution_matrix.shape[0]
    else:
        frames = num_solution_matrix.shape[0]

    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

    if save:
        _save_ani(ani=ani, equation_name=equation_name, fig_type='1d')

    plt.show()


def _save_fig(fig: Figure, equation_name: str, fig_type: str = 'figure') -> None:
    equation_filename  = equation_name.lower().replace(" ", "_")
    directory = 'results/figures' if fig_type == 'figure' else f'results/figures/{fig_type}'

    os.makedirs(directory, exist_ok=True)
    fig.savefig(f'{directory}/{equation_filename }_solution_{fig_type}.png')


def _save_ani(ani: FuncAnimation, equation_name: str, fig_type: str = 'animations') -> None:
    equation_filename  = equation_name.lower().replace(' ', '_')
    directory = 'results/animations' if fig_type == 'animations' else f'results/animations/{fig_type}'

    os.makedirs(directory, exist_ok=True)
    ani.save(f'{directory}/{equation_filename }_solution_{fig_type}.mp4', writer='ffmpeg')
import numpy as np


def solve_heat_equation_1d(
    series_terms: np.ndarray, 
    mode_indices: np.ndarray,
    x_array: np.ndarray, 
    time_array: np.ndarray, 
    viscosity: float, 
    basis: str="periodic",
) -> np.ndarray:
    """Compute the analytical solution to the 1D heat equation from Fourier modes."""

    domain_length = np.round(np.max(x_array))
    
    t = time_array[:, None, None]
    n = mode_indices[None, None, :]

    if basis == "periodic":
        decay = np.exp(-viscosity * ((2 * np.pi / domain_length * n)**2 * t))
    
    elif basis == "cosine":
        decay = np.exp(-viscosity * ((np.pi / domain_length * n)**2 * t))
    
    else:
        raise ValueError("basis must be 'periodic' or 'cosine'")
    
    heat_equation_solution = np.sum(series_terms[None, :, :] * decay, axis=2).real

    return heat_equation_solution

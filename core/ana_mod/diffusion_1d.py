import numpy as np


def solve_diffusion_1d_ana(
    series_terms: np.ndarray, 
    mode_indices: np.ndarray,
    x_array: np.ndarray, 
    time_array: np.ndarray, 
    viscosity: float, 
    basis: str="periodic",
) -> np.ndarray:
    """Compute the analytical solution to the 1D diffusion equation from Fourier modes."""

    domain_length = np.round(np.max(x_array))
    
    t = time_array[:, None, None]
    n = mode_indices[None, None, :]

    if basis == "periodic":
        decay = np.exp(-viscosity * ((2 * np.pi / domain_length * n)**2 * t))
    
    elif basis == "cosine":
        decay = np.exp(-viscosity * ((np.pi / domain_length * n)**2 * t))
    
    else:
        raise ValueError("basis must be 'periodic' or 'cosine'")

    return np.sum(series_terms[None, :, :] * decay, axis=2).real

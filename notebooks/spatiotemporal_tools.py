"""
Set of utilities for spatiotemporal data processing and modeling, including distance-based weighting functions and spatial cross-validation tools
"""

import torch

def checkerboard_splits(coords, grid_size, dim=-1):
    """
    Creates checkerboard splits for spatial cross-validation based on the provided coordinates and grid size.
    This supports N-Dimensional coordinates, where dim specifies the dimension along which to compute the checkerboard pattern.
    
    Args:
        coords (torch.Tensor): Tensor of shape (..., N) containing (e.g. latitude, longitude, time) pairs.
        grid_size (float or N-Dimensional tensor): Size of the grid cells (e.g. in degrees or days).
        dim (int): Dimension along which to compute the checkerboard pattern (default: -1).
        
    Returns:
        torch.Tensor: Tensor of shape (..., 1) containing split assignments (0 or 1) for each point based on the checkerboard pattern.
    """

    splits = ((coords // grid_size).sum(dim=dim) % 2).unsqueeze(dim).to(torch.int8)

    return splits
    


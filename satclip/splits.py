"""
Set of utilities for spatiotemporal data processing and modeling, including distance-based weighting functions and spatial cross-validation tools
"""

import torch

def checkerboard_splits(coords, grid_size, offset=0, dim=-1):
    """
    Creates checkerboard splits for spatial cross-validation based on the provided coordinates and grid size.
    This supports N-Dimensional coordinates, where dim specifies the dimension along which to compute the checkerboard pattern.
    
    Args:
        coords (torch.Tensor): Tensor of shape (..., N) containing (e.g. latitude, longitude, time) pairs.
        grid_size (float or N-Dimensional tensor): Size of the grid cells (e.g. in degrees or days).
        offset (float or N-Dimensional tensor): Offset to apply to the coordinates before computing the checkerboard pattern (default: 0).
        dim (int): Dimension along which to compute the checkerboard pattern (default: -1).
        
    Returns:
        torch.Tensor: Tensor of shape (..., 1) containing split assignments (0 or 1) for each point based on the checkerboard pattern.
    """

    splits = (( (coords + offset) // grid_size).sum(dim=dim) % 2).unsqueeze(dim).to(torch.int8)

    return splits

def temporal_splits(coords, temporal_grid_size):
    """
    Creates temporal splits for spatial cross-validation based on the provided coordinates and temporal grid size.

    Args:
        coords (torch.Tensor): Tensor of shape (..., 1) containing time coordinates.
        temporal_grid_size (float): Size of the temporal grid cells (e.g. in days).

    Returns:
        torch.Tensor: Tensor of shape (..., 1) containing split assignments (0 or 1) for each point based on the temporal pattern.
    """

    if temporal_grid_size <= 0:
        return torch.ones_like(coords, dtype=torch.int8)

    splits = ((coords // temporal_grid_size) % 2).to(torch.int8)
 
    return splits

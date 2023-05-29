"""Utility to identify centerlines of polygons."""
import numpy as np
from scipy.ndimage import label, distance_transform_edt, percentile_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def identify_centerlines(masked_array, filter_size = 9, percentile = 70, outlet = 'south'):
    """Given a masked array, identify the centerlines of outlined shapes."""

    # Identify pixels that are at or near the centerline
    distance_array = distance_transform_edt(masked_array.mask)
    filter = np.ones((filter_size, filter_size))
    center = distance_array > percentile_filter(distance_array, percentile, footprint=filter)

    # Identify the largest cohesive grouping of these pixels
    labeled, n_labels = label(center)
    sizes = np.bincount(labeled.ravel())[1:]
    group = np.where(
        labeled == np.argmax(sizes) + 1,
        1,
        0
    )

    # Identify the pixel values of the outlet
    if isinstance(outlet, str):
        if outlet == 'east':
            idx = np.argmax(group * np.indices(group.shape)[1])
        elif outlet == 'north':
            idx = np.argmin(group * np.indices(group.shape)[0])
        elif outlet == 'west':
            idx = np.argmin(group * np.indices(group.shape)[1])
        elif outlet == 'south':
            idx = np.argmax(group * np.indices(group.shape)[0])
        else:
            raise ValueError('Outlet argument should be east, north, west, or south.')
        x0 = np.ravel(np.indices(group.shape)[0])[idx]
        y0 = np.ravel(np.indices(group.shape)[1])[idx]

    else:
        try:
            x0, y0 = outlet
        except:
            raise ValueError('Outlet must either be a string or iterable of length 2.')

    # Build the adjacency matrix
    region = np.argwhere(group == 1)
    num_points = len(region)
    adjacency_matrix = np.zeros((num_points, num_points), dtype = float)

    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(region[i] - region[j])

            adjacency_matrix[i, j] = distance
            adjacency_matrix[j, i] = distance

    # Calculate distances through the connected region
    sparse = csr_matrix(adjacency_matrix)
    distance, _ = dijkstra(sparse, indices = [y0, x0])
    indices = np.nonzero(group)
    group_distance = np.zeros_like(group)
    group_distance[indices] = np.max(distance) - distance

    return group, group_distance
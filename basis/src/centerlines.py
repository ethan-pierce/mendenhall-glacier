"""Utility to identify centerlines of polygons."""
import numpy as np
from scipy.ndimage import label, generate_binary_structure

def identify_centerline(masked_array: np.ndarray) -> np.ndarray:
    """Identify the centerline of shapes in a binary array."""
    centerline = np.zeros_like(masked_array)

    labeled_array, n_labels = label(masked_array)
    structure = generate_binary_structure(2, 1)

    for lab in range(1, n_labels + 1):
        label_mask = labeled_array == lab
        dilated_mask = np.logical_and(label_mask, np.logical_not(centerline))

        eroded_mask = np.logical_and(label_mask, centerline)
        eroded_mask = np.logical_and(eroded_mask, structure)

        centerline[dilated_mask] = True
        centerline[eroded_mask] = False

    return centerline

import numpy as np
from shapely.geometry import Point, Polygon


def sort_coordinates(coords):
    """
    Sorts coordinates in counterclockwise sense.

    Args:
        coords (numpy.ndarray): Coordinates to be sorted.

    Returns:
        numpy.ndarray: Sorted coordinates.
    """
    centroid = np.mean(coords, axis=0)
    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_coords = coords[sorted_indices]
    return sorted_coords


def mask_from_polygon(meshgrid, polygon_coordinates):
    """
    Create a mask matrix indicating whether points in a meshgrid are inside a polygon.
    Args:
    - meshgrid: Tuple of arrays from np.meshgrid.
    -polygon_coordinates: Polygon's coordinates; sorted counterclockwise.

    Returns:
    - numpy.ndarray: Mask from meshgrid
    """
    X, Y = meshgrid
    mask = np.zeros_like(X, dtype=bool)
    polygon = Polygon(polygon_coordinates)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = Point(X[i, j], Y[i, j])
            mask[i, j] = polygon.contains(point)

    return mask

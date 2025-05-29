from time import time

import numpy as np
from matplotlib.path import Path


# Mock data for testing
polygon_coords = np.array(
    [
        [25.7617, -80.1918],
        [25.7618, -80.1917],
        [25.7616, -80.1919],
        [25.7617, -80.1919],
        [25.7618, -80.1918],
    ]
)
direction = np.array([0.001, 0.001])
x_inv = np.array([25.7617, -80.1918])
lr = 1.0

# Precompute rotation angles
n = 3
_base_angles = np.linspace(0, np.pi, n, endpoint=False)
_rotation_angles = np.empty(n * 2)
_rotation_angles[0::2] = _base_angles
_rotation_angles[1::2] = -_base_angles


# Original implementation
def original_rotate_within_domain(direction, x_inv, lr, polygon_coords):
    next_norm = x_inv + lr * direction
    next_point = next_norm  # Mock normalization
    polygon_path = Path(polygon_coords)
    if polygon_path.contains_point(next_point):
        return next_point
    for angle in _rotation_angles:
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        rotated_dir = rot @ direction
        print(rotated_dir)  # Debugging line to check shape
        candidate_norm = x_inv + lr * rotated_dir
        candidate = candidate_norm  # Mock normalization
        if polygon_path.contains_point(candidate):
            return candidate
    return x_inv  # Fallback to the original point if no valid rotation found


# Optimized implementation using vectorized operations
# Compute all rotated directions at once
rotations = np.array(
    [
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        for angle in _rotation_angles
    ]
)
polygon_path = Path(polygon_coords)


def optimized_rotate_within_domain(direction, x_inv, lr, polygon_coords):
    next_norm = x_inv + lr * direction
    next_point = next_norm  # Mock normalization

    if polygon_path.contains_point(next_point):
        return next_point

    # print(rotations.shape)  # Debugging line to check shape
    rotated_dirs = rotations @ direction
    print(rotated_dirs)  # Debugging line to check shape
    # Check all candidates
    candidates_norm = x_inv + lr * rotated_dirs
    for candidate in candidates_norm:
        if polygon_path.contains_point(candidate):
            return candidate

    return x_inv  # Fallback to the original point if no valid rotation found


# Benchmarking
iterations = 1

# Original implementation
start_time = time()
for _ in range(iterations):
    original_rotate_within_domain(direction, x_inv, lr, polygon_coords)
original_duration = time() - start_time
print("-" * 50)
# Optimized implementation
start_time = time()
for _ in range(iterations):
    optimized_rotate_within_domain(direction, x_inv, lr, polygon_coords)
optimized_duration = time() - start_time

# Results
print(f"Original implementation duration: {original_duration:.4f} seconds")
print(f"Optimized implementation duration: {optimized_duration:.4f} seconds")

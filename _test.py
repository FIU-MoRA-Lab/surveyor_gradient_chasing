import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

class DummyNormalizer:
    def __init__(self, center):
        self.center = np.array(center)

    def inverse(self, point):
        # Simulates converting a normalized point back to world coordinates
        return point + self.center

class DomainHandler:
    def __init__(self, polygon_coords):
        self.polygon_coords = polygon_coords
        self._normalizer = DummyNormalizer(center=[5.0, 5.0])

    def rotate_within_domain(self, direction, x_inv, lr):
        """
        Rotates the direction vector to ensure it stays within the domain polygon.
        """
        next_point_inv = x_inv + lr * direction
        next_point = self._normalizer.inverse(next_point_inv)
        polygon_path = Path(np.array(self.polygon_coords)[:, [1, 0]])  # (lon, lat)

        if polygon_path.contains_point(next_point[::-1]):
            return next_point

        rotation_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)  # 8 directions
        for angle in rotation_angles:
            print('Trying rotation angle (rad):', angle)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
            tentative_direction = rotation_matrix @ direction
            next_point_inv = x_inv + lr * tentative_direction
            next_point = self._normalizer.inverse(next_point_inv)
            if polygon_path.contains_point(next_point[::-1]):
                print('Point is inside the polygon after rotation.')
                return next_point

        print('No valid direction found; returning last attempt')
        return next_point


if __name__ == "__main__":
    polygon = [
        [2, 2],
        [8, 2],
        [8, 8],
        [2, 8],
        [2, 2]
    ]

    handler = DomainHandler(polygon_coords=polygon)

    x_inv = np.array([0.0, 0.0])              # Relative to the center
    direction = np.array([1.0, 0.0])          # Initially toward the right
    lr = 6.0                                  # Step size

    new_point = handler.rotate_within_domain(direction, x_inv, lr)

    # Visualization
    poly = np.array(polygon)
    fig, ax = plt.subplots()
    ax.plot(poly[:, 1], poly[:, 0], 'k-', label="Polygon")

    origin = np.array([5.0, 5.0])
    ax.plot(origin[1], origin[0], 'ro', label='Origin')
    ax.plot(new_point[1], new_point[0], 'go', label='Final Point in Polygon')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Test: rotate_within_domain")
    plt.grid(True)
    plt.show()

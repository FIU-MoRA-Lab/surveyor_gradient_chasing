import numpy as np
from matplotlib.path import Path
from numpy.linalg import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from .normalizer import Normalizer
from .wp_plotter import WaterPhenomenonPlotter


class WaterPhenomenonGP:
    """
    Gaussian Process model for water-related phenomena over a geographic domain.

    Fits a GP to spatial data (latitude, longitude) and provides
    utilities for gradient-based exploration and visualization within a polygonal domain.
    """

    # Precompute rotation angles for _rotate_within_domain
    _base_angles = np.linspace(0, np.pi, 18, endpoint=False)
    _rotation_angles = np.empty(36)
    _rotation_angles[0::2] = _base_angles
    _rotation_angles[1::2] = -_base_angles
    _rotations = np.array(
        [
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            for angle in _rotation_angles
        ]
    )

    def __init__(self, domain=np.array([[25.7581572, -80.3734494]]), kernel=RBF()):
        """
        Args:
            domain (np.ndarray): Array of coordinates defining the domain polygon (lat, lon).
            kernel (sklearn.gaussian_process.kernels.Kernel): Kernel for the GP.
        """
        self.domain = np.atleast_2d(domain)
        self.polygon_coords = np.vstack([self.domain, self.domain[0]])
        self.origin = np.min(self.domain, axis=0)
        self._polygon_path = Path(self.polygon_coords)
        self._normalizer = Normalizer(self.origin)
        self._gaussian_process = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5, alpha=1e-5
        )
        self.function_to_plot = lambda x: self._gaussian_process.predict(
            self._normalizer.forward(x)
        )
        self.plotter = WaterPhenomenonPlotter(
            self.origin, self.domain, self.polygon_coords, self.function_to_plot
        )

    def fit(self, X, y):
        """
        Fit the GP model to training data.

        Args:
            X (np.ndarray): Training coordinates (lat, lon).
            y (np.ndarray): Target values.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        X_norm = self._normalizer.forward(X)
        self._gaussian_process.fit(X_norm, y)

    def _tap_gradient(self, x, h=0.001):
        """
        Estimate the gradient of the GP prediction at x using central finite differences.

        Args:
            x (np.ndarray): Input point (lat, lon).
            h (float): Step size for finite differences.

        Returns:
            np.ndarray: Gradient vector at x.
        """
        x = x.reshape(1, -1)
        dim = x.shape[-1]
        perturb = h * np.eye(dim)
        pred_plus = self._gaussian_process.predict(x + perturb)
        pred_minus = self._gaussian_process.predict(x - perturb)
        grad = (pred_plus - pred_minus) / (2 * h)
        return grad.flatten()

    def next_point(self, x, lr=1.0):
        """
        Suggest the next point by moving in the direction of the GP gradient.

        Args:
            x (np.ndarray): Current point (lat, lon).
            lr (float): Step size (learning rate).

        Returns:
            np.ndarray: Next suggested point (lat, lon).
        """
        x_inv = self._normalizer.forward(x)
        grad = self._tap_gradient(x_inv)
        direction = grad / norm(grad) if norm(grad) > 1e-4 else np.zeros_like(grad)
        return self._rotate_within_domain(direction, x_inv, lr)

    def _rotate_within_domain(self, direction, x_inv, lr):
        """
        Rotate the direction vector to keep the next point inside the domain polygon.

        Args:
            direction (np.ndarray): Direction vector.
            x_inv (np.ndarray): Normalized current point.
            lr (float): Step size.

        Returns:
            np.ndarray: Valid next point inside the polygon.
        """
        next_norm = x_inv + lr * direction
        next_point = self._normalizer.inverse(next_norm)
        if self._polygon_path.contains_point(next_point):
            return next_point

        rotated_dirs = self._rotations @ direction
        candidates_norm = x_inv + lr * rotated_dirs
        candidates = self._normalizer.inverse(candidates_norm)
        for candidate in candidates:
            if self._polygon_path.contains_point(candidate):
                return candidate

        return self._normalizer.inverse(
            x_inv
        )  # Fallback to the original point if no valid rotation found

    def update_plot(self, path, next_point, heading=0.0):
        """
        Update the plot with the current path and next point.

        Args:
            path (np.ndarray): Array of visited coordinates (lat, lon).
            next_point (np.ndarray): Next suggested point (lat, lon).
        """
        self.plotter.plot_env_and_path(path, next_point, heading)


if __name__ == "__main__":
    # Example usage
    kernel = RBF()
    X = np.array(
        [
            [25.7617, -80.1918],
            [25.7618, -80.1917],
            [25.7616, -80.1919],
            [25.7617, -80.1919],
            [25.7618, -80.1918],
        ],
        dtype=np.float64,
    )
    y = np.random.uniform(25, 30, size=5)
    domain = np.array(
        [
            [25.7615, -80.1920],
            [25.7620, -80.1920],
            [25.7620, -80.1915],
            [25.7615, -80.1915],
        ]
    )
    water_feature = WaterPhenomenonGP(domain, kernel)
    water_feature.fit(X, y)

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from numpy.linalg import norm
from normalizer import Normalizer
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import pickle

class WaterPhenomenonGP:
    """
    WaterPhenomenonGP models water-related phenomena using Gaussian Processes.

    This class uses a Gaussian Process (GP) to model and make predictions about water-related phenomena
    based on latitude and longitude coordinates. It normalizes the input data, fits the GP model, and 
    computes gradients to suggest the next point for analysis.

    Attributes:
        _normalizer (Normalizer): An instance of the Normalizer class to handle data normalization.
        _gaussian_process (GaussianProcessRegressor): The Gaussian Process model for regression.
    """
    
    def __init__(self, domain =  np.array([25.7581572, -80.3734494]), kernel = RBF()):
        """
        Initializes the WaterPhenomenonGP with normalization parameters and a kernel for the GP.

        Args:
            mins (tuple, numpy.array): Minimum coordinates of the domain of interest. (latitude, longitude)
            maxs (tuple, numpy.array): Maximum coordinates of the domain of interest.
            kernel (sklearn.gaussian_process.kernels): The kernel to be used by the Gaussian Process.
        """
        # Extract origin for normalization
        self.domain = np.atleast_2d(domain)
        self.origin = np.min(domain, axis = 0)
        
        # Initialize the normalizer
        self._normalizer = Normalizer(self.origin)
        
        # Initialize the Gaussian Process regressor
        self._gaussian_process = GaussianProcessRegressor(kernel=kernel,
                                                          n_restarts_optimizer=5,
                                                          alpha=1e-5)
        self._plot_environment = lambda x : self._gaussian_process.predict(self._normalizer.forward(x))
        # Initialize static variables for the plotting function
        self._fig = None
        self._ax = None
        self._c = None
        self._path_plt = None
        self._current_point_plt = None
    
    def _tap_gradient(self, x, h=0.001):
        """
        Computes the gradient of the GP predictions at point x using finite differences.

        Args:
            x (numpy.ndarray): The input point where the gradient is to be computed.
            h (float): The perturbation step size for finite differences (default is 0.001).

        Returns:
            numpy.ndarray: The gradient vector at point x.
        """
        x = x.reshape(1, -1)
        dim = x.shape[-1]
        perturbations = h * np.eye(dim)
        
        # Compute predictions for perturbed points
        predictions_plus = self._gaussian_process.predict(x + perturbations)
        predictions_minus = self._gaussian_process.predict(x - perturbations)
        
        # Calculate gradient using central differences
        grad = (predictions_plus - predictions_minus) / (2 * h)
        
        return grad.flatten()

    def fit(self, X, y):
        """
        Fits the Gaussian Process model to the training data.

        Args:
            X (numpy.ndarray): The input training data (coordinates).
            y (numpy.ndarray): The target values corresponding to X.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Normalize the input data
        X_transform = self._normalizer.forward(X)
        
        # Fit the GP model
        self._gaussian_process.fit(X_transform, y)

    def next_point(self, x, lr=1.0):
        """
        Suggests the next point for analysis based on the gradient of the GP predictions.

        Args:
            x (numpy.ndarray): The current point (coordinates).
            lr (float): The learning rate determining the step size in the direction of the gradient (default is 1.0).

        Returns:
            numpy.ndarray: The suggested next point (coordinates).
        """
        x_inv = self._normalizer.forward(x)
        gradient = self._tap_gradient(x_inv)
        
        # Determine the direction based on the gradient
        if norm(gradient) > 1e-4:
            direction = gradient / norm(gradient)
        else:
            print('No movement')
            direction = np.array([0., 0.])
        print(gradient, direction, x_inv)
        # Compute the next point in the normalized space and transform it back
        next_point_inv = x_inv + lr * direction
        return self._normalizer.inverse(next_point_inv)

    def plot_initialization(self, delta=0.00015,
                        plot_args={'marker': 'x', 'color': 'black'},
                        contourf_args={'cmap': 'jet', 'alpha': 0.5, 'levels' : 15}):
        min_lat, min_lon = self.origin
        max_lat, max_lon = np.max(self.domain, axis= 0)
        self._lat_grid, self._lon_grid = np.meshgrid(np.linspace(min_lat, max_lat, 100), np.linspace(min_lon, max_lon, 100))
        self._mesh_points = np.column_stack((self._lat_grid.ravel(), self._lon_grid.ravel()))

        # Initialize the map with satellite imagery if not already initialized
        if self._fig is None or self._ax is None:
            tiler = cimgt.GoogleTiles(style='satellite')
            transform = ccrs.PlateCarree()
            self._fig, self._ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': transform})
            self._ax.add_image(tiler, 20)
            self._ax.set_aspect('equal', adjustable='box')
            gl = self._ax.gridlines(draw_labels=True)
            gl.top_labels = False
            gl.right_labels = False

        # Set the extent of the map with a margin
        self._ax.set_extent([min_lon - delta, max_lon + delta, min_lat - delta, max_lat + delta], crs=ccrs.PlateCarree())
        self._path_plt,  = self._ax.plot([],[], transform=ccrs.PlateCarree(), label = 'current path', **plot_args)
        self._current_point_plt = self._ax.scatter([], [], transform=ccrs.PlateCarree(), label = 'Next point', color = 'red')


    def plot_env_and_path(self, path, next_point):
        """
        Plots the interpolated environment data and the path on a satellite image.

        Parameters:
        - environment: Callable
            Function to evaluate the environment at given coordinates (latitude, longitude).
        - path: np.ndarray
            Array of coordinates representing the path (shape: [n_points, 2]).
        - extent: tuple
            Tuple specifying the extent of the plot (min_lat, min_lon, max_lat, max_lon).
        - delta: float, optional
            Margin to add around the extent for better visualization (default is 0.00015).
        - plot_args: dict, optional
            Additional arguments for the path plot (default is empty dict).
        - contourf_args: dict, optional
            Additional arguments for the contour plot (default is empty dict).

        Returns:
        - None
        """

        # Evaluate the environment function on the meshgrid points
        interpolated_values = self._plot_environment(self._mesh_points).reshape(self._lat_grid.shape)

        # Plot the interpolated environment values as contours
        try:
            for c in self._c.collections:
                c.remove()
        except:
            pass
        self._c = self._ax.contourf(self._lon_grid, self._lat_grid, interpolated_values, transform=ccrs.PlateCarree())
        # Update the path plot data
        self._path_plt.set_xdata(path[:-1, 1])  # Update x data
        self._path_plt.set_ydata(path[:-1, 0])  # Update y data
        
        # Update the current point plot data
        self._current_point_plt.set_offsets((next_point[1], next_point[0]))  # Update position of the scatter point

        self._ax.legend()
        plt.draw()
        plt.pause(0.1)


if __name__ == '__main__':
    # Example
    kernel = RBF()
    X = np.array([
    [25.7617, -80.1918],  # Miami Downtown
    [25.7618, -80.1917],  # 10 meters north-east
    [25.7616, -80.1919],  # 10 meters south-west
    [25.7617, -80.1919],  # 10 meters west
    [25.7618, -80.1918]   # 10 meters north
    ], dtype = np.float64)

    # Example temperature values (randomly generated)
    y = np.random.uniform(25, 30, size=5)

    water_feature = WaterPhenomenonGP(X, kernel)

    water_feature.fit(X,y)

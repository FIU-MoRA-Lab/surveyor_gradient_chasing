import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


class WaterPhenomenonPlotter:
    """
    Plots water phenomenon data and exploration paths on satellite imagery.
    """

    def __init__(
        self,
        origin: np.ndarray,
        domain: np.ndarray,
        polygon_coords: np.ndarray,
        function_to_call: callable = lambda x: x,
    ):
        self.origin = origin
        self.domain = domain
        self.polygon_coords = polygon_coords
        self.function_to_call = function_to_call

        self._figure = None
        self._axis = None
        self._contour = None
        self._path_plot = None
        self._current_point_plot = None
        self._poly_patch = None
        self._lat_grid = None
        self._lon_grid = None
        self._mesh_points = None

    def plot_initialization(
        self,
        delta: float = 0.00015,
        plot_args: dict = None,
        contourf_args: dict = None,
        zoom: int = 20,
    ):
        """
        Initialize the plot with satellite imagery and domain polygon.
        """
        plot_args = plot_args or {"marker": "x", "color": "black"}
        min_lat, min_lon = self.origin
        max_lat, max_lon = np.max(self.domain, axis=0)
        self._lat_grid, self._lon_grid = np.meshgrid(
            np.linspace(min_lat, max_lat, 100), np.linspace(min_lon, max_lon, 100)
        )
        self._mesh_points = np.column_stack(
            (self._lat_grid.ravel(), self._lon_grid.ravel())
        )

        if self._figure is None or self._axis is None:
            tiler = cimgt.GoogleTiles(style="satellite")
            transform = ccrs.PlateCarree()
            self._figure, self._axis = plt.subplots(
                figsize=(8, 10), subplot_kw={"projection": transform}
            )
            self._axis.add_image(tiler, zoom)
            self._axis.set_aspect("equal", adjustable="box")
            gl = self._axis.gridlines(draw_labels=True)
            gl.top_labels = gl.right_labels = False

        self._axis.set_extent(
            [min_lon - delta, max_lon + delta, min_lat - delta, max_lat + delta],
            crs=ccrs.PlateCarree(),
        )
        (self._path_plot,) = self._axis.plot(
            [], [], transform=ccrs.PlateCarree(), label="Path", **plot_args
        )
        self._current_point_plot = self._axis.scatter(
            [0],
            [0],
            transform=ccrs.PlateCarree(),
            label="Next point",
            color="red",
            zorder=10,
        )
        self._poly_patch = Polygon(
            self.polygon_coords[:, [1, 0]],
            closed=True,
            edgecolor="orange",
            facecolor="none",
            linewidth=2,
            zorder=5,
            label="Domain",
        )
        self._axis.add_patch(self._poly_patch)

    def plot_env_and_path(self, path: np.ndarray, next_point: np.ndarray):
        """
        Plot the interpolated environment and the exploration path.
        """
        interpolated = self.function_to_call(self._mesh_points).reshape(
            self._lat_grid.shape
        )
        if self._contour is not None:
            try:
                self._contour.remove()
            except AttributeError:
                pass
        self._contour = self._axis.contourf(
            self._lon_grid, self._lat_grid, interpolated, transform=ccrs.PlateCarree()
        )
        self._contour.set_clip_path(self._poly_patch)
        self._path_plot.set_xdata(path[:-1, 1])
        self._path_plot.set_ydata(path[:-1, 0])
        self._current_point_plot.set_offsets([[next_point[1], next_point[0]]])
        self._axis.legend()
        plt.draw()
        plt.pause(0.1)

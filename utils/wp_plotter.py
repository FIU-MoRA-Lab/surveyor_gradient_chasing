import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pygeomag 

from .normalizer import LATITUDE_TO_METERS


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
        asvid: int = 0,
        water_feature: str = "Water Phenomenon",
    ):
        self.origin = origin
        self.domain = domain
        self.polygon_coords = polygon_coords
        self.function_to_call = function_to_call
        self.water_feature_name = water_feature

        self._asvid = asvid
        self._figure = None
        self._axis = None
        self._contour = None
        self._path_plot = None
        self._next_point_plot = None
        self._poly_patch = None
        self._lat_grid = None
        self._lon_grid = None
        self._mesh_points = None
        self._gm = pygeomag.GeoMag()

    def get_plot_arrow(self, x=0, y=0, angle=0, length=1.5, arrow_kwargs=None):
        """
        Plots a FancyArrow centered at (x, y) with a given angle.

        Parameters:
            ax (matplotlib.axes.Axes): The axes to plot on.
            x (float): X-coordinate of the arrow center.
            y (float): Y-coordinate of the arrow center.
            angle (float): Angle in degrees (0 is pointing right).
            length (float): Length of the arrow.
            arrow_kwargs (dict, optional): Additional keyword arguments for FancyArrow.
        """
        if arrow_kwargs is None:
            arrow_kwargs = {}

        angle_rad = np.radians(angle)
        length /= LATITUDE_TO_METERS
        dx = length * np.cos(angle_rad)
        dy = length * np.sin(angle_rad)

        # Start the arrow so that its center is at (x, y)
        x_start = x - dx / 2
        y_start = y - dy / 2
        arrow = FancyArrow(
            x_start,
            y_start,
            dx,
            dy,
            width=0.75 / LATITUDE_TO_METERS,
            label=f"ASV {self._asvid}",
            facecolor="white",
            edgecolor="black",
            **arrow_kwargs,
        )
        return arrow

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
        plot_args = plot_args or {"color": "black", "linestyle": "--"}
        self._contourf_args = contourf_args or {
            "cmap": "coolwarm",
            "levels": 15,
        }
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
                figsize=(6, 6), subplot_kw={"projection": transform}
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
        self._next_point_plot = self._axis.scatter(
            [0],
            [0],
            transform=ccrs.PlateCarree(),
            label="Target point",
            color="red",
            zorder=10,
        )
        self._current_point = self._axis.add_patch(self.get_plot_arrow())

        self._poly_patch = Polygon(
            self.polygon_coords[:, [1, 0]],
            closed=True,
            edgecolor="orange",
            facecolor="none",
            linewidth=4,
            label="Domain",
        )
        self._axis.add_patch(self._poly_patch)
        # Initialize the contourf plot with zeros inside the polygon area
        self._contour = self._axis.contourf(
            self._lon_grid,
            self._lat_grid,
            np.zeros(self._lat_grid.shape),
            transform=ccrs.PlateCarree(),
            **self._contourf_args,
        )

        self._contour.set_clip_path(self._poly_patch)

        self._cax = make_axes_locatable(self._axis).append_axes(
            "right", size="5%", pad=0.1, axes_class=plt.Axes
        )
        self._colorbar = self._figure.colorbar(
            self._contour, cax=self._cax, orientation="vertical"
        )
        self._axis.set_title(
            f"{self.water_feature_name} - ASV {self._asvid}",
            fontsize=14,
        )
        self._axis.legend()

    def magnetic_to_true_heading(self, latitude, longitude, magnetic_heading):
        """
        Convert magnetic heading to true heading based on location.

        Parameters:
            latitude (float): Latitude in degrees
            longitude (float): Longitude in degrees
            magnetic_heading (float): Magnetic heading in degrees (0-359)

        Returns:
            float: True heading in degrees (0-359)
        """
        
        mag_field = self._gm.calculate(latitude, longitude, alt=0, time=2025)  # altitude in meters
        declination = mag_field.d  # magnetic declination in degrees
        true_heading = magnetic_heading + declination

        # Normalize to 0-359 degrees
        true_heading = true_heading % 360

        return true_heading

    def plot_env_and_path(
        self, path: np.ndarray, next_point: np.ndarray, current_heading: float = 0.0
    ):
        """
        Plot the interpolated environment and the exploration path.
        """
        interpolated = self.function_to_call(self._mesh_points).reshape(
            self._lat_grid.shape
        )
        if self._contour is not None:
            try:
                self._contour.remove()
                self._cax.cla()
                self._current_point.remove()
            except AttributeError:
                pass
        self._contour = self._axis.contourf(
            self._lon_grid, self._lat_grid, interpolated,
            transform=ccrs.PlateCarree(), **self._contourf_args,
        )
        self._contour.set_clip_path(self._poly_patch)
        self._colorbar = self._figure.colorbar(
            self._contour, cax=self._cax, orientation="vertical"
        )
        self._path_plot.set_xdata(path[:-1, 1])
        self._path_plot.set_ydata(path[:-1, 0])
        self._next_point_plot.set_offsets([[next_point[1], next_point[0]]])
        current_arrow = self.get_plot_arrow(
            x=path[-1, 1],
            y=path[-1, 0],
            angle=self.magnetic_to_true_heading(path[-1, 0], path[-1, 1], current_heading) + 90,
        )
        self._current_point = self._axis.add_patch(current_arrow)
        # self._current_point.set_offsets([[path[-1, 1], path[-1, 0]]])

        self._colorbar.ax.set_ylabel("Interpolated Value")
        plt.draw()
        plt.pause(0.1)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pykrige.ok import OrdinaryKriging
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

# Load the data from the CSV files
file_path = 'out/20240604test3.csv'
path_path = 'out/20240604grad.csv'
water_feature = 'ODO (%Sat)'
data = pd.read_csv(file_path)
path_data = pd.read_csv(path_path)
data = data[(data['Latitude'] != 0.0) & (data['Longitude'] != 0.0)]
# Extract the extent coordinates
extent_coordinates = np.asarray(data[['Latitude', 'Longitude']])
# extent_coordinates = np.array([
#     [25.9097434,-80.1371339],
#     [25.9095974,-80.1368845],
#     [25.9094865,-80.1370132],
#     [25.9096831,-80.1372372],
#     [25.9097205,-80.1371808],
#     [25.9095456,-80.1369569],
#     [25.9096855,-80.1370159],
#     [25.9095818,-80.1371272]
# ])

MAXS, MINS = np.max(extent_coordinates, axis=0), np.min(extent_coordinates, axis=0)

# Print the number of data points before concatenation
print(len(data))

# Concatenate data
data = pd.concat([data, path_data])

# Extract longitude, latitude, and feature columns
longitude = data['Longitude']
latitude = data['Latitude']
feature = data[water_feature]

# Create a grid for longitude and latitude
DELTA = 0.00018
lon_grid, lat_grid = np.meshgrid(
    np.linspace(MINS[1] - DELTA, MAXS[1] + DELTA, 20),
    np.linspace(MINS[0] - DELTA, MAXS[0] + DELTA, 20)
)

# Perform Ordinary Kriging
OK = OrdinaryKriging(longitude, latitude, feature, variogram_model='linear',
                     verbose=False, enable_plotting=False)

# Interpolate temperature on the grid
z, ss = OK.execute('points', lon_grid.reshape(-1, 1), lat_grid.reshape(-1, 1))

# Prepare the map with Cartopy
tiler = cimgt.GoogleTiles(style='satellite')
transform = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={'projection': transform})
from matplotlib.ticker import ScalarFormatter

formatter = ScalarFormatter(useOffset=False)
formatter.set_scientific(False)


# Plot the path data
ax.plot(path_data['Longitude'], path_data['Latitude'], marker='x', linestyle='-.', transform=transform)

# Plot the contour plot
contour = ax.contourf(lon_grid, lat_grid, z.reshape(lon_grid.shape), cmap='coolwarm', levels=20, transform=transform, alpha=0.5)

# Add a colorbar
cbar = plt.colorbar(contour, label='Temperature')
cbar.ax.yaxis.set_major_formatter(formatter)
# Set aspect ratio and add the satellite image
ax.set_aspect('equal', adjustable='box')
ax.add_image(tiler, 21)

# Set the extent of the plot
plt.xlim((MINS[1] - DELTA, MAXS[1] + DELTA))
plt.ylim((MINS[0] - DELTA, MAXS[0] + DELTA))

# Add labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'{water_feature} Contour Plot')

# Show the plot
plt.show()

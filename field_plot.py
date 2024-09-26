import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pykrige.ok import OrdinaryKriging
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import argparse

def main(file_path, path_path, water_feature):
    # Load the data from the CSV files
    data = pd.read_csv(file_path)
    path_data = pd.read_csv(path_path)

    # Filter out invalid latitude and longitude values
    data = data[(data['Latitude'] != 0.0) & (data['Longitude'] != 0.0)]
    
    # Extract the extent coordinates
    extent_coordinates = np.asarray(data[['Latitude', 'Longitude']])
    MAXS, MINS = np.max(extent_coordinates, axis=0), np.min(extent_coordinates, axis=0)

    # Print the number of data points before concatenation
    print(f'Number of data points before concatenation: {len(data)}')

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

    # Interpolate feature on the grid
    z, ss = OK.execute('points', lon_grid.reshape(-1, 1), lat_grid.reshape(-1, 1))

    # Prepare the map with Cartopy
    tiler = cimgt.GoogleTiles(style='satellite')
    transform = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={'projection': transform})

    # Plot the path data
    ax.plot(path_data['Longitude'], path_data['Latitude'], marker='x', linestyle='-.', transform=transform)

    # Plot the contour plot
    contour = ax.contourf(lon_grid, lat_grid, z.reshape(lon_grid.shape), cmap='coolwarm', levels=20, transform=transform, alpha=0.5)

    # Add a colorbar
    plt.colorbar(contour, label=water_feature)
    
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

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Plot water feature contours.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file containing water feature data.')
    parser.add_argument('path_path', type=str, help='Path to the CSV file containing path data.')
    parser.add_argument('water_feature', type=str, default='ODO (%Sat)', help='Water feature to be plotted.')

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.file_path, args.path_path)

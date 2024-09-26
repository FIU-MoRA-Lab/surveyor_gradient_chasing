import surveyor_library.surveyor_helper as hlp
import surveyor_library
from gradient_chasing_utils import water_phenomenon
import argparse
import time
import sys
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.gaussian_process.kernels import DotProduct, Matern
import matplotlib.pyplot as plt


# Global Variables
DATA = pd.DataFrame()
THROTTLE = 30
FEATURE_TO_CHASE = 'ODO (%Sat)'

# Initialize Gaussian Process for water phenomenon
kernel = 10 * Matern(nu=0.5, length_scale_bounds=(1e-2, 1e5)) + 1e-2 * DotProduct() ** 1
extent_coordinates = np.loadtxt('out/polygon_coordinates_mmc.csv', delimiter=',', skiprows=1)
water_feature_gp = water_phenomenon.WaterPhenomenonGP(extent_coordinates, kernel)


def allocate_data_df(boat):
    """Allocate data from the boat to a DataFrame."""
    return pd.DataFrame([boat.get_data()])


def start_mission(boat):
    """Start the mission by waiting for the operator to switch to waypoint mode."""
    countdown(2, "Starting mission in")
    while boat.get_control_mode() != "Waypoint":
        boat.set_waypoint_mode()
    print('Mission started!')


def countdown(count, message):
    """
    Print a countdown with the given message.

    Args:
        count (int): The number of seconds to count down.
        message (str): The message to display before the countdown.
    """
    for i in range(count, 0, -1):
        print(f'{message} {i}.', end="\r")
        time.sleep(1)
    print()  


def plot_caller(boat, water_phenomenon, next_point):
    """Plot the current GPS coordinates and the next point."""
    plot_caller.coordinates.append(boat.get_gps_coordinates())
    water_phenomenon.plot_env_and_path(np.asarray(plot_caller.coordinates), next_point)
    return plot_caller.coordinates[-1]

plot_caller.coordinates = []  


def data_updater(boat, mission_postfix=''):
    """
    Update global DATA with new information from the boat.

    Args:
        boat: The boat object to get data from.
        mission_postfix (str): Optional postfix for mission-specific data saving.
    """
    global DATA
    data_dict = boat.get_data()
    DATA = pd.concat([DATA, pd.DataFrame([data_dict])])
    hlp.save(data_dict, mission_postfix)


def next_waypoint(step_size=4.5):
    """
    Calculate the next waypoint using Gaussian Process.

    Args:
        step_size (float): The distance to move towards the next waypoint.

    Returns:
        numpy.ndarray: The next waypoint coordinates.
    """
    global DATA
    X = np.asarray(DATA[['Latitude', 'Longitude']])
    y = np.asarray(DATA[FEATURE_TO_CHASE])
    water_feature_gp.fit(X, y)
    return water_feature_gp.next_point(X[-1], step_size)


def main(filename, erp_filename, mission_postfix=""):
    """
    Main function to execute the gradient chasing mission.

    Args:
        filename (str): The filename of the waypoints CSV.
        erp_filename (str): The filename of the ERP CSV.
        mission_postfix (str): Optional postfix for mission-specific data saving.
    """
    print(f'Reading waypoints from {filename} and ERP from {erp_filename}')
    initial_waypoints = hlp.read_csv_into_tuples(filename)
    erp = hlp.read_csv_into_tuples(erp_filename)
    boat = surveyor_library.Surveyor()
    
    print(f'{len(initial_waypoints)} initial waypoints')
    
    with boat:
        start_mission(boat)
        water_feature_gp.plot_initialization(delta=0.00015)

        for initial_waypoint in initial_waypoints:
            boat.go_to_waypoint(initial_waypoint, erp, THROTTLE)

            while boat.get_control_mode() == 'Waypoint':
                current_coordinates = plot_caller(boat, water_feature_gp, initial_waypoint)
                print(f'Initial collection mission waypoint {initial_waypoint}. '\
                      f'Distance {geodesic(current_coordinates, initial_waypoint).meters}', end="\r")
                plot_caller(boat, water_feature_gp, initial_waypoint)

            data_updater(boat, mission_postfix=mission_postfix, water_feature=FEATURE_TO_CHASE)  # Finished, getting data

        print('Starting gradient chasing')
        for i in range(30):
            waypoint = next_waypoint()
            print(f'Loading waypoint {i + 1}')
            boat.go_to_waypoint(waypoint, erp, THROTTLE)

            while boat.get_control_mode() == 'Waypoint':
                current_coordinates = plot_caller(boat, water_feature_gp, waypoint)
                print(f'Navigating to waypoint {i + 1} {waypoint}. '\
                      f'Distance {geodesic(current_coordinates, initial_waypoint).meters}', end="\r")
                

            data_updater(boat, mission_postfix=mission_postfix, water_feature=FEATURE_TO_CHASE)

    plt.show()  # Show the plot after the mission ends


if __name__ == "__main__":
    # Add arguments
    if len(sys.argv) == 1:
        print(f'Run {sys.argv[0]} -h  for help')
        sys.exit(0)
    parser = argparse.ArgumentParser(description='Gradient chasing script.')
    parser.add_argument('filename', type=str, help='Path to the main data CSV file.')
    parser.add_argument('erp_filename', type=str, help='Path to the ERP data CSV file.')
    parser.add_argument('--mission_postfix', type=str, default="", help='Optional postfix for the mission (default: empty).')

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.filename, args.erp_filename, args.mission_postfix)

import argparse
import sys
import time
import warnings
from test import mock_surveyor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import DotProduct, Matern
from tqdm import tqdm

import surveyor_library.surveyor_lib.helpers as hlp
import surveyor_library.surveyor_lib.surveyor as surveyor
import utils.send_data_utils
from utils import water_phenomenon

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Global Variables
from config import (ASVID, DATA, FEATURE_TO_CHASE, IS_SIMULATION,
                    NUM_WAYPOINTS, SEND_TO_MONGO, STEP_SIZE, THROTTLE,
                    PATH_PLOT_ARGS, CONTOURF_ARGS)


def start_mission(boat):
    """Start the mission by waiting for the operator to switch to waypoint mode."""
    countdown(2, "Starting mission in")
    while boat.get_control_mode() != "Waypoint":
        boat.set_waypoint_mode()
    print("Mission started!")


def countdown(count, message):
    """
    Print a countdown with the given message.

    Args:
        count (int): The number of seconds to count down.
        message (str): The message to display before the countdown.
    """
    for i in range(count, 0, -1):
        print(f"{message} {i}.", end="\r")
        time.sleep(1)
    print()


def initialize_boat(erp):
    if IS_SIMULATION:
        print("Running in simulation mode.")
        return mock_surveyor.MockSurveyor(erp[0])
    else:
        sensors_to_use = ["exo2"]
        sensors_config = {"exo2": {"server_ip": "192.168.0.20"}}
        return surveyor.Surveyor(
            sensors_to_use=sensors_to_use, sensors_config=sensors_config
        )


def plot_caller(boat, water_phenomenon, next_point):
    """Plot the current GPS coordinates and the next point."""
    current_coordinates = np.asarray(boat.get_gps_coordinates())
    plot_caller.coordinates = np.vstack((plot_caller.coordinates, current_coordinates))
    water_phenomenon.update_plot(
        plot_caller.coordinates, next_point, boat.get_data(["state"]).get("Heading (degrees Magnetic)", 0)
    )
    return plot_caller.coordinates[-1]


plot_caller.coordinates = np.empty((0, 2))

def save_and_send_data(boat, mission_postfix=""):
    
    hlp.save(boat.get_data(["exo2", "state"]), mission_postfix)
    if SEND_TO_MONGO:
        utils.send_data_utils.send_to_mongo(
            boat, asvid=ASVID, mission_postfix=mission_postfix
        )


def data_updater(boat, mission_postfix=""):
    
    """
    Update global DATA with new information from the boat.

    Args:
        boat: The boat object to get data from.
        mission_postfix (str): Optional postfix for mission-specific data saving.
    """
    global DATA
    data_dict = boat.get_data(["exo2", "state"])
    DATA = pd.concat([DATA, pd.DataFrame([data_dict])])
    save_and_send_data(boat, mission_postfix)


def next_waypoint(water_feature_gp, step_size=4.0):
    """
    Calculate the next waypoint using Gaussian Process.

    Args:
        step_size (float): The distance to move towards the next waypoint.

    Returns:
        numpy.ndarray: The next waypoint coordinates.
    """
    global DATA
    X = np.asarray(DATA[["Latitude", "Longitude"]])
    y = np.asarray(DATA[FEATURE_TO_CHASE])
    water_feature_gp.fit(X, y)
    return water_feature_gp.next_point(X[-1], step_size)


def process_waypoint(boat, water_feature_gp, waypoint, mission_postfix="", delay=0.5):
    """
    Process a single waypoint: update plot, log information, and send data to MongoDB if enabled.

    Args:
        boat: The boat object.
        water_feature_gp: The water phenomenon Gaussian Process object.
        waypoint (tuple): The current waypoint (lat, lon).
        mission_postfix (str): Optional postfix for mission-specific data saving.
    """
    current_coordinates = plot_caller(boat, water_feature_gp, waypoint)
    print(
        f"Waypoint {waypoint}. Distance {geodesic(current_coordinates, waypoint).meters:.3f}", end='\r'
    )
    save_and_send_data(boat, mission_postfix)
    time.sleep(delay)


def main(filename, erp_filename, extent_filename, mission_postfix=""):
    """
    Main function to execute the gradient chasing mission.

    Args:
        filename (str): The filename of the waypoints CSV.
        erp_filename (str): The filename of the ERP CSV.
        extent_filename (str): The filename of the extent coordinates CSV.
        mission_postfix (str): Optional postfix for mission-specific data saving.
    """
    print(
        f"Reading waypoints from {filename}, ERP from {erp_filename}, and extent coordinates from {extent_filename}"
    )
    initial_waypoints = np.asarray(hlp.read_csv_into_tuples(filename))
    erp = hlp.read_csv_into_tuples(erp_filename)
    extent_coordinates = np.loadtxt(extent_filename, delimiter=",", skiprows=1)

    # Initialize Gaussian Process for water phenomenon
    kernel = (
        10 * Matern(nu=0.5, length_scale_bounds=(1e-2, 1e5)) + 1e-2 * DotProduct() ** 1
    )
    water_feature_gp = water_phenomenon.WaterPhenomenonGP(extent_coordinates, kernel)

    boat = initialize_boat(erp)

    print(f"Initial mission has {len(initial_waypoints)} waypoints")

    with boat:
        # start_mission(boat)
        water_feature_gp.plotter.plot_initialization(delta=0.00015, plot_args=PATH_PLOT_ARGS,contourf_args=CONTOURF_ARGS)

        for initial_waypoint in tqdm(
            initial_waypoints, desc="Initial Collection Progress"
        ):
            boat.go_to_waypoint(initial_waypoint, erp, THROTTLE)

            while boat.get_control_mode() == "Waypoint":
                process_waypoint(
                    boat, water_feature_gp, initial_waypoint, mission_postfix, 0.5
                )

            data_updater(
                boat, mission_postfix=mission_postfix
            )  # Finished, getting data

        print("Starting gradient tracking mission...")
        for i in tqdm(range(NUM_WAYPOINTS), desc="Gradient Chasing Progress"):
            waypoint = next_waypoint(water_feature_gp, step_size=STEP_SIZE)
            print(f"Loading waypoint {i + 1}")
            boat.go_to_waypoint(waypoint, erp, THROTTLE)

            while boat.get_control_mode() == "Waypoint":
                process_waypoint(boat, water_feature_gp, waypoint, mission_postfix, 0.5)

            print(f"Arrived at waypoint {i + 1}.")
            data_updater(boat, mission_postfix=mission_postfix)
        plt.show()  # Show the plot after the mission ends


if __name__ == "__main__":
    # Add arguments
    if len(sys.argv) == 1:
        print(f"Run {sys.argv[0]} -h  for help")
        sys.exit(0)
    parser = argparse.ArgumentParser(description="Gradient chasing script.")
    parser.add_argument("filename", type=str, help="Path to the main data CSV file.")
    parser.add_argument("erp_filename", type=str, help="Path to the ERP data CSV file.")
    parser.add_argument(
        "extent_filename", type=str, help="Path to the extent coordinates CSV file."
    )
    parser.add_argument(
        "--mission_postfix",
        type=str,
        default="",
        help="Optional postfix for the mission (default: empty).",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    try:
        main(
            args.filename, args.erp_filename, args.extent_filename, args.mission_postfix
        )
    except KeyboardInterrupt:
        print("Mission interrupted by user.")
        sys.exit(0)

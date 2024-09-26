import surveyor_library.surveyor_helper as hlp
import surveyor_library
import gradient_chasing_utils
from gradient_chasing_utils import water_phenomenon

import sys
import time
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.gaussian_process.kernels import DotProduct, Matern
import matplotlib.pyplot as plt


def allocate_data_df(boat):
    return pd.DataFrame([boat.get_data()])

def start_mission(boat):
    """
    Start the mission by waiting for the operator to switch to waypoint mode.
    """
    countdown(2, "Starting mission in")
    while boat.get_control_mode() != "Waypoint":
        boat.set_waypoint_mode()
    print('Mission started!')

def countdown(count, message):
    """
    Print a countdown with the given message and optional additional message.

    Args:
        count (int): The number of seconds to count down.
        message (str): The message to display before the countdown.
    """
    for i in range(count, 0, -1):
        print(f'{message} {i}.', end="\r")
        time.sleep(1)
    print()  
 
def plot_caller(boat, water_phenomenon, next_point):
    plot_caller.coordinates.append(boat.get_gps_coordinates())
    water_phenomenon.plot_env_and_path(np.asarray(plot_caller.coordinates), next_point)
plot_caller.coordinates = []  

def data_updater(boat, water_feature, mission_postfix = ''):
    global DATA
    data_dict = boat.get_data()
    DATA = pd.concat([DATA,
                    pd.DataFrame([data_dict])])
    print(DATA[['Latitude', 'Longitude', water_feature]])
    hlp.save(data_dict, mission_postfix)

def next_waypoint(step_size = 4.5):
    global DATA
    X = np.asarray(DATA[['Latitude', 'Longitude']])
    y = np.asarray(DATA[FEATURE_TO_CHASE])
    water_feature_gp.fit(X, y)
    return water_feature_gp.next_point(X[-1], step_size)



# GP initialization
kernel = 10 * Matern(nu=0.5, length_scale_bounds=(1e-2, 1e5)) + 1e-2 * DotProduct() ** 1
extent_coordinates = np.loadtxt('out/polygon_coordinates_mmc.csv', delimiter=',', skiprows=1)

FEATURE_TO_CHASE = 'ODO (%Sat)'
water_feature_gp = water_phenomenon.WaterPhenomenonGP(extent_coordinates, kernel)
THROTTLE = 30; DATA = pd.DataFrame()
# plt.ion()

def main(filename, erp_filename, mission_postfix= ""):
    print(f'Reading waypoints from {filename} and ERP from {erp_filename}')
    initial_waypoints = hlp.read_csv_into_tuples(filename)
    erp = hlp.read_csv_into_tuples(erp_filename)
    # print(initial_waypoints, erp)
    boat = surveyor_library.Surveyor()
    
    print(f'{len(initial_waypoints)} initial waypoints')
    
    with boat:
        start_mission(boat)
        water_feature_gp.plot_initialization(delta = 0.0004)
        for initial_waypoint in initial_waypoints:
            boat.go_to_waypoint(initial_waypoint, erp, THROTTLE)

            while boat.get_control_mode() == 'Waypoint':
                print(f'Initial collection mission waypoint {initial_waypoint}', end="\r")
                plot_caller(boat, water_feature_gp, initial_waypoint )

            data_updater(boat, mission_postfix = mission_postfix, water_feature=FEATURE_TO_CHASE) # Finished, getting data

        print('Starting gradient chasing')
        for i in range(30):
            waypoint = next_waypoint()
            print(f'Loading waypoint {i + 1}')
            boat.go_to_waypoint(waypoint, erp, THROTTLE)

            while boat.get_control_mode() == 'Waypoint':
                print(f'Navigating to waypoint {i + 1}' ,waypoint, end="\r")
                plot_caller(boat, water_feature_gp, waypoint )
                # When you break this while loop you should have reached the waypoint, ready to assign a new waypoint

            data_updater(boat, mission_postfix = mission_postfix, water_feature=FEATURE_TO_CHASE)

    # plt.ioff() 
    plt.show()
            
if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage: gradient_chasing.py <filename> <erp_filename> <mission_postfix>")
        sys.exit(1)

    main(*sys.argv[1:])
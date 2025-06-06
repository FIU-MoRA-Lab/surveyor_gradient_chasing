import random
import threading
import time

import numpy as np
from geopy.distance import geodesic


def mock_ODO_generator(maximum_location=(0, 0), max_value=100, min_value=50, decay=0.1):
    """ """

    def mock_ODO(x):
        """
        Calculate the mock ODO value based on the distance from the maximum location.
        """
        distance = geodesic(maximum_location, x).meters
        value = min_value + (max_value - min_value) * np.exp(-decay * distance)
        return value

    return mock_ODO


class MockSurveyor:
    def __init__(self, current_location, odo_source=(25.913112, -80.138074)):
        """
        Initialize the MockSurveyor with the current location.

        Args:
            current_location (tuple): Initial coordinates (latitude, longitude).
        """
        self.current_location = np.asarray(current_location)
        print(f"MockSurveyor initialized at {self.current_location}.")
        self.control_mode = "Station Keep"
        self._moving = False
        self._odo_gen = mock_ODO_generator(
            maximum_location=odo_source, max_value=100, min_value=50, decay=0.0001
        )

    def _get_data(self):
        """
        Get mock data including ODO (%Sat), Temperature, and coordinates.

        Returns:
            dict: Mock data with ODO (%Sat), Temperature, and coordinates.
        """
        return {
            "ODO (%Sat)": self._odo_gen(self.current_location),
            "Temperature (C)": round(random.uniform(15, 30), 2),
            "Chlorophyll (ug/L)": round(random.uniform(0, 10), 2),
            "Turbidity (NTU)": round(random.uniform(0, 5), 2),
            "Latitude": self.current_location[0],
            "Longitude": self.current_location[1],
            "Date": time.strftime("%Y%m%d"),
            "Time": time.strftime("%H%M%S"),
            "Velocity": 0.0,  # Mock velocity
            "Heading (degrees Magnetic)": round(random.uniform(0, 90), 2),  # Mock heading
            "Acceleration x, forward (G)": 0.0,  # Mock acceleration
        }

    def get_data(self, list):
        data_dict = {}
        data = self._get_data()
        for list_item in list:
            if list_item == "exo2":
                data_dict["ODO (%Sat)"] = data["ODO (%Sat)"]
                data_dict["Temperature (C)"] = data["Temperature (C)"]
                data_dict["Chlorophyll (ug/L)"] = data["Chlorophyll (ug/L)"]
                data_dict["Turbidity (NTU)"] = data["Turbidity (NTU)"]
            elif list_item == "state":
                for key in data:
                    if key not in [
                        "ODO (%Sat)",
                        "Temperature (C)",
                        "Chlorophyll (ug/L)",
                        "Turbidity (NTU)",
                    ]:
                        data_dict[key] = data[key]
        return data_dict

    def get_gps_coordinates(self):
        """
        Get the current GPS coordinates of the boat.

        Returns:
            tuple: Current coordinates (latitude, longitude).
        """
        # print(self.current_location[0])
        return float(self.current_location[0]), float(self.current_location[1])

    def go_to_waypoint(self, waypoint, erp, THROTTLE):
        """
        Set the target waypoint and throttle.

        Args:
            waypoint (tuple): Target coordinates (latitude, longitude).
            THROTTLE (int): Speed of the boat.
        """
        self.waypoint = np.array(waypoint)
        # print(f"Setting waypoint to {self.waypoint}.")
        self.THROTTLE = THROTTLE
        print(f"Setting waypoint to {self.waypoint} with throttle {self.THROTTLE}.")
        if self.control_mode != "Waypoint":
            self.control_mode = "Waypoint"

    def station_keep(self):
        """
        Set the control mode to Station Keep (stop moving).
        """
        self.control_mode = "Station Keep"

    def _move_loop(self):
        """
        Continuously move towards the current waypoint if in Waypoint mode.
        """
        radius = 2  # meters
        while True:
            # print('moving')
            if getattr(self, "control_mode", None) == "Waypoint" and hasattr(
                self, "waypoint"
            ):
                speed = getattr(self, "THROTTLE", 0) / 20  # meters per second
                distance = geodesic(self.current_location, self.waypoint).meters
                if distance <= radius:
                    print(
                        f"Reached waypoint {self.current_location}. {distance} meters away."
                    )
                    self.control_mode = "Station Keep"
                else:
                    direction = self.waypoint - self.current_location
                    direction /= np.linalg.norm(direction)
                    self.current_location += direction * (speed / 111139)
            time.sleep(1.1)

    def start(self):
        """
        Start the move loop in a background thread.
        """
        # print("Starting MockSurveyor move loop.", self._move_thread)
        if not hasattr(self, "_move_thread"):
            self._move_thread = threading.Thread(target=self._move_loop, daemon=True)
            self._move_thread.start()

    def get_control_mode(self):
        """
        Get the current control mode of the boat.

        Returns:
            str: Current control mode.
        """
        return self.control_mode

    def __enter__(self):
        """
        Enter the context manager.

        Returns:
            MockSurveyor: The instance of the MockSurveyor.
        """
        print("MockSurveyor initialized.")
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context manager.

        Args:
            exc_type: Exception type.
            exc_value: Exception value.
            traceback: Traceback object.
        """
        print("MockSurveyor shutting down.")
        self._moving = False

import numpy as np

LATITUDE_TO_METERS = 111320


class Normalizer:
    """
    Normalizes and denormalizes geographic coordinates (latitude, longitude)
    to and from a local metric (meter) coordinate system, using a specified origin.

    Args:
        origin (np.ndarray or list, optional): The reference point (latitude, longitude)
            for normalization. Defaults to [25.7581572, -80.3734494].

    Attributes:
        origin (np.ndarray): The reference latitude and longitude.
        scale (np.ndarray): Scaling factors (meters per degree) for latitude and longitude.

    Methods:
        forward(x):
            Converts latitude and longitude coordinates to local metric coordinates (meters)
            relative to the origin.

        inverse(x):
            Converts local metric coordinates (meters) back to latitude and longitude
            relative to the origin.
    """

    def __init__(self, origin=np.array([25.7581572, -80.3734494])):
        self.origin = origin
        self.scale = LATITUDE_TO_METERS * np.array([1, np.cos(np.radians(origin[0]))])

    def forward(self, x):
        """
        Converts latitude and longitude coordinates to local metric coordinates (meters)
        relative to the origin.

        Args:
            x (array-like): Input coordinates as [latitude, longitude] or an array of such pairs.

        Returns:
            np.ndarray: Normalized coordinates in meters relative to the origin.
        """
        x = np.asarray(x)
        return self.scale * (x - self.origin)

    def inverse(self, x):
        """
        Converts local metric coordinates (meters) back to latitude and longitude
        relative to the origin.

        Args:
            x (array-like): Normalized coordinates in meters.

        Returns:
            np.ndarray: Denormalized coordinates as [latitude, longitude].
        """
        x = np.asarray(x)
        return (x / self.scale) + self.origin

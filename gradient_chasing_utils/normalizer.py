import numpy as np

LATITUDE_TO_METERS = 111320

class Normalizer:
    """
    Normalizes and denormalizes coordinates. 
    It manages coordinates in the format (Latitude, Longitude)

    Args:
        mins_maxs (tuple): Tuple containing the minimum values of the coordinates.
        maxs (tuple): Tuple containing the maximum values of the coordinates.
        dom_maxs (numpy.ndarray): Maximum values of the domain.

    Attributes:
        mins (numpy.ndarray): Minimum values of the coordinates.
        maxs (numpy.ndarray): Maximum values of the coordinates.
        dom_maxs (numpy.ndarray): Maximum values of the domain.
    """
    def __init__(self, origin = np.array([25.7581572, -80.3734494])):
        self.origin = origin
        self.scale = LATITUDE_TO_METERS * np.array([1, np.cos(np.radians(origin[0]))])

    def forward(self, x):
        """
        Forward normalization. (From coordinates domain to transformed domain)

        Args:
        - x: Input coordinates.

        Returns:
        - Normalized coordinates.
        """
        x = np.asarray(x)
        return self.scale * (x - self.origin)
    
    def inverse(self, x):
        """
        Inverse normalization.

        Args:
        - x: Normalized coordinates.

        Returns:
        - Inverse normalized coordinates.
        """
        x = np.asarray(x)
        return (x / self.scale) + self.origin
import pandas as pd

DATA = pd.DataFrame()
THROTTLE = 20
FEATURE_TO_CHASE = "ODO (%Sat)"
IS_SIMULATION = False
ASVID = 17
NUM_WAYPOINTS = 30
STEP_SIZE = 5.0
SEND_TO_MONGO = True

# Plot configurations
PATH_PLOT_ARGS = {"color": "black", "linestyle": "--"}
CONTOURF_ARGS = {"cmap": "viridis",
    "levels": 15,}


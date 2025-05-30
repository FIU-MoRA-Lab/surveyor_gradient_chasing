import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrow

fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Initial state
x, y = 5, 5
angle_deg = 0
length = 0.1

# Create initial arrow
dx = length * np.cos(np.radians(angle_deg))
dy = length * np.sin(np.radians(angle_deg))
arrow = FancyArrow(x, y, dx, dy, width=0.1, color="green")
arrow_patch = ax.add_patch(arrow)


# Update function
def update(frame):
    global arrow_patch
    angle = (frame * 10) % 360
    dx = length * np.cos(np.radians(angle))
    dy = length * np.sin(np.radians(angle))

    # Remove old arrow and add a new one
    arrow_patch.remove()
    arrow_patch = FancyArrow(x, y, dx, dy, width=0.1, color="green")
    ax.add_patch(arrow_patch)


ani = FuncAnimation(fig, update, frames=range(36), interval=100)
plt.show()

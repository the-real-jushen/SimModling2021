#%%
import numpy as np
import matplotlib.pyplot as plt

def plot_ball_motion(v, h):
    g = 9.81  # acceleration due to gravity (m/s^2)
    v_vertical = 0  # initial vertical velocity
    t = 0  # initial time
    dt = 0.01  # time step
    positions = [(0, h)]  # initial position

    while True:
        t += dt
        h = h - v_vertical * dt - 0.5 * g * dt**2
        v_vertical -= g * dt
        if h <= 0:
            h = 0
            v_vertical *= -0.7  # reduce vertical speed by 0.7 on bounce
            if abs(v_vertical) < 8:
                break
        positions.append((v * t, h))

    x, y = zip(*positions)
    plt.plot(x, y)
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Height (m)')
    plt.title('Ball Motion')
    plt.grid(True)
    plt.show()

# Example usage
initial_speed = 10  # m/s
initial_height = 5  # meters
plot_ball_motion(initial_speed, initial_height)

# %%

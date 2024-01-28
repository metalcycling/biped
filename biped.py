"""
Simulation of a biped walking
"""

# %% Modules

import numpy as np
import matplotlib.pyplot as plt

from spring import spring
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# %% Constants

k = 2.0e3 # N/m
l_0 = 2.0 # m
m = 80 # kg
g = 9.8 # m / s^2

# %% Functions

def apex_1(t, pop):
    """
    First leg is anchered
    """
    u_1, u_2, u_3, u_4 = pop

    u_1_p = u_2
    u_2_p = k * (l_0 / np.sqrt(u_1 ** 2.0 + u_3 ** 2.0) - 1.0) * u_1
    u_3_p = u_4
    u_4_p = k * (l_0 / np.sqrt(u_1 ** 2.0 + u_3 ** 2.0) - 1.0) * u_3 - m * g

    return [u_1_p, u_2_p, u_3_p, u_4_p]

# %% Main function

if __name__ == "__main__":
    t_start, t_stop = [0.0, 1.8]
    num_steps = 1000
    t = np.linspace(t_start, t_stop, num_steps)
    x, y = (0.0, l_0)
    v_x, v_y = (1.0, 0.0)

    solve = solve_ivp(apex_1, t_span = [t_start, t_stop], y0 = [x, v_x, y, v_y], t_eval = t)
    x = solve.y[0]
    y = solve.y[2]

    num_cycles = 41
    width = 0.25
    line_width = 2

    fig, ax = plt.subplots(figsize = (12, 8))
    anchor_point = ax.scatter([], [], color = "red", s = 100, zorder = 5)
    mass_center = ax.scatter([], [], color = "blue", s = 1000, zorder = 5)
    spring_line, = ax.plot([], [], color = "black", linewidth = line_width, zorder = 3)

    def init():
        canvas_width = 4.0 * l_0
        ax.set_xlim(-canvas_width, canvas_width)
        ax.set_ylim(-canvas_width, canvas_width)
        ax.axis("equal")
        return spring_line,

    def update(fdx):
        spring_coords = spring(0.0, 0.0, x[fdx], y[fdx], num_cycles, width)
        spring_line.set_data(spring_coords[0], spring_coords[1])

        anchor_point.set_offsets([0.0, 0.0])
        mass_center.set_offsets([spring_coords[0][-1], spring_coords[1][-1]])

        return [spring_line, anchor_point, mass_center]
    
    animation = FuncAnimation(fig, update, frames = np.arange(t.shape[0]), init_func = init, blit = True, interval = 14)
    plt.show()

# %% End of program

# %% Modules

import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt

# %% Functions

def spring(x_start, y_start, x_end, y_end, num_cycles, width):
    """
    Compute the coordinates of the points to draw a string that goes from
    (x_start, y_start) to (x_end, y_end). The string has 'num_cycles' twists
    in the middle.
    """

    # Check that num_cycles is at least 1.

    assert num_cycles > 0, "'num_cycles' must be greater than 0"

    # Convert to numpy array to account for inputs of different types/shapes.

    start = np.array([x_start, y_start])
    end = np.array([x_end, y_end])

    # If both points are coincident, return the x and y coords of one of them.

    if (start == end).all():
        return start

    # Calculate length of spring (distance between endpoints).

    length = npla.norm(end - start)

    # Calculate unit vectors tangent (u_t) and normal (u_t) to spring.

    u_t = (end - start) / length
    u_n = np.array([[0.0, -1.0], [1.0, 0.0]]) @ u_t

    # Initialize array of x (row 0) and y (row 1) coords of the num_cycles + 2 points.

    spring_coords = np.zeros((2, num_cycles + 2))
    spring_coords[:, 0], spring_coords[:, -1] = start, end

    # Check that length is not greater than the total length the spring
    # can extend (otherwise, math domain error will result), and compute the
    # normal distance from the centerline of the spring.

    normal_dist = np.sqrt(max(0.0, width ** 2.0 - (length ** 2.0 / num_cycles ** 2.0))) / 2.0

    # Compute the coordinates of each point (each node).

    for i in range(1, num_cycles + 1):
        spring_coords[:,i] = (start + ((length * (2.0 * i - 1) * u_t) / (2.0 * num_cycles)) + (normal_dist * ((-1.0) ** i) * u_n))

    return spring_coords[0], spring_coords[1]

# %% Testing

if __name__ == "__main__":
    x_start, y_start = (0.0, 0.0)
    x_end, y_end = (0.5, 1.0)
    num_cycles = 20
    width = 0.1

    x_spring, y_spring = spring(x_start, y_start, x_end, y_end, num_cycles, width)

    title_size = 18
    label_size = 16
    ticks_size = 13
    line_width = 2
    marker_size = 380

    plt.figure(figsize = (12, 8))
    plt.title("Spring drawing", fontsize = title_size)
    plt.xlabel("x-coords", fontsize = label_size)
    plt.ylabel("y-coords", fontsize = label_size)
    plt.scatter(x_start, y_start, s = marker_size, color = "red", zorder = 5)
    plt.scatter(x_end, y_end, s = marker_size, color = "blue", zorder = 5)
    plt.plot(x_spring, y_spring, color = "black", linewidth = line_width, zorder = 3)
    plt.axis("equal")
    plt.xticks(fontsize = ticks_size)
    plt.yticks(fontsize = ticks_size)
    plt.show()

# %% End of program

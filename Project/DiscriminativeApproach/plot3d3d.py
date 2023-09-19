import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
if __name__ == "__main__":

    dict1 ={(10**-5, 10**-3): 0.5164608068442819, (10**-5, 10**-2): 0.5164608068442819, (10**-5, 10**-1): 0.5164608068442819, (10**-3, 10**-3): 0.48196949170780545, (10**-3, 10**-2): 0.38563674163191264, (10**-3, 10**-1): 0.38563674163191264, (10**-1, 10**-3): 0.38563674163191264, (10**-1, 10**-2): 0.38563674163191264, (10**-1, 10**-1): 0.38563674163191264, (10**1, 10**-3): 0.38563674163191264, (10**1, 10**-2): 0.38563674163191264, (10**1, 10**-1): 0.38563674163191264, (10**3, 10**-3): 0.38563674163191264, (10**3, 10**-2): 0.38563674163191264, (10**3, 10**-1): 0.38563674163191264, (10**5, 10**-3): 0.38563674163191264, (10**5, 10**-2): 0.38563674163191264, (10**5, 10**-1): 0.38563674163191264}

    # Extract x and y values
    x_values = np.array([key[0] for key in dict1.keys()])
    y_values = np.array([key[1] for key in dict1.keys()])
    z_values = np.array(list(dict1.values()))

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the 3D histogram
    hist, xedges, yedges = np.histogram2d(x_values, y_values, bins=10,
                                          range=[[min(x_values), max(x_values)], [min(y_values), max(y_values)]])
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.1, yedges[:-1] + 0.1, indexing="ij")

    # Flatten the 2D histogram
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.1
    dz = hist.ravel()

    # Plot the 3D histogram
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

    # Set axis labels
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    ax.set_zlabel('Frequency')

    # Show the plot
    plt.show()

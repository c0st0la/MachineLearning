import matplotlib.pyplot as plt
import numpy
if __name__ == "__main__":
    applicationWorkingPoint1 = numpy.array([9 / 10, 1 / 10], dtype=float)
    applicationWorkingPoint2 = numpy.array([5 / 10, 5 / 10], dtype=float)
    applicationWorkingPoint3 = numpy.array([1 / 10, 9 / 10], dtype=float)
    dict1={(10**-5, 10**-3): 0.2708982754724667, (10**-5, 10**-2): 0.22511829851413237, (10**-5, 10**-1): 0.15794438366883473, (10**-3, 10**-3): 0.08841034978606155, (10**-3, 10**-2): 0.08841034978606155, (10**-3, 10**-1): 0.08841034978606155, (10**-1, 10**-3): 0.08841034978606155, (10**-1, 10**-2): 0.08841034978606155, (10**-1, 10**-1): 0.08841034978606155, (10**1, 10**-3): 0.08841034978606155, (10**1, 10**-2): 0.08841034978606155, (10**1, 10**-1): 0.08841034978606155, (10**3, 10**-3): 0.08841034978606155, (10**3, 10**-2): 0.08841034978606155, (10**3, 10**-1): 0.08841034978606155, (10**5, 10**-3): 0.08841034978606155, (10**5, 10**-2): 0.08841034978606155, (10**5, 10**-1): 0.08841034978606155}


    x_values = sorted(list(set(key[0] for key in dict1.keys())))
    y_values = sorted(list(set(key[1] for key in dict1.keys())))
    x=10**-1
    # Convert x and y values to float
    x_values = [float(x) for x in x_values]
    y_values = [float(y) for y in y_values]

    # Create a 2D grid of values
    grid = numpy.zeros((len(y_values), len(x_values)))

    for i, x_val in enumerate(x_values):
        for j, y_val in enumerate(y_values):
            grid[j, i] = dict1.get((x_val, y_val))  # Fill the grid with dictionary values

    # Create a heatmap using plt.imshow
    plt.imshow(grid, cmap='viridis')
    plt.grid(True, color='white', linestyle='--', linewidth=0.5)
    plt.xticks(range(len(x_values)), x_values)
    plt.yticks(range(len(y_values)), y_values)
    plt.colorbar(label='Values')  # Add a colorbar for reference

    # Set axis labels
    plt.xlabel('C')
    plt.ylabel('Gamma')
    plt.savefig("./figures/RbSVM_Raw_Pt0_5")
    # Show the plot

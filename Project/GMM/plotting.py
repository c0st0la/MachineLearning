import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":

    x = ("1", "2", "4","8","16","32","64")
    xx= np.arange(len(x))  # the label locations
    # gmm diagonal
    # dict1 = {
    #     'raw': (0.967, 0.97, 0.715, 0.521,0.43,0.333,0.308),
    #     'zscore': (0.967, 0.97, 0.715, 0.521,0.416,0.342,0.302)
    # }

    # # gmm
    # dict1 = {
    #     'raw': (0.084, 0.083, 0.088, 0.1,0.109,0.129,0.15),
    #     'zscore': (0.084, 0.089, 0.088, 0.096,0.103,0.11,0.118)
    # }
    #gmm tied
    # dict1 = {
    #      'raw': (0.084, 0.084, 0.092, 0.09,0.092,0.091,0.097),
    #      'zscore': (0.084, 0.084, 0.096, 0.091,0.092,0.092,0.101)
    #  }
    #gmm tied diagonal
    dict1 = {
        'raw': (0.967, 0.958, 0.705, 0.498, 0.395, 0.348, 0.290),
        'zscore': (0.967, 0.958, 0.706, 0.498, 0.395, 0.349, 0.292)
    }

    mass = 0
    width = 0.25
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in dict1.items():
        mass = max(max(measurement), mass)
        offset = width * multiplier
        rects = ax.bar(xx + offset, measurement, width, label=attribute)
        multiplier += 1.2



    ax.set_ylabel('minDCF')
    ax.set_title('GMM Tied Diagonal')
    ax.set_xlabel('GMM Tied Diagonal Component')
    ax.set_xticks(xx +0.15, x)
    ax.legend(loc='upper right', ncols=6)
    ax.set_ylim(0, mass+0.05)
    plt.savefig("./figures/GmmDiagonalTied")

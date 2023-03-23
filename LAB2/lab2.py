import numpy
import matplotlib.pyplot as plt
from itertools import combinations

FILEPATH = 'Solution/iris.csv'


def to_column(array):
    return array.reshape((array.size, 1))


def load_iris_datasets_from_file(filename):
    """

    :param filename: the csv file
    :return: It returns a pair. The first element is the dataset, the second one is the labels
    """
    class_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    sample_mapping = list()
    D_list = list()
    with open(filename, "r") as fp:
        for line in fp:
            attrs = line.rstrip().split(',')[0:4]
            D_list.append(to_column(numpy.array([float(i) for i in attrs])))
            sample_mapping.append(class_mapping[line.rstrip().split(',')[4]])
    sample_mapping = numpy.array(sample_mapping, dtype=numpy.int32)
    return numpy.hstack(D_list), sample_mapping


def filter_dataset_by_labels(D, labels):
    """

    :param D: It is the dataset to filter
    :param labels: It is the array containing the labels of the dataset
    :return: the dataset D filtered by the labels provided
    """
    mask_setosa = (labels == 0)
    mask_versicolor = (labels == 1)
    mask_virginica = (labels == 2)
    return D[:, mask_setosa], D[:, mask_versicolor], D[:, mask_virginica]


def plot_hist_attributes_X_label(D_setosa, D_versicolor, D_virginica, attributes):
    """
    It plots the datasets using histograms plotting type for each of the attribute present in the
    attributes parameter
    :param D1_setosa:
    :param D1_versicolor:
    :param D1_virginica:
    :param attributes: It contains the attribute of the dataset
    :return: None
    """
    for i in range(len(attributes)):
        plt.figure()
        plt.hist(D_setosa[i, :], bins=10, ec='black', density=True, alpha=0.3, label='Iris_Setosa')
        plt.hist(D_versicolor[i, :], bins=10, ec='black', density=True, alpha=0.3, label='Iris_Versicolor')
        plt.hist(D_virginica[i, :], bins=10, ec='black', density=True, alpha=0.3, label='Iris_Virginica')
        plt.legend()
        plt.xlabel(attributes[i])
        plt.show()


def plot_scatter_pair_attributes_values(D_setosa, D_versicolor, D_virginica, attributes):
    comb = list(combinations([i for i in range(len(attributes))], 2))
    for x, y in comb:
        plt.figure()
        plt.scatter(D_setosa[x, :], D_setosa[y, :], label= 'Iris_Setosa')
        plt.scatter(D_versicolor[x, :], D_versicolor[y, :], label= 'Iris_Versicolor')
        plt.scatter(D_virginica[x, :], D_virginica[y, :], label='Iris_Virginica')
        plt.legend()
        plt.xlabel(attributes[x])
        plt.ylabel(attributes[y])
        plt.show()


def center_data(D):
    mu = to_column(D.mean(axis=1))
    DC = D - mu
    return DC


if __name__ == '__main__':
    attributes = [
        'Sepal length',
        'Sepal width',
        'Petal length',
        'Petal width'
    ]
    D, L = load_iris_datasets_from_file(FILEPATH)
    D_setosa, D_versicolor, D_virginica = filter_dataset_by_labels(D, L)
    plot_hist_attributes_X_label(D_setosa, D_versicolor, D_virginica, attributes)
    plot_scatter_pair_attributes_values(D_setosa, D_versicolor, D_virginica, attributes)
    DC = center_data(D)
    DC_setosa, DC_versicolor, DC_virginica = filter_dataset_by_labels(DC, L)
    plot_hist_attributes_X_label(DC_setosa, DC_versicolor, DC_virginica, attributes)
